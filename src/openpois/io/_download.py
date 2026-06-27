"""Resilient, resumable HTTP download shared by the OSM snapshot and history
loaders.

Two problems motivate this module:

* **Per-connection throttling.** Geofabrik's free download server caps each
  connection at roughly 1 MB/s but does *not* tightly cap per IP (two
  connections measured ~2.4 MB/s combined vs ~0.9 MB/s single). Splitting a
  download across a few parallel byte-range connections therefore multiplies
  throughput on the multi-GB extracts (~12 GB snapshot, ~23 GB full-history).
* **Flaky TLS streams.** Long transfers intermittently die mid-stream with
  ``SSL: RECORD_LAYER_FAILURE`` (observed across multiple unrelated hosts, i.e.
  a local-link issue). A single-shot GET that discards its partial file on any
  error restarts a multi-hour download from zero.

``download_resilient`` handles both:

* It probes for HTTP range support and, when available, downloads ``n_segments``
  contiguous byte ranges concurrently, each to its own ``<output>.partN`` file.
* Every segment resumes from the bytes already on disk (``Range: bytes=have-``)
  and retries connection-level failures with exponential backoff, so both a
  transient drop *and* a full process restart resume rather than restart.
* Segments are concatenated into ``<output>.part`` and atomically renamed into
  place, so a partial never masquerades as a finished download.
* If the server ignores range requests, it falls back to a single resumable
  stream with the same retry/resume behaviour.

``n_segments`` is deliberately modest (default 4) to stay within Geofabrik's
fair-use tolerance for parallel connections.
"""

from __future__ import annotations

import http.cookiejar
import mmap
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

# Connection-level failures a byte-range resume can recover from. HTTP status
# errors (403 expired cookie, 404 missing extract) are intentionally absent so
# they propagate immediately instead of retrying.
_RETRYABLE = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.SSLError,
    requests.exceptions.Timeout,
)


# Force writeback + drop the page cache this often during a large sequential
# write. WSL2 caps the VM at 24 GB; without this, the page cache for a 12-24 GB
# write grows unbounded and OOM-panics the whole VM (observed twice). 256 MB
# keeps peak cache to a few hundred MB regardless of file size.
_WRITEBACK_INTERVAL = 256 * 1024 * 1024

# O_DIRECT alignment unit. Buffers, file offsets, and write lengths must all be
# multiples of the device logical block size; the 4 KiB page size is a safe
# superset of the usual 512 B and matches the mmap page alignment we rely on.
_BLOCK = 4096


class _RangeUnsupported(Exception):
    """Raised when the server ignores a Range header (answers 200, not 206)."""


def _drop_cache(fd: int, offset: int, length: int) -> None:
    """Evict page-cache pages for ``[offset, offset+length)`` if supported.

    ``length`` of 0 is treated as a no-op here (callers pass explicit lengths);
    a no-op fallback is used on platforms without ``posix_fadvise``.
    """
    if length <= 0:
        return
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def _new_session(cookie_file: Path | str | None) -> requests.Session:
    """Fresh session (one per thread; Session isn't safe for concurrent use).

    Loads a Netscape/Mozilla cookie jar when ``cookie_file`` is given, matching
    the authenticated Geofabrik internal-server flow.
    """
    session = requests.Session()
    if cookie_file is None:
        return session
    cookie_path = Path(cookie_file).expanduser()
    if not cookie_path.exists():
        raise FileNotFoundError(
            f"Geofabrik cookie file not found: {cookie_path}. Generate one by "
            "logging in at https://osm-internal.download.geofabrik.de/ and "
            "exporting cookies, or run Geofabrik's oauth_cookie_client.py."
        )
    jar = http.cookiejar.MozillaCookieJar(str(cookie_path))
    jar.load(ignore_discard=True, ignore_expires=True)
    session.cookies = jar
    return session


class _Progress:
    """Thread-safe aggregate byte counter with throttled stdout reporting."""

    def __init__(self, total: int, label: str, initial: int = 0):
        self.total = total
        self.label = label
        self.done = initial
        self._lock = threading.Lock()
        self._last_print = 0.0

    def add(self, n: int) -> None:
        with self._lock:
            self.done += n
            now = time.time()
            if now - self._last_print >= 5 or self.done >= self.total:
                self._last_print = now
                pct = 100 * self.done / self.total if self.total else 0
                print(
                    f"  {self.label}: {self.done / 1e9:.2f}/"
                    f"{self.total / 1e9:.2f} GB ({pct:.1f}%)",
                    flush=True,
                )


def _probe(
    url: str, cookie_file, connect_timeout: int, read_timeout: int
) -> tuple[int, bool]:
    """Return ``(total_bytes, range_supported)`` via a one-byte range GET.

    A ``206`` with a ``Content-Range`` total is the reliable signal — some
    servers don't advertise ``Accept-Ranges`` on HEAD but still honour ranges.
    """
    session = _new_session(cookie_file)
    try:
        with session.get(
            url,
            headers={"Range": "bytes=0-0"},
            stream=True,
            allow_redirects=True,
            timeout=(connect_timeout, read_timeout),
        ) as resp:
            if resp.status_code == 206:
                content_range = resp.headers.get("content-range", "")
                total = (
                    int(content_range.split("/")[-1])
                    if "/" in content_range
                    else 0
                )
                return total, True
            resp.raise_for_status()
            return int(resp.headers.get("content-length", 0)), False
    finally:
        session.close()


def _download_segment(
    url: str,
    cookie_file,
    seg_path: Path,
    start: int,
    end: int,
    progress: _Progress,
    max_retries: int,
    chunk_size: int,
    connect_timeout: int,
    read_timeout: int,
) -> None:
    """Download the inclusive byte range [start, end] into ``seg_path``.

    Resumes from whatever is already in ``seg_path`` and retries connection
    errors. Raises ``_RangeUnsupported`` if the server answers 200 (no ranges).
    """
    seg_len = end - start + 1
    attempt = 0
    while True:
        have = seg_path.stat().st_size if seg_path.exists() else 0
        if have >= seg_len:
            return
        range_start = start + have
        session = _new_session(cookie_file)
        try:
            with session.get(
                url,
                headers={"Range": f"bytes={range_start}-{end}"},
                stream=True,
                allow_redirects=True,
                timeout=(connect_timeout, read_timeout),
            ) as resp:
                if resp.status_code == 200:
                    raise _RangeUnsupported()
                resp.raise_for_status()
                with open(seg_path, "ab") as f:
                    fd = f.fileno()
                    pos = os.fstat(fd).st_size  # resume-aware end of file
                    drop_from = pos
                    since_flush = 0
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        n = len(chunk)
                        pos += n
                        since_flush += n
                        progress.add(n)
                        if since_flush >= _WRITEBACK_INTERVAL:
                            f.flush()
                            os.fdatasync(fd)
                            _drop_cache(fd, drop_from, pos - drop_from)
                            drop_from = pos
                            since_flush = 0
                    f.flush()
                    os.fdatasync(fd)
                    _drop_cache(fd, drop_from, pos - drop_from)
            return
        except _RETRYABLE as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(min(60, 2 ** attempt))
            print(
                f"\n  segment {seg_path.name}: transient error "
                f"({type(exc).__name__}); retry {attempt}/{max_retries}",
                flush=True,
            )
        finally:
            session.close()


def _single_stream_resumable(
    url: str,
    output_path: Path,
    cookie_file,
    label: str,
    max_retries: int,
    chunk_size: int,
    connect_timeout: int,
    read_timeout: int,
) -> Path:
    """Single-connection resumable download (fallback when ranges are absent)."""
    part_path = output_path.with_name(output_path.name + ".part")
    have = part_path.stat().st_size if part_path.exists() else 0
    if have:
        print(f"Resuming {label}: {have / 1e9:.2f} GB already present")
    attempt = 0
    while True:
        headers = {"Range": f"bytes={have}-"} if have else {}
        session = _new_session(cookie_file)
        try:
            with session.get(
                url,
                stream=True,
                headers=headers,
                allow_redirects=True,
                timeout=(connect_timeout, read_timeout),
            ) as resp:
                if resp.status_code == 416:
                    break
                if have and resp.status_code == 200:
                    have = 0  # server ignored Range; restart from zero
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                total += have if resp.status_code == 206 else 0
                mode = "ab" if (have and resp.status_code == 206) else "wb"
                if mode == "wb":
                    have = 0
                with open(part_path, mode) as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        have += len(chunk)
                        if total:
                            print(f"  {100 * have / total:.1f}%", end="\r")
            break
        except _RETRYABLE as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            wait = min(60, 2 ** attempt)
            have = part_path.stat().st_size if part_path.exists() else 0
            print(
                f"\n  transient download error ({type(exc).__name__}); retry "
                f"{attempt}/{max_retries} in {wait}s, resuming at "
                f"{have / 1e9:.2f} GB",
                flush=True,
            )
            time.sleep(wait)
        finally:
            session.close()
    part_path.rename(output_path)
    return output_path


def _odirect_write(fd: int, buf: mmap.mmap, length: int, offset: int) -> None:
    """Write ``length`` block-aligned bytes of ``buf`` at ``offset`` via O_DIRECT.

    ``length`` and ``offset`` must be multiples of ``_BLOCK``; O_DIRECT short
    writes are themselves block-aligned, so ``written`` stays aligned and every
    ``memoryview`` slice remains page-aligned for the next pwrite.
    """
    view = memoryview(buf)
    written = 0
    while written < length:
        written += os.pwrite(fd, view[written:length], offset + written)


def assemble_parts(
    part_paths: list[Path],
    output_path: Path,
    *,
    expected_total: int | None = None,
    buffer_size: int = 8 * 1024 * 1024,
) -> Path:
    """Concatenate ordered segment files into ``output_path`` using O_DIRECT.

    WSL2's cached-write path reliably hangs the whole VM once a single file
    grows past ~10 GB (microsoft/WSL#5410) — an I/O-handler/page-cache bug,
    independent of write speed (throttling does not help; ``/mnt/c`` and direct
    I/O do). The assembled PBFs (~12 GB, ~23 GB) cross that threshold, so the
    output is written with ``O_DIRECT``, bypassing the page cache entirely. The
    input parts are each < 10 GB, so they are read with ordinary buffered I/O
    and their cache is dropped as they are consumed.

    O_DIRECT requires block-aligned buffers, offsets, and lengths: bytes flow
    through a page-aligned ``mmap`` buffer in ``_BLOCK`` multiples, and the final
    sub-block tail (< ``_BLOCK``) is written through a normal descriptor.
    """
    output_path = Path(output_path)
    sizes = [p.stat().st_size for p in part_paths]
    total = sum(sizes)
    if expected_total is not None and total != expected_total:
        raise OSError(
            f"segment bytes {total} != expected {expected_total}; "
            "download is incomplete, refusing to assemble"
        )
    tmp_path = output_path.with_name(output_path.name + ".part")
    # O_DIRECT needs the buffer length block-aligned; round down to _BLOCK.
    buffer_size = max(_BLOCK, (buffer_size // _BLOCK) * _BLOCK)
    buf = mmap.mmap(-1, buffer_size)  # anonymous mmap is page-aligned for O_DIRECT
    o_direct = getattr(os, "O_DIRECT", 0)
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | o_direct, 0o644)
    offset = 0  # block-aligned bytes committed via O_DIRECT
    fill = 0    # bytes currently buffered
    tail = b""
    try:
        for part_path, size in zip(part_paths, sizes):
            with open(part_path, "rb", buffering=0) as src:
                while True:
                    got = src.readinto(memoryview(buf)[fill:])
                    if got == 0:
                        break  # part exhausted; keep filling from the next part
                    fill += got
                    if fill == buffer_size:
                        _odirect_write(fd, buf, buffer_size, offset)
                        offset += buffer_size
                        fill = 0
                _drop_cache(src.fileno(), 0, size)
        # Trailing partial buffer: block-aligned head via O_DIRECT, tiny tail
        # (< _BLOCK) deferred to a normal descriptor below.
        if fill:
            aligned = (fill // _BLOCK) * _BLOCK
            if aligned:
                _odirect_write(fd, buf, aligned, offset)
                offset += aligned
            if fill > aligned:
                tail = bytes(buf[aligned:fill])
    finally:
        os.close(fd)
        buf.close()
    if tail:
        fd2 = os.open(tmp_path, os.O_WRONLY)  # buffered; tail is < _BLOCK bytes
        try:
            written = 0
            while written < len(tail):
                written += os.pwrite(fd2, tail[written:], offset + written)
            os.fsync(fd2)
        finally:
            os.close(fd2)
    actual = tmp_path.stat().st_size
    if actual != total:
        tmp_path.unlink(missing_ok=True)
        raise OSError(f"assembled size {actual} != expected {total}")
    tmp_path.rename(output_path)
    for part_path in part_paths:
        part_path.unlink(missing_ok=True)
    return output_path


def download_resilient(
    url: str,
    output_path: Path,
    *,
    cookie_file: Path | str | None = None,
    overwrite: bool = False,
    label: str = "file",
    n_segments: int = 4,
    chunk_size: int = 8 * 1024 * 1024,
    connect_timeout: int = 30,
    read_timeout: int = 180,
    max_retries: int = 20,
) -> Path:
    """Download ``url`` to ``output_path`` with parallel ranges + resume.

    Args:
        url: source URL (redirects are followed).
        output_path: final destination path.
        cookie_file: Netscape cookie jar for authenticated endpoints, else None.
        overwrite: if False and the final file already exists, skip.
        label: noun used in log lines (e.g. "PBF", "history PBF").
        n_segments: parallel byte-range connections (1 disables parallelism).
        chunk_size / connect_timeout / read_timeout / max_retries: tunables.

    Returns:
        ``output_path``.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        print(f"{label} already exists at {output_path}; skipping download.")
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total, ranges_ok = 0, False
    if n_segments > 1:
        try:
            total, ranges_ok = _probe(
                url, cookie_file, connect_timeout, read_timeout
            )
        except _RETRYABLE:
            ranges_ok = False  # let the single-stream path handle the retries

    if n_segments > 1 and ranges_ok and total > 0:
        seg_paths = [
            output_path.with_name(output_path.name + f".part{i}")
            for i in range(n_segments)
        ]
        try:
            initial = sum(p.stat().st_size for p in seg_paths if p.exists())
            progress = _Progress(total, label, initial=initial)
            print(
                f"Downloading {label} from {url} with {n_segments} parallel "
                f"connections ({total / 1e9:.2f} GB)..."
            )
            base = total // n_segments
            with ThreadPoolExecutor(max_workers=n_segments) as pool:
                futures = []
                for i, seg_path in enumerate(seg_paths):
                    start = i * base
                    end = total - 1 if i == n_segments - 1 else (i + 1) * base - 1
                    futures.append(
                        pool.submit(
                            _download_segment,
                            url,
                            cookie_file,
                            seg_path,
                            start,
                            end,
                            progress,
                            max_retries,
                            chunk_size,
                            connect_timeout,
                            read_timeout,
                        )
                    )
                for future in futures:
                    future.result()
            # Offset-write assembly with per-part cache eviction; safe because
            # the run is serialised (no second large download in flight).
            print(f"  assembling {n_segments} segments -> {output_path.name} ...")
            assemble_parts(seg_paths, output_path, expected_total=total)
            print(f"\nDownload complete: {output_path}")
            return output_path
        except _RangeUnsupported:
            print(
                "  server did not honour range requests; "
                "falling back to a single stream"
            )
            for seg_path in seg_paths:
                seg_path.unlink(missing_ok=True)

    print(f"Downloading {label} from {url} to {output_path} (single stream)...")
    return _single_stream_resumable(
        url,
        output_path,
        cookie_file,
        label,
        max_retries,
        chunk_size,
        connect_timeout,
        read_timeout,
    )
