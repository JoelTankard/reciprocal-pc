from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass

import torch


class ResourceLimitExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class ResourceSnapshot:
    system_memory_percent: float | None
    system_load_percent: float | None
    gpu_memory_percent: float | None

    def format(self) -> str:
        fields = []
        if self.system_memory_percent is not None:
            fields.append(f"system_memory={self.system_memory_percent:.1f}%")
        if self.system_load_percent is not None:
            fields.append(f"system_load={self.system_load_percent:.1f}%")
        if self.gpu_memory_percent is not None:
            fields.append(f"gpu_memory={self.gpu_memory_percent:.1f}%")
        return " ".join(fields) if fields else "resource_usage=unknown"


@dataclass(frozen=True)
class ResourceGuard:
    max_system_memory_percent: float = 90.0
    max_system_load_percent: float = 90.0
    max_gpu_memory_percent: float = 90.0

    def snapshot(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            system_memory_percent=system_memory_percent(),
            system_load_percent=system_load_percent(),
            gpu_memory_percent=gpu_memory_percent(),
        )

    def format_caps(self) -> str:
        return (
            f"max_system_memory={self.max_system_memory_percent:.1f}% "
            f"max_system_load={self.max_system_load_percent:.1f}% "
            f"max_gpu_memory={self.max_gpu_memory_percent:.1f}%"
        )

    def check(self, *, label: str) -> ResourceSnapshot:
        snapshot = self.snapshot()
        self.check_snapshot(snapshot, label=label)
        return snapshot

    def check_snapshot(self, snapshot: ResourceSnapshot, *, label: str) -> None:
        breaches = []
        if (
            self.max_system_memory_percent > 0
            and snapshot.system_memory_percent is not None
            and snapshot.system_memory_percent >= self.max_system_memory_percent
        ):
            breaches.append(f"system memory {snapshot.system_memory_percent:.1f}%")
        if (
            self.max_system_load_percent > 0
            and snapshot.system_load_percent is not None
            and snapshot.system_load_percent >= self.max_system_load_percent
        ):
            breaches.append(f"system load {snapshot.system_load_percent:.1f}%")
        if (
            self.max_gpu_memory_percent > 0
            and snapshot.gpu_memory_percent is not None
            and snapshot.gpu_memory_percent >= self.max_gpu_memory_percent
        ):
            breaches.append(f"GPU memory {snapshot.gpu_memory_percent:.1f}%")
        if breaches:
            raise ResourceLimitExceeded(f"{label}: resource guard stopped run: {', '.join(breaches)}")

    def wait_until_safe(self, *, label: str, poll_seconds: float, timeout_seconds: float) -> ResourceSnapshot:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be nonnegative")

        deadline = time.monotonic() + timeout_seconds
        last_error: ResourceLimitExceeded | None = None
        while True:
            try:
                return self.check(label=label)
            except ResourceLimitExceeded as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    raise ResourceLimitExceeded(f"{label}: timed out waiting for safe resources: {exc}") from exc
                time.sleep(poll_seconds)

        if last_error is not None:
            raise last_error
        raise ResourceLimitExceeded(f"{label}: timed out waiting for safe resources")


BYTES_PER_GB = 1024**3


def platform_name() -> str:
    if hasattr(os, "uname"):
        return os.uname().sysname
    return os.name


class PeakMemoryTracker:
    def __init__(self, *, poll_seconds: float = 0.05) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.poll_seconds = poll_seconds
        self.peak_cpu_memory_gb = 0.0
        self.peak_gpu_memory_gb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakMemoryTracker":
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_seconds * 4)
        self._sample()

    def _run(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            self._sample()

    def _sample(self) -> None:
        cpu_gb = process_memory_gb()
        gpu_gb = gpu_memory_gb()
        if cpu_gb is not None:
            self.peak_cpu_memory_gb = max(self.peak_cpu_memory_gb, cpu_gb)
        if gpu_gb is not None:
            self.peak_gpu_memory_gb = max(self.peak_gpu_memory_gb, gpu_gb)


def process_memory_gb() -> float | None:
    platform = platform_name()
    if platform == "Darwin":
        return _darwin_process_memory_gb()
    if os.name != "posix":
        return None
    return _linux_process_memory_gb()


def gpu_memory_gb() -> float | None:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / BYTES_PER_GB

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / BYTES_PER_GB

    return 0.0


def system_load_percent() -> float | None:
    if not hasattr(os, "getloadavg"):
        return None
    try:
        load1, _, _ = os.getloadavg()
    except OSError:
        return None
    cpu_count = os.cpu_count()
    if not cpu_count:
        return None
    return 100.0 * load1 / cpu_count


def system_memory_percent() -> float | None:
    platform = platform_name()
    if platform == "Darwin":
        return _darwin_memory_percent()
    if os.name != "posix":
        return None
    return _linux_memory_percent()


def gpu_memory_percent() -> float | None:
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if total_bytes <= 0:
            return None
        return 100.0 * (total_bytes - free_bytes) / total_bytes

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        recommended = torch.mps.recommended_max_memory()
        if recommended <= 0:
            return None
        return 100.0 * torch.mps.driver_allocated_memory() / recommended

    return None


def _darwin_memory_percent() -> float | None:
    try:
        mem_size_text = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        vm_text = subprocess.check_output(["vm_stat"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None

    total_bytes = int(mem_size_text)
    page_size = _parse_vm_stat_page_size(vm_text)
    if page_size is None:
        return None
    counts = _parse_vm_stat(vm_text)
    free_pages = counts.get("Pages free", 0) + counts.get("Pages speculative", 0)
    used_bytes = max(total_bytes - free_pages * page_size, 0)
    return 100.0 * used_bytes / total_bytes


def _darwin_process_memory_gb() -> float | None:
    try:
        rss_kb = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if not rss_kb:
        return None
    return int(rss_kb) * 1024 / BYTES_PER_GB


def _parse_vm_stat_page_size(text: str) -> int | None:
    first_line = text.splitlines()[0] if text else ""
    match = re.search(r"page size of ([0-9]+) bytes", first_line)
    if not match:
        return None
    return int(match.group(1))


def _parse_vm_stat(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"([^:]+):\s+([0-9]+)\.", line.strip())
        if match:
            counts[match.group(1)] = int(match.group(2))
    return counts


def _linux_memory_percent() -> float | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return None

    values: dict[str, int] = {}
    for line in lines:
        name, _, rest = line.partition(":")
        raw_value = rest.strip().split()[0]
        values[name] = int(raw_value)
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if not total or available is None:
        return None
    return 100.0 * (total - available) / total


def _linux_process_memory_gb() -> float | None:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            parts = handle.read().split()
    except OSError:
        return None
    if len(parts) < 2:
        return None
    return int(parts[1]) * os.sysconf("SC_PAGE_SIZE") / BYTES_PER_GB
