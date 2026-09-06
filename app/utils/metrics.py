from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import psutil

from app.models import PipelineMetrics, StageMetric
from app.services.ffmpeg_service import find_executable

LOGGER = logging.getLogger(__name__)
MetricDetails = dict[str, str | int | float | bool]


def _process_tree() -> list[psutil.Process]:
    try:
        process = psutil.Process(os.getpid())
        return [process, *process.children(recursive=True)]
    except (psutil.Error, OSError):
        return []


def _memory_mb(processes: list[psutil.Process]) -> float:
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except (psutil.Error, OSError):
            continue
    return total / (1024 * 1024)


def _gpu_memory_mb(process_ids: set[int], nvidia_smi: Path | None) -> float | None:
    if not nvidia_smi:
        return None
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            [
                str(nvidia_smi),
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            creationflags=flags,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None

    total = 0.0
    found = False
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=1)]
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            memory = float(parts[1])
        except ValueError:
            continue
        if pid in process_ids:
            total += memory
            found = True
    return total if found else 0.0


class _ResourceSampler:
    def __init__(self, enabled: bool, interval: float = 0.5) -> None:
        self.enabled = enabled
        self.interval = interval
        self.peak_memory_mb = 0.0
        self.peak_gpu_memory_mb: float | None = None
        self._nvidia_smi = find_executable("nvidia-smi") if enabled else None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._sample()
        if not self.enabled:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="resource-metrics",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(1.0, self.interval * 2))
        self._sample()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self) -> None:
        processes = _process_tree()
        self.peak_memory_mb = max(
            self.peak_memory_mb,
            _memory_mb(processes),
        )
        if not self.enabled:
            return
        gpu_memory = _gpu_memory_mb(
            {process.pid for process in processes},
            self._nvidia_smi,
        )
        if gpu_memory is not None:
            self.peak_gpu_memory_mb = max(
                self.peak_gpu_memory_mb or 0.0,
                gpu_memory,
            )


class MetricsCollector:
    def __init__(self, *, sample_resources: bool = False) -> None:
        self.sample_resources = sample_resources
        self.started_at = datetime.now(timezone.utc)
        self._started_perf = time.perf_counter()
        self.stages: list[StageMetric] = []

    @contextmanager
    def stage(
        self,
        name: str,
        **details: str | int | float | bool,
    ) -> Iterator[MetricDetails]:
        mutable_details: MetricDetails = dict(details)
        sampler = _ResourceSampler(self.sample_resources)
        started = time.perf_counter()
        cpu_started = time.process_time()
        sampler.start()
        try:
            yield mutable_details
        finally:
            sampler.stop()
            metric = StageMetric(
                name=name,
                elapsed_seconds=round(time.perf_counter() - started, 4),
                cpu_seconds=round(time.process_time() - cpu_started, 4),
                peak_memory_mb=round(sampler.peak_memory_mb, 2),
                peak_gpu_memory_mb=(
                    round(sampler.peak_gpu_memory_mb, 2)
                    if sampler.peak_gpu_memory_mb is not None
                    else None
                ),
                details=mutable_details,
            )
            self.stages.append(metric)
            LOGGER.info(
                "Stage %s completed in %.3fs; peak RAM %.1f MB; peak GPU %s MB",
                name,
                metric.elapsed_seconds,
                metric.peak_memory_mb,
                (
                    f"{metric.peak_gpu_memory_mb:.1f}"
                    if metric.peak_gpu_memory_mb is not None
                    else "n/a"
                ),
            )

    def snapshot(
        self,
        *,
        source_duration_seconds: float | None = None,
    ) -> PipelineMetrics:
        elapsed = max(0.0, time.perf_counter() - self._started_perf)
        speed = None
        if source_duration_seconds and elapsed > 0:
            speed = source_duration_seconds / elapsed
        return PipelineMetrics(
            started_at=self.started_at,
            completed_at=datetime.now(timezone.utc),
            total_elapsed_seconds=round(elapsed, 4),
            source_duration_seconds=source_duration_seconds,
            speed_vs_realtime=round(speed, 4) if speed is not None else None,
            resource_sampling_enabled=self.sample_resources,
            stages=list(self.stages),
        )
