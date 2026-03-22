"""Energy measurement via zeus-ml's Apple Silicon IOKit interface."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PowerReading:
    avg_watts: float  # average combined power draw
    total_joules: float  # total energy consumed
    duration_s: float
    components: dict[str, float]  # component -> watts (cpu, gpu, dram, ane)


class PowerMonitor:
    """Power measurement using zeus-ml's Apple Silicon support.

    Uses the AppleSilicon SoC interface directly (bypassing ZeusMonitor)
    because ZeusMonitor.end_window() doesn't aggregate SoC energy fields
    into total_energy in zeus 0.15.0.

    IOKit-based — no sudo required on macOS.
    """

    def __init__(self) -> None:
        self._soc = None
        self._available = False
        self._start_times: dict[str, float] = {}
        try:
            # Patch zeus 0.15.0 bug: DeprecatedAliasABCMeta registers
            # "zeroAllFields" as abstract but the concrete implementation
            # only defines "zero_all_fields", making the class uninstantiable.
            from zeus.device.soc.apple import AppleSiliconMeasurement

            if "zeroAllFields" in getattr(
                AppleSiliconMeasurement, "__abstractmethods__", frozenset()
            ):
                AppleSiliconMeasurement.__abstractmethods__ = (
                    AppleSiliconMeasurement.__abstractmethods__ - {"zeroAllFields"}
                )

            from zeus.device.soc.apple import AppleSilicon

            self._soc = AppleSilicon()
            self._available = True
            metrics = self._soc.get_available_metrics()
            logger.info("Power monitoring initialized, metrics: %s", metrics)
        except Exception as e:
            logger.info(
                "Power monitoring not available (%s). "
                "To enable tokens/watt metrics, run: uv sync --extra power",
                e,
            )

    @property
    def available(self) -> bool:
        return self._available

    def begin_window(self, name: str) -> None:
        """Start a named measurement window."""
        if not self._available:
            return
        try:
            self._soc.begin_window(name)
            self._start_times[name] = time.perf_counter()
        except Exception as e:
            logger.warning("Failed to begin power window '%s': %s", name, e)

    def end_window(self, name: str) -> PowerReading | None:
        """End a named measurement window and return the reading."""
        if not self._available:
            return None
        try:
            duration = time.perf_counter() - self._start_times.pop(name, 0)
            measurement = self._soc.end_window(name)

            # Extract millijoule fields and convert to joules
            cpu_mj = getattr(measurement, "cpu_total_mj", None) or 0
            gpu_mj = getattr(measurement, "gpu_mj", None) or 0
            dram_mj = getattr(measurement, "dram_mj", None) or 0
            ane_mj = getattr(measurement, "ane_mj", None) or 0
            total_mj = cpu_mj + gpu_mj + dram_mj + ane_mj
            total_joules = total_mj / 1000.0

            avg_watts = total_joules / duration if duration > 0 else 0.0

            components: dict[str, float] = {}
            if duration > 0:
                components["cpu"] = (cpu_mj / 1000.0) / duration
                components["gpu"] = (gpu_mj / 1000.0) / duration
                components["dram"] = (dram_mj / 1000.0) / duration
                components["ane"] = (ane_mj / 1000.0) / duration

            return PowerReading(
                avg_watts=avg_watts,
                total_joules=total_joules,
                duration_s=duration,
                components=components,
            )
        except Exception as e:
            logger.warning("Failed to end power window '%s': %s", name, e)
            return None
