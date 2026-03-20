"""Energy measurement via zeus-ml."""

from __future__ import annotations

import logging
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

    Uses IOKit APIs internally — no sudo required on macOS.
    """

    def __init__(self) -> None:
        self._monitor = None
        self._available = False
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

            from zeus.monitor.energy import ZeusMonitor

            self._monitor = ZeusMonitor(approx_instant_energy=True)
            self._available = True
            logger.info("zeus-ml power monitoring initialized")
        except Exception as e:
            logger.error(
                "Failed to initialize zeus-ml power monitoring: %s. "
                "Power metrics will be unavailable. "
                "Install with: uv add 'zeus[apple]'",
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
            self._monitor.begin_window(name)
        except Exception as e:
            logger.warning("Failed to begin power window '%s': %s", name, e)

    def end_window(self, name: str) -> PowerReading | None:
        """End a named measurement window and return the reading."""
        if not self._available:
            return None
        try:
            measurement = self._monitor.end_window(name)
            total_joules = measurement.total_energy
            duration = measurement.time
            avg_watts = total_joules / duration if duration > 0 else 0.0

            # Extract per-component breakdown if available
            components: dict[str, float] = {}
            if hasattr(measurement, "cpu_energy"):
                cpu_j = measurement.cpu_energy or 0.0
                components["cpu"] = cpu_j / duration if duration > 0 else 0.0
            if hasattr(measurement, "gpu_energy"):
                gpu_j = measurement.gpu_energy or 0.0
                components["gpu"] = gpu_j / duration if duration > 0 else 0.0

            return PowerReading(
                avg_watts=avg_watts,
                total_joules=total_joules,
                duration_s=duration,
                components=components,
            )
        except Exception as e:
            logger.warning("Failed to end power window '%s': %s", name, e)
            return None
