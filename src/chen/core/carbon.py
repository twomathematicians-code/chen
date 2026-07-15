"""Carbon-aware scheduling — route to lower-carbon experts based on real-time grid intensity.

Integrates with the Electricity Maps API (https://api.electricitymap.org)
to fetch real-time carbon intensity for a given zone. When multiple
experts can handle a prompt, the router prefers the one running in a
lower-carbon region.

Usage::

    from chen.core.carbon import CarbonAwareScheduler

    scheduler = CarbonAwareScheduler(api_key="...")
    zone = scheduler.get_zone("us-east-1")
    intensity = scheduler.get_intensity(zone)
    # intensity in gCO2eq/kWh

    # Get the best expert based on carbon intensity
    best = scheduler.choose_expert(prompt, experts, router_scores)

Environment variables:
    CHEN_CARBON_AWARE: "1" to enable (default off).
    CHEN_ELECTRICITY_MAPS_API_KEY: API key for Electricity Maps.
    CHEN_DEFAULT_ZONE: Default zone (e.g. "US-NW" for US Northwest).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from chen.observability.logging import get_logger

_log = get_logger("chen.core.carbon")


@dataclass
class CarbonIntensity:
    """Carbon intensity for a grid zone at a point in time.

    Attributes:
        zone: Zone identifier (e.g. "US-NW", "FR", "DE").
        intensity_gco2_kwh: Carbon intensity in gCO2eq/kWh.
        renewable_ratio: Fraction of generation from renewables (0..1).
        fossil_ratio: Fraction from fossil fuels (0..1).
        timestamp: Unix timestamp of the reading.
        source: Data source ("electricity_maps", "mock", "cache").
    """

    zone: str
    intensity_gco2_kwh: float
    renewable_ratio: float = 0.0
    fossil_ratio: float = 0.0
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"


# Approximate carbon intensity for common zones (gCO2eq/kWh).
# Source: Electricity Maps 2024 averages.
_ZONE_DEFAULTS: dict[str, float] = {
    "US-NW": 50.0,  # Pacific Northwest — lots of hydro
    "US-CA": 100.0,  # California — solar + wind
    "US-SE": 400.0,  # Southeast — coal + gas
    "US-TX": 350.0,  # Texas — wind + gas
    "US-NY": 200.0,  # New York — mixed
    "FR": 50.0,  # France — nuclear
    "DE": 350.0,  # Germany — coal + renewables
    "GB": 250.0,  # Great Britain — gas + wind
    "NO": 20.0,  # Norway — hydro
    "PL": 600.0,  # Poland — coal
    "CN": 550.0,  # China — coal
    "IN": 700.0,  # India — coal
    "AU": 500.0,  # Australia — coal
    "DEFAULT": 400.0,
}


@dataclass
class CarbonAwareScheduler:
    """Carbon-aware expert scheduler.

    When enabled, decorates the router's scoring with a carbon penalty:
    experts in higher-carbon zones get a score penalty, making the
    router prefer lower-carbon experts when scores are close.

    Attributes:
        enabled: If False, carbon-aware scheduling is disabled (pass-through).
        api_key: Electricity Maps API key. If None, uses zone defaults.
        default_zone: Fallback zone when expert zone is unknown.
        cache_ttl: Cache duration in seconds for intensity data.
        carbon_weight: How much carbon affects routing (0.0 = ignore, 1.0 = only carbon).
    """

    enabled: bool = False
    api_key: Optional[str] = None  # noqa: UP045
    default_zone: str = "US-NW"
    cache_ttl: float = 300.0  # 5 minutes
    carbon_weight: float = 0.2

    _cache: dict[str, tuple[float, CarbonIntensity]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.enabled:
            self.api_key = self.api_key or os.environ.get("CHEN_ELECTRICITY_MAPS_API_KEY")
            self.default_zone = os.environ.get("CHEN_DEFAULT_ZONE", self.default_zone)
            _log.info(
                "carbon.scheduler_enabled",
                default_zone=self.default_zone,
                weight=self.carbon_weight,
            )

    @classmethod
    def from_env(cls) -> CarbonAwareScheduler:
        """Create scheduler from environment variables."""
        return cls(
            enabled=os.environ.get("CHEN_CARBON_AWARE", "") == "1",
        )

    def get_intensity(self, zone: Optional[str] = None) -> CarbonIntensity:  # noqa: UP045
        """Get carbon intensity for a zone.

        Uses the Electricity Maps API if an API key is configured.
        Falls back to zone defaults if not.

        Args:
            zone: Zone identifier. Defaults to self.default_zone.

        Returns:
            CarbonIntensity for the zone.
        """
        z = zone or self.default_zone

        # Check cache.
        if z in self._cache:
            ts, cached = self._cache[z]
            if time.time() - ts < self.cache_ttl:
                return cached

        # Try the API.
        if self.api_key:
            try:
                intensity = self._fetch_from_api(z)
                self._cache[z] = (time.time(), intensity)
                return intensity
            except Exception as e:
                _log.warning("carbon.api_failed", zone=z, error=str(e))

        # Fall back to defaults.
        gco2 = _ZONE_DEFAULTS.get(z, _ZONE_DEFAULTS["DEFAULT"])
        intensity = CarbonIntensity(
            zone=z,
            intensity_gco2_kwh=gco2,
            source="zone_default",
        )
        self._cache[z] = (time.time(), intensity)
        return intensity

    def _fetch_from_api(self, zone: str) -> CarbonIntensity:
        """Fetch real-time intensity from Electricity Maps API."""
        import urllib.request

        url = f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={zone}"
        req = urllib.request.Request(url, headers={"auth-token": self.api_key or ""})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        return CarbonIntensity(
            zone=zone,
            intensity_gco2_kwh=data.get("carbonIntensity", _ZONE_DEFAULTS["DEFAULT"]),
            renewable_ratio=data.get("renewableRatio", 0.0),
            fossil_ratio=data.get("fossilRatio", 0.0),
            source="electricity_maps",
        )

    def score_expert(
        self,
        expert_zone: Optional[str],  # noqa: UP045
        base_score: float,
    ) -> float:
        """Adjust an expert's routing score based on its zone's carbon intensity.

        Lower carbon = higher adjusted score (more likely to be selected).

        Args:
            expert_zone: Zone where the expert runs. None = default zone.
            base_score: The router's original score (0..1).

        Returns:
            Adjusted score (0..1+). Higher is better.
        """
        if not self.enabled:
            return base_score
        intensity = self.get_intensity(expert_zone)
        # Carbon penalty: normalize to 0..1 where 0 = max carbon (700+), 1 = zero carbon.
        max_intensity = 700.0
        carbon_score = max(0.0, 1.0 - intensity.intensity_gco2_kwh / max_intensity)
        # Weighted combination.
        adjusted = base_score * (1.0 - self.carbon_weight) + carbon_score * self.carbon_weight
        _log.debug(
            "carbon.score",
            zone=intensity.zone,
            gco2=intensity.intensity_gco2_kwh,
            base=base_score,
            adjusted=adjusted,
        )
        return adjusted

    def estimate_co2(
        self,
        energy_kwh: float,
        zone: Optional[str] = None,  # noqa: UP045
    ) -> float:
        """Estimate CO2 emissions for a given energy consumption.

        Args:
            energy_kwh: Energy in kWh.
            zone: Grid zone. Defaults to self.default_zone.

        Returns:
            CO2 in grams.
        """
        intensity = self.get_intensity(zone)
        return energy_kwh * intensity.intensity_gco2_kwh
