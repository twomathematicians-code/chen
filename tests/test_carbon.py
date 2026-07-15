"""Tests for carbon-aware scheduling."""

from __future__ import annotations

import pytest

from chen.core.carbon import _ZONE_DEFAULTS, CarbonAwareScheduler, CarbonIntensity


class TestCarbonIntensity:
    def test_construction(self):
        ci = CarbonIntensity(
            zone="US-NW",
            intensity_gco2_kwh=50.0,
            renewable_ratio=0.7,
            fossil_ratio=0.3,
        )
        assert ci.zone == "US-NW"
        assert ci.intensity_gco2_kwh == 50.0
        assert ci.source == "unknown"


class TestCarbonAwareScheduler:
    def test_disabled_by_default(self):
        s = CarbonAwareScheduler()
        assert s.enabled is False

    def test_from_env_disabled(self, monkeypatch):
        monkeypatch.delenv("CHEN_CARBON_AWARE", raising=False)
        s = CarbonAwareScheduler.from_env()
        assert s.enabled is False

    def test_from_env_enabled(self, monkeypatch):
        monkeypatch.setenv("CHEN_CARBON_AWARE", "1")
        monkeypatch.setenv("CHEN_DEFAULT_ZONE", "FR")
        s = CarbonAwareScheduler.from_env()
        assert s.enabled is True
        assert s.default_zone == "FR"

    def test_get_intensity_uses_zone_defaults(self):
        s = CarbonAwareScheduler(enabled=True)
        ci = s.get_intensity("FR")
        assert ci.zone == "FR"
        assert ci.intensity_gco2_kwh == _ZONE_DEFAULTS["FR"]
        assert ci.source == "zone_default"

    def test_get_intensity_unknown_zone(self):
        s = CarbonAwareScheduler(enabled=True)
        ci = s.get_intensity("UNKNOWN-ZONE")
        assert ci.intensity_gco2_kwh == _ZONE_DEFAULTS["DEFAULT"]

    def test_get_intensity_uses_default_zone(self):
        s = CarbonAwareScheduler(enabled=True, default_zone="NO")
        ci = s.get_intensity()
        assert ci.zone == "NO"
        assert ci.intensity_gco2_kwh == _ZONE_DEFAULTS["NO"]

    def test_intensity_caches(self):
        s = CarbonAwareScheduler(enabled=True)
        ci1 = s.get_intensity("DE")
        ci2 = s.get_intensity("DE")
        assert ci1 is ci2  # same cached object

    def test_score_expert_disabled_returns_base(self):
        s = CarbonAwareScheduler(enabled=False)
        assert s.score_expert("FR", 0.8) == 0.8

    def test_score_expert_penalizes_high_carbon(self):
        s = CarbonAwareScheduler(enabled=True, carbon_weight=0.5)
        # Low-carbon zone (NO = 20 gCO2)
        low_carbon_score = s.score_expert("NO", 0.5)
        # High-carbon zone (PL = 600 gCO2)
        high_carbon_score = s.score_expert("PL", 0.5)
        assert low_carbon_score > high_carbon_score

    def test_score_expert_respects_weight(self):
        s = CarbonAwareScheduler(enabled=True, carbon_weight=0.0)
        # Weight = 0 means carbon is ignored
        score = s.score_expert("PL", 0.8)
        assert score == pytest.approx(0.8, abs=0.01)

    def test_estimate_co2(self):
        s = CarbonAwareScheduler(enabled=True)
        # FR = 50 gCO2/kWh, 0.001 kWh = 0.05 g CO2
        co2 = s.estimate_co2(0.001, "FR")
        assert co2 == pytest.approx(0.05, abs=0.01)

    def test_estimate_co2_high_carbon_zone(self):
        s = CarbonAwareScheduler(enabled=True)
        # PL = 600 gCO2/kWh
        co2 = s.estimate_co2(1.0, "PL")
        assert co2 == pytest.approx(600.0, abs=1.0)
