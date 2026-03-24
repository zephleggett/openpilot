"""Tests for speed-dependent torque mechanism (vehicle-agnostic).

Uses get_speed_dependent_torque_params() to discover configured cars from
speed_dependent.toml, then tests the CarInterfaceBaseSP callbacks.
"""
import numpy as np
import pytest

from unittest.mock import MagicMock  # noqa: TID251
from opendbc.sunnypilot.car.interfaces import _get_speed_dep_config
from opendbc.sunnypilot.car.interfaces import CarInterfaceBaseSP, LatControlInputs

# Discover all cars with speed-dependent torque config
SPEED_DEP_CARS = _get_speed_dep_config()


def make_ci(fingerprint):
  """Create a CarInterfaceBaseSP instance with the given fingerprint."""
  ci = CarInterfaceBaseSP.__new__(CarInterfaceBaseSP)
  ci.CP = MagicMock()
  ci.CP.carFingerprint = fingerprint
  ci._ensure_speed_dep_init()
  return ci


def make_inputs(lat_accel, vego):
  return LatControlInputs(lateral_acceleration=lat_accel, roll_compensation=0.0, vego=vego, aego=0.0)


@pytest.mark.skipif(len(SPEED_DEP_CARS) == 0, reason="No cars configured in speed_dependent.toml")
class TestSpeedDepTorqueCallbacks:
  """Test speed-dependent torque callbacks for every configured car."""

  def test_config_has_required_keys(self):
    for fingerprint, cfg in SPEED_DEP_CARS.items():
      assert 'speed_bp' in cfg, f"{fingerprint} missing speed_bp"
      assert 'laf_bp' in cfg, f"{fingerprint} missing laf_bp"
      assert len(cfg['speed_bp']) == len(cfg['laf_bp']), f"{fingerprint} speed_bp/laf_bp length mismatch"
      assert len(cfg['speed_bp']) > 1, f"{fingerprint} needs at least 2 breakpoints"
      for i in range(len(cfg['speed_bp']) - 1):
        assert cfg['speed_bp'][i] < cfg['speed_bp'][i + 1], f"{fingerprint} speed_bp not sorted at index {i}"
      if 'friction_bp' in cfg:
        assert len(cfg['friction_bp']) == len(cfg['speed_bp']), f"{fingerprint} friction_bp/speed_bp length mismatch"

  def test_laf_values_positive(self):
    for fingerprint, cfg in SPEED_DEP_CARS.items():
      for laf in cfg['laf_bp']:
        assert laf > 0, f"{fingerprint}: LAF must be positive"

  def test_torque_space_callback_uses_table_values(self):
    for fingerprint, cfg in SPEED_DEP_CARS.items():
      ci = make_ci(fingerprint)
      tp = MagicMock()
      inputs = make_inputs(lat_accel=1.0, vego=15.0)
      laf = float(np.interp(15.0, cfg['speed_bp'], cfg['laf_bp']))
      torque = ci._torque_from_lateral_accel_speed_dep_torque_space(inputs, tp, False)
      assert torque == pytest.approx(1.0 / laf, abs=1e-6), f"{fingerprint}: callback should use table LAF"

  def test_torque_space_callback_tracks_updated_table(self):
    for fingerprint, cfg in SPEED_DEP_CARS.items():
      ci = make_ci(fingerprint)
      tp = MagicMock()
      inputs = make_inputs(lat_accel=1.0, vego=cfg['speed_bp'][0])
      torque_before = ci._torque_from_lateral_accel_speed_dep_torque_space(inputs, tp, False)
      nudged = [v * 1.1 for v in ci._speed_dep_laf_v]
      friction = cfg.get('friction_bp', [0.1] * len(cfg['speed_bp']))
      ci.update_speed_dep_laf(cfg['speed_bp'], nudged, friction, [True] * len(cfg['speed_bp']))
      torque_after = ci._torque_from_lateral_accel_speed_dep_torque_space(inputs, tp, False)
      assert torque_before != pytest.approx(torque_after, abs=1e-3), \
        f"{fingerprint}: callback should reflect updated LAF"

  def test_configured_car_returns_speed_dep_callback(self):
    for fingerprint in SPEED_DEP_CARS:
      ci = make_ci(fingerprint)
      cb = ci.torque_from_lateral_accel_in_torque_space()
      assert cb.__name__ != 'torque_from_lateral_accel_linear_in_torque_space', \
        f"{fingerprint} should not use linear callback"

  def test_update_speed_dep_laf_within_bounds(self):
    for fingerprint, cfg in SPEED_DEP_CARS.items():
      ci = make_ci(fingerprint)
      original_laf = list(ci._speed_dep_laf_v)
      original_fric = list(ci._speed_dep_friction_v)
      nudged_laf = [v * 1.05 for v in original_laf]
      nudged_fric = [v * 1.05 for v in original_fric]
      ci.update_speed_dep_laf(cfg['speed_bp'], nudged_laf, nudged_fric, [True] * len(cfg['speed_bp']))
      for i in range(len(original_laf)):
        assert ci._speed_dep_laf_v[i] == pytest.approx(nudged_laf[i], abs=1e-4), \
          f"{fingerprint} bin {i}: LAF should accept +5% nudge"
        assert ci._speed_dep_friction_v[i] == pytest.approx(nudged_fric[i], abs=1e-4), \
          f"{fingerprint} bin {i}: friction should accept +5% nudge"

  def test_update_speed_dep_laf_out_of_bounds_rejected(self):
    for fingerprint, cfg in SPEED_DEP_CARS.items():
      ci = make_ci(fingerprint)
      original = list(ci._speed_dep_laf_v)
      bad_laf = [v * 5.0 for v in original]
      friction = cfg.get('friction_bp', [0.1] * len(cfg['speed_bp']))
      ci.update_speed_dep_laf(cfg['speed_bp'], bad_laf, friction, [True] * len(cfg['speed_bp']))
      assert ci._speed_dep_laf_v == original, f"{fingerprint}: out-of-bounds LAF should be rejected"

  def test_update_speed_dep_laf_invalid_bins_skipped(self):
    for fingerprint, cfg in SPEED_DEP_CARS.items():
      ci = make_ci(fingerprint)
      n = len(cfg['speed_bp'])
      original = list(ci._speed_dep_laf_v)
      nudged_laf = [v * 1.05 for v in original]
      friction = cfg.get('friction_bp', [0.1] * n)
      valid = [(i % 2 == 0) for i in range(n)]
      ci.update_speed_dep_laf(cfg['speed_bp'], nudged_laf, friction, valid)
      for i in range(n):
        if valid[i]:
          assert ci._speed_dep_laf_v[i] == pytest.approx(nudged_laf[i], abs=1e-4)
        else:
          assert ci._speed_dep_laf_v[i] == pytest.approx(original[i], abs=1e-4)

  def test_update_speed_dep_laf_mismatched_length_rejected(self):
    for fingerprint, cfg in SPEED_DEP_CARS.items():
      ci = make_ci(fingerprint)
      original = list(ci._speed_dep_laf_v)
      short_laf = [v * 1.05 for v in original[:-1]]
      short_valid = [True] * (len(original) - 1)
      friction = cfg.get('friction_bp', [0.1] * len(cfg['speed_bp']))
      ci.update_speed_dep_laf(cfg['speed_bp'], short_laf, friction, short_valid)
      assert ci._speed_dep_laf_v == original, f"{fingerprint}: mismatched-length update should be rejected"


@pytest.mark.skipif(len(SPEED_DEP_CARS) == 0, reason="No cars configured in speed_dependent.toml")
class TestNonConfiguredCarsUnaffected:
  """Cars NOT in speed_dependent.toml should use linear model."""

  def test_unconfigured_car_uses_linear(self):
    ci = make_ci('DEFINITELY_NOT_A_REAL_CAR')
    cb = ci.torque_from_lateral_accel_in_torque_space()
    assert cb.__name__ == 'torque_from_lateral_accel_linear_in_torque_space'

  def test_unconfigured_car_update_is_noop(self):
    ci = make_ci('DEFINITELY_NOT_A_REAL_CAR')
    ci.update_speed_dep_laf([5.0, 15.0], [2.0, 2.0], [0.1, 0.1], [True, True])
