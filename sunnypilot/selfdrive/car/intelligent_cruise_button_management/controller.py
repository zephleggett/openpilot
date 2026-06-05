"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

from cereal import car, custom
from opendbc.car import structs, apply_hysteresis
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_CTRL
from openpilot.sunnypilot.selfdrive.car.intelligent_cruise_button_management.helpers import get_minimum_set_speed
from openpilot.sunnypilot.selfdrive.car.cruise_ext import CRUISE_BUTTON_TIMER, update_manual_button_timers

# Mazda-specific MRCC inverse solver (Mazda CX-5 2022 only). Module is gated
# inside _maybe_apply_mazda_inverse_solver so non-Mazda cars never call it.
from opendbc.sunnypilot.car.mazda.mrcc_inverse import inverse_solve as mazda_mrcc_inverse_solve

LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource
State = custom.IntelligentCruiseButtonManagement.IntelligentCruiseButtonManagementState
SendButtonState = custom.IntelligentCruiseButtonManagement.SendButtonState

ALLOWED_SPEED_THRESHOLD = 1.8  # m/s, ~4 MPH
HYST_GAP = 0.0  # currently disabled; TODO-SP: might need to be brand-specific
INACTIVE_TIMER = 0.4

# Sentinel-rejection ceiling for solver inputs. SCC Vision publishes
# V_CRUISE_UNSET = 255 kph (~70 m/s) when its source is not active; the
# paired dTarget check should already gate us, but accept m/s values up
# to a generous-but-finite max so an out-of-band sentinel cannot leak in
# and command an unintended large set speed.
MAZDA_SOLVER_MAX_V_MS = 60.0  # ~134 mph, well above any real curve target

# Carbon-copy of the fingerprint for which the empirical MRCC model is
# calibrated. Other Mazda models would need their own characterization
# before being routed through the solver.
MAZDA_SMARTCRUISE_FINGERPRINTS = {'MAZDA_CX5_2022'}


SEND_BUTTONS = {
  State.increasing: SendButtonState.increase,
  State.decreasing: SendButtonState.decrease,
}


class IntelligentCruiseButtonManagement:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP):
    self.CP = CP
    self.CP_SP = CP_SP

    self.v_target = 0
    self.v_cruise_cluster = 0
    self.v_cruise_min = 0
    self.cruise_button = SendButtonState.none
    self.state = State.inactive
    self.pre_active_timer = 0

    self.is_ready = False
    self.is_ready_prev = False
    self.v_target_ms_last = 0.0
    self.is_metric = False

    self.cruise_button_timers = CRUISE_BUTTON_TIMER

  @property
  def v_cruise_equal(self) -> bool:
    return self.v_target == self.v_cruise_cluster

  def _maybe_apply_mazda_inverse_solver(self, CS: car.CarState, LP_SP: custom.LongitudinalPlanSP, v_target_ms: float) -> float:
    """Mazda CX-5 2022 only: when the long plan source is Vision/Map and a
    distance is published, replace v_target with the MRCC inverse solver's
    sp_command so ICBM commands a distance-aware overshoot instead of chasing
    the raw vTarget.

    Falls back to the original v_target_ms in all other cases (other brands,
    other Mazda models, other sources, no distance, sentinel values).
    """
    # Strict fingerprint gate: the MRCC characterization in mrcc_inverse.py is
    # derived from CX-5 2022 logs only. Don't run it for other Mazdas (CX-9,
    # MX-30, Mazda3, etc.) — their MRCC tuning differs.
    if self.CP.brand != 'mazda' or self.CP.carFingerprint not in MAZDA_SMARTCRUISE_FINGERPRINTS:
      return v_target_ms

    # Reject negative/zero/NaN/sentinel v_ego (uninitialized carState).
    v_ego = CS.vEgo
    if not (0.0 < v_ego < MAZDA_SOLVER_MAX_V_MS):
      return v_target_ms

    source = LP_SP.longitudinalPlanSource
    v_curve_ms = 0.0
    d_m = 0.0
    if source == LongitudinalPlanSource.sccVision:
      d_m = LP_SP.smartCruiseControl.vision.dTarget
      v_curve_ms = LP_SP.smartCruiseControl.vision.vTargetCurve
    elif source == LongitudinalPlanSource.sccMap:
      d_m = LP_SP.smartCruiseControl.map.dTarget
      v_curve_ms = LP_SP.smartCruiseControl.map.vTarget
    else:
      return v_target_ms

    # Bound both inputs explicitly. v_curve must be positive AND below the
    # sentinel ceiling (V_CRUISE_UNSET in m/s is ~70). d must be positive and
    # finite. NaN comparisons evaluate False so non-finite values are also
    # rejected here. Critical: v_curve must be below v_ego or the solver would
    # be asked to "decelerate" to a higher target -- skip and let raw vTarget
    # path handle that (e.g. cruise speed is below the curve speed -> no-op).
    if not (0.0 < v_curve_ms < v_ego):
      return v_target_ms
    if not (0.0 < d_m < 1000.0):
      return v_target_ms

    # Pass current a_ego so forward_simulate uses the actual decel/accel state
    # rather than assuming the vehicle is at steady cruise.
    a_ego_initial = CS.aEgo if math.isfinite(CS.aEgo) else 0.0

    result = mazda_mrcc_inverse_solve(v_ego * CV.MS_TO_MPH,
                                       v_curve_ms * CV.MS_TO_MPH,
                                       d_m,
                                       a_ego_initial=a_ego_initial)
    sp_mph = result.get('sp_command_mph')
    # If the solver returned a malformed dict or NaN, refuse to override
    # vTarget rather than ship a garbage command.
    if sp_mph is None or not math.isfinite(sp_mph):
      return v_target_ms
    # Clamp to [v_curve, v_ego] in mph. Guarded by the v_curve < v_ego check
    # above so sp_mph_min <= sp_mph_max always holds.
    sp_mph_max = v_ego * CV.MS_TO_MPH
    sp_mph_min = v_curve_ms * CV.MS_TO_MPH
    sp_mph = max(min(sp_mph, sp_mph_max), sp_mph_min)
    return sp_mph * CV.MPH_TO_MS

  def update_calculations(self, CS: car.CarState, LP_SP: custom.LongitudinalPlanSP) -> None:
    speed_conv = CV.MS_TO_KPH if self.is_metric else CV.MS_TO_MPH
    ms_conv = CV.KPH_TO_MS if self.is_metric else CV.MPH_TO_MS

    v_target_ms = self._maybe_apply_mazda_inverse_solver(CS, LP_SP, LP_SP.vTarget)
    self.v_target_ms_last = apply_hysteresis(v_target_ms, self.v_target_ms_last, HYST_GAP * ms_conv)

    self.v_target = round(self.v_target_ms_last * speed_conv)
    self.v_cruise_min = get_minimum_set_speed(self.is_metric)
    self.v_cruise_cluster = round(CS.cruiseState.speedCluster * speed_conv)

  def update_state_machine(self) -> custom.IntelligentCruiseButtonManagement.SendButtonState:
    self.pre_active_timer = max(0, self.pre_active_timer - 1)

    # HOLDING, ACCELERATING, DECELERATING, PRE_ACTIVE
    if self.state != State.inactive:
      if not self.is_ready:
        self.state = State.inactive

      else:
        # PRE_ACTIVE
        if self.state == State.preActive:
          if self.pre_active_timer <= 0:
            if self.v_cruise_equal:
              self.state = State.holding

            elif self.v_target > self.v_cruise_cluster:
              self.state = State.increasing

            elif self.v_target < self.v_cruise_cluster and self.v_cruise_cluster > self.v_cruise_min:
              self.state = State.decreasing

        # HOLDING
        elif self.state == State.holding:
          if not self.v_cruise_equal:
            self.state = State.preActive

        # ACCELERATING
        elif self.state == State.increasing:
          if self.v_target <= self.v_cruise_cluster:
            self.state = State.holding

        # DECELERATING
        elif self.state == State.decreasing:
          if self.v_target >= self.v_cruise_cluster or self.v_cruise_cluster <= self.v_cruise_min:
            self.state = State.holding

    # INACTIVE
    elif self.state == State.inactive:
      if self.is_ready and not self.is_ready_prev:
        self.pre_active_timer = int(INACTIVE_TIMER / DT_CTRL)
        self.state = State.preActive

    send_button = SEND_BUTTONS.get(self.state, SendButtonState.none)

    return send_button

  def update_readiness(self, CS: car.CarState, CC: car.CarControl) -> None:
    update_manual_button_timers(CS, self.cruise_button_timers)

    ready = CC.enabled and not CC.cruiseControl.override and not CC.cruiseControl.cancel and not CC.cruiseControl.resume
    button_pressed = any(self.cruise_button_timers[k] > 0 for k in self.cruise_button_timers)

    self.is_ready = ready and not button_pressed

  def run(self, CS: car.CarState, CC: car.CarControl, LP_SP: custom.LongitudinalPlanSP, is_metric: bool) -> None:
    if self.CP_SP.pcmCruiseSpeed:
      return

    self.is_metric = is_metric

    self.update_calculations(CS, LP_SP)
    self.update_readiness(CS, CC)

    self.cruise_button = self.update_state_machine()

    self.is_ready_prev = self.is_ready
