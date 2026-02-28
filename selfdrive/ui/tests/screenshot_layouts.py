#!/usr/bin/env python3
"""Render every mici settings screen to PNG for PR documentation.

Requires a display context (run from a regular terminal, not SSH).

Usage:
  cd /path/to/sunnypilot
  PYTHONPATH=. SCALE=1 .venv/bin/python selfdrive/ui/tests/screenshot_layouts.py
"""
import os

os.environ["BIG"] = "0"
os.environ.setdefault("SCALE", "1")

import pyray as rl

from pathlib import Path
from cereal import car, custom
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix


OUTPUT_DIR = Path(__file__).parent / "screenshots"
SETTLE_FRAMES = 30


def setup_params():
  params = Params()

  cp = car.CarParams.new_message(
    enableBsm=True,
    brand="debug",
    openpilotLongitudinalControl=True,
    steerControlType="torque",
  )
  params.put("CarParamsPersistent", cp.to_bytes())

  cp_sp = custom.CarParamsSP.new_message(
    intelligentCruiseButtonManagementAvailable=True,
  )
  params.put("CarParamsSPPersistent", cp_sp.to_bytes())
  params.put_bool("IntelligentCruiseButtonManagement", True)

  # Visuals
  params.put_bool("BlindSpot", True)
  params.put_bool("TorqueBar", True)
  params.put_bool("RoadNameToggle", True)
  params.put_bool("TrueVEgoUI", True)
  params.put_bool("ShowTurnSignals", True)
  params.put("ChevronInfo", 3)
  params.put("DevUIInfo", 1)

  # Display
  params.put("OnroadScreenOffBrightness", 10)
  params.put("OnroadScreenOffTimer", 2)
  params.put("InteractivityTimeout", 30)

  # Steering
  params.put_bool("Mads", True)
  params.put_bool("MadsMainCruiseAllowed", True)
  params.put_bool("MadsUnifiedEngagementMode", False)
  params.put("MadsSteeringMode", 1)
  params.put_bool("AutoLaneChangeTimer", True)
  params.put_bool("AutoLaneChangeBsmDelay", True)
  params.put_bool("BlinkerPauseLateralControl", True)
  params.put("BlinkerMinLateralControlSpeed", 25)
  params.put("BlinkerLateralReengageDelay", 3)
  params.put_bool("EnforceTorqueControl", True)
  params.put_bool("LiveTorqueParamsToggle", True)
  params.put_bool("LiveTorqueParamsRelaxedToggle", True)
  params.put_bool("CustomTorqueParams", True)
  params.put_bool("TorqueParamsOverrideEnabled", True)
  params.put("TorqueParamsOverrideLatAccelFactor", 250.0)
  params.put("TorqueParamsOverrideFriction", 50.0)

  # Cruise
  params.put_bool("CustomAccIncrementsEnabled", True)
  params.put("CustomAccShortPressIncrement", 3)
  params.put("CustomAccLongPressIncrement", 2)
  params.put("SpeedLimitMode", 2)
  params.put("SpeedLimitPolicy", 1)
  params.put("SpeedLimitOffsetType", 1)
  params.put("SpeedLimitValueOffset", 5)

  # Trips
  params.put("ApiCache_DriveStats", {
    "all": {"routes": 142, "distance": 3456, "minutes": 8760},
    "week": {"routes": 7, "distance": 89, "minutes": 420},
  })


def setup_ui_state():
  from openpilot.selfdrive.ui.ui_state import ui_state

  ui_state.params = Params()
  ui_state.CP = car.CarParams.new_message(
    enableBsm=True,
    brand="debug",
    openpilotLongitudinalControl=True,
    steerControlType="torque",
  )
  ui_state.CP_SP = custom.CarParamsSP.new_message(
    intelligentCruiseButtonManagementAvailable=True,
  )
  ui_state.started = False
  ui_state.has_longitudinal_control = True
  ui_state.has_icbm = True
  ui_state.is_metric = False
  ui_state.is_sp_release = False


def capture(widget, filename, frames=SETTLE_FRAMES):
  from openpilot.system.ui.lib.application import gui_app

  rt = rl.load_render_texture(gui_app.width, gui_app.height)

  if hasattr(widget, '_trigger_animate_in'):
    widget._trigger_animate_in = False
  if hasattr(widget, '_pos_filter'):
    widget._pos_filter.x = 0.0

  # Keep alpha opaque: blend RGB normally, but force alpha to stay at dst (1.0 from clear)
  GL_FUNC_ADD = 0x8006
  rl.rl_set_blend_factors_separate(
    0x0302, 0x0303,  # RGB: GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
    0x0000, 0x0001,  # Alpha: GL_ZERO, GL_ONE (preserves dst alpha = 1.0)
    GL_FUNC_ADD, GL_FUNC_ADD,
  )

  rect = rl.Rectangle(0, 0, gui_app.width, gui_app.height)
  for _ in range(frames):
    rl.begin_texture_mode(rt)
    rl.clear_background(rl.BLACK)
    rl.begin_blend_mode(rl.BLEND_CUSTOM_SEPARATE)
    widget.render(rect)
    rl.end_blend_mode()
    rl.end_texture_mode()
    rl.begin_drawing()
    rl.end_drawing()

  image = rl.load_image_from_texture(rt.texture)
  rl.image_flip_vertical(image)

  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  path = str(OUTPUT_DIR / filename)
  rl.export_image(image, path)

  rl.unload_image(image)
  rl.unload_render_texture(rt)
  print(f"  {filename}")


def capture_display():
  from openpilot.selfdrive.ui.sunnypilot.mici.layouts.display import DisplayLayoutMici

  d = DisplayLayoutMici(back_callback=lambda: None)
  d.show_event()
  capture(d, "display.png")

  d._show_picker_for(d._brightness)
  capture(d, "display_brightness_picker.png")
  d._back_callback()

  d._show_picker_for(d._brightness_timer)
  capture(d, "display_timer_picker.png")
  d._back_callback()

  d._show_picker_for(d._ui_timeout)
  capture(d, "display_timeout_picker.png")


def capture_visuals():
  from openpilot.selfdrive.ui.sunnypilot.mici.layouts.visuals import VisualsLayoutMici

  v = VisualsLayoutMici(back_callback=lambda: None)
  v.show_event()
  capture(v, "visuals.png")


def capture_cruise():
  from openpilot.selfdrive.ui.sunnypilot.mici.layouts.cruise import CruiseLayoutMici

  c = CruiseLayoutMici(back_callback=lambda: None)
  c.show_event()
  capture(c, "cruise.png")

  # Custom ACC sub-panel
  c._show_custom_acc()
  capture(c, "cruise_custom_acc.png")

  c._show_picker_for(c._acc_short)
  capture(c, "cruise_acc_short_picker.png")
  c._back_callback()

  c._show_picker_for(c._acc_long)
  capture(c, "cruise_acc_long_picker.png")
  c._reset_main_view()

  # Speed limit sub-panel
  c._show_speed_limit()
  capture(c, "cruise_speed_limit.png")

  c._show_picker_for(c._sl_offset_value)
  capture(c, "cruise_sl_offset_picker.png")
  c._reset_main_view()


def capture_steering():
  from openpilot.selfdrive.ui.sunnypilot.mici.layouts.steering import SteeringLayoutMici

  s = SteeringLayoutMici(back_callback=lambda: None)
  s.show_event()
  capture(s, "steering.png")

  # MADS sub-panel
  s._show_mads()
  capture(s, "steering_mads.png")
  s._reset_main_view()

  # Lane change sub-panel
  s._show_lane_change()
  capture(s, "steering_lane_change.png")
  s._reset_main_view()

  # Blinker sub-panel
  s._show_blinker()
  capture(s, "steering_blinker.png")

  s._show_picker_for(s._blinker_speed)
  capture(s, "steering_blinker_speed_picker.png")
  s._back_callback()

  s._show_picker_for(s._blinker_delay)
  capture(s, "steering_blinker_delay_picker.png")
  s._reset_main_view()

  # Torque sub-panel
  s._show_torque()
  capture(s, "steering_torque.png")

  # Self-tune sub-panel (nested under torque)
  s._show_self_tune()
  capture(s, "steering_self_tune.png")
  s._back_callback()  # back to torque

  # Custom tune sub-panel (nested under torque)
  s._show_custom_tune()
  capture(s, "steering_custom_tune.png")

  s._show_picker_for(s._tq_lat_accel)
  capture(s, "steering_lat_accel_picker.png")
  s._back_callback()  # back to custom tune

  s._show_picker_for(s._tq_friction)
  capture(s, "steering_friction_picker.png")
  s._reset_main_view()


def capture_trips():
  from openpilot.selfdrive.ui.sunnypilot.mici.layouts.trips import TripsLayoutMici

  t = TripsLayoutMici(back_callback=lambda: None)
  t.show_event()
  capture(t, "trips.png")


def main():
  with OpenpilotPrefix():
    setup_params()

    rl.set_config_flags(rl.FLAG_WINDOW_HIDDEN)

    from openpilot.system.ui.lib.application import gui_app
    gui_app.init_window("screenshot_layouts", fps=30)

    setup_ui_state()

    print("Capturing screenshots...")
    capture_display()
    capture_visuals()
    capture_cruise()
    capture_steering()
    capture_trips()

    gui_app.close()
    print(f"\nDone — {OUTPUT_DIR}/")


if __name__ == "__main__":
  main()
