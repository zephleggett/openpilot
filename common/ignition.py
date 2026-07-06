from cereal import log


# Process-lifetime latch: once CAN ignition has been seen on any panda, we stop trusting
# ignitionLine. On Mazda, ignitionLine stays high for ~30s after the ignition is
# turned off; ignitionCan goes false promptly, so it is the authoritative signal.
ignition_can_seen = False


def get_ignition_state(panda_states) -> bool:
  global ignition_can_seen

  valid_states = [ps for ps in panda_states if ps.pandaType != log.PandaState.PandaType.unknown]
  if not valid_states:
    return False

  if any(ps.ignitionCan for ps in valid_states):
    ignition_can_seen = True
    return True

  return False if ignition_can_seen else any(ps.ignitionLine for ps in valid_states)
