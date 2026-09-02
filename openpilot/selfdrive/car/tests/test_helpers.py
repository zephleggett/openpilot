from types import SimpleNamespace

from openpilot.cereal import custom
from openpilot.selfdrive.car.helpers import convert_carControlSP
from opendbc.car import structs


class TestConvertCarControlSP:
  def test_substructs_convert_to_dataclasses(self):
    msg = custom.CarControlSP.new_message()
    msg.mads.enabled = True
    msg.leadOne.dRel = 12.5

    out = convert_carControlSP(msg.as_reader())
    assert isinstance(out.mads, structs.ModularAssistiveDrivingSystem)
    assert out.mads.enabled is True
    assert isinstance(out.leadOne, structs.LeadData)
    assert abs(out.leadOne.dRel - 12.5) < 1e-6

  def test_capnp_only_substructs_are_dropped(self):
    # a substruct added to the capnp schema before the opendbc dataclass has a field for it
    # must be dropped, not passed to the dataclass constructor as a kwarg
    struct_dict = custom.CarControlSP.new_message().to_dict()
    struct_dict['newSubStruct'] = {'enabled': True}

    out = convert_carControlSP(SimpleNamespace(to_dict=lambda: struct_dict))
    assert not hasattr(out, 'newSubStruct')
    assert isinstance(out.mads, structs.ModularAssistiveDrivingSystem)
