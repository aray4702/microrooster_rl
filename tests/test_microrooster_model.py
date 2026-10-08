import importlib.util

import mujoco

from mjlab_microduck.robot.microrooster_constants import (
    MICROROOSTER_XML,
    ROOSTER_STAND_Z,
    get_rooster_spec,
)

GENERATOR = MICROROOSTER_XML.parent / "make_rooster.py"
SCENE_XML = MICROROOSTER_XML.parent / "scene_rooster.xml"


def test_model_compiles_with_the_duck_meshes():
    # meshdir points back at ../microduck/assets/: compiling fails if a mesh
    # path no longer resolves after a file move.
    model = get_rooster_spec().compile()
    assert model.nmesh > 0
    assert model.nu == 12
    actuated = {model.actuator(i).name for i in range(model.nu)}
    assert "head_yaw" not in actuated and "head_roll" not in actuated


def test_scene_stand_keyframe_matches_stand_z():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    key = model.key("STAND")
    assert abs(key.qpos[2] - ROOSTER_STAND_Z) < 1e-4


def test_committed_xml_matches_generator(tmp_path, capsys):
    # The XML is generated: an edit made to it by hand (or a generator change
    # without a re-run) shows up here.
    spec = importlib.util.spec_from_file_location("make_rooster", GENERATOR)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    gen.OUT_ROBOT = tmp_path / MICROROOSTER_XML.name
    gen.OUT_SCENE = tmp_path / SCENE_XML.name
    gen.main()
    assert gen.OUT_ROBOT.read_text() == MICROROOSTER_XML.read_text()
    assert gen.OUT_SCENE.read_text() == SCENE_XML.read_text()
