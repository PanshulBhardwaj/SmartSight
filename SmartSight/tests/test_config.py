import json
from config.settings import Settings


def test_defaults():
    s = Settings.load(path="/nonexistent.json", env={})
    assert s.verbosity == "standard" and s.online_lookup is False and s.camera_index == 0


def test_file_and_env_override(tmp_path):
    f = tmp_path / "s.json"; f.write_text(json.dumps({"verbosity": "concise", "speech_rate": 200}))
    s = Settings.load(f, env={"SMARTSIGHT_SPEECH_RATE": "150", "SMARTSIGHT_ONLINE_LOOKUP": "true"})
    assert s.verbosity == "concise" and s.speech_rate == 150 and s.online_lookup is True


def test_invalid_values_fall_back_with_warnings(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"confidence_threshold": 7, "verbosity": "loud", "speech_rate": "fast", "ocr_engine": "x"}))
    s = Settings.load(f, env={})
    assert s.confidence_threshold == 0.95 and s.verbosity == "standard" and s.speech_rate == 170 and s.ocr_engine == "auto"
    assert len(s.load_warnings) == 4


def test_corrupt_json_does_not_crash(tmp_path):
    f = tmp_path / "s.json"; f.write_text("{broken")
    s = Settings.load(f, env={})
    assert s.verbosity == "standard" and s.load_warnings


def test_unknown_keys_ignored_and_list_parsing(tmp_path):
    f = tmp_path / "s.json"; f.write_text(json.dumps({"mystery": 1}))
    s = Settings.load(f, env={"SMARTSIGHT_IGNORE_CLASSES": "person, dog"})
    assert s.ignore_classes == ["person", "dog"]


def test_save_roundtrip(tmp_path):
    s = Settings(); s.verbosity = "detailed"; s.save(tmp_path / "x.json")
    assert Settings.load(tmp_path / "x.json", env={}).verbosity == "detailed"


def test_privacy_defaults_off():
    s = Settings()
    assert s.online_lookup is False and s.voice_input is False


def test_yaml_load_save_roundtrip_and_load_order(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("verbosity: detailed\nstable_frames: 6\nignore_classes: [person, dog]\n")
    s = Settings.load(f, env={})
    assert s.verbosity == "detailed" and s.stable_frames == 6 and s.ignore_classes == ["person", "dog"]
    s.speech_rate = 200; s.save(tmp_path / "u.yaml")
    assert Settings.load(tmp_path / "u.yaml", env={}).speech_rate == 200


def test_shipped_config_yaml_matches_dataclass_defaults():
    import yaml
    from config.settings import CONFIG_PATH
    from dataclasses import fields
    data = yaml.safe_load(CONFIG_PATH.read_text())
    names = {f.name for f in fields(Settings)}
    assert set(data) <= names, set(data) - names          # no typos / dead keys
    s = Settings.load(CONFIG_PATH, env={})
    assert s.load_warnings == []
    assert s.cloud_consent is False                        # nothing is sent until the user agrees


def test_yaml_parse_error_does_not_crash(tmp_path):
    f = tmp_path / "bad.yaml"; f.write_text("a: [unclosed")
    s = Settings.load(f, env={})
    assert s.load_warnings and s.verbosity == "standard"
