import pytest
from smartsight.voice_commands import Command as C, match_command


@pytest.mark.parametrize("phrase,cmd", [
    ("What is this?", C.IDENTIFY), ("what's this", C.IDENTIFY), ("Read the price.", C.PRICE), ("how much is it", C.PRICE),
    ("Read the ingredients", C.INGREDIENTS), ("read the full ingredients", C.FULL_INGREDIENTS),
    ("Read the expiry date.", C.EXPIRY), ("best before", C.EXPIRY), ("Describe this product", C.DESCRIBE),
    ("Read the label", C.LABEL), ("Repeat that", C.REPEAT), ("Stop speaking", C.STOP), ("Scan again", C.SCAN_AGAIN),
])
def test_phrases(phrase, cmd):
    assert match_command(phrase) == cmd


def test_unrecognised_and_empty():
    assert match_command("banana") is None and match_command("") is None and match_command(None) is None


class FakeRec:
    def __init__(self, texts): self.texts = list(texts)
    def AcceptWaveform(self, data): return bool(self.texts)
    def Result(self):
        import json
        return json.dumps({"text": self.texts.pop(0)})


def test_listener_dispatches_commands_and_ignores_unknown():
    from smartsight.voice_commands import VoiceListener
    got = []
    v = VoiceListener(got.append, recognizer=FakeRec(["read the price", "[unk]", "", "what is this"]))
    assert [v.feed(b"x") for _ in range(4)] == [C.PRICE, None, None, C.IDENTIFY] and got == [C.PRICE, C.IDENTIFY]


def test_listener_only_accepts_stop_while_speaking():
    from smartsight.voice_commands import VoiceListener
    got = []
    v = VoiceListener(got.append, is_speaking=lambda: True, recognizer=FakeRec(["read the price", "stop speaking"]))
    assert v.feed(b"x") is None and v.feed(b"x") == C.STOP and got == [C.STOP]


def test_grammar_phrases_all_map_to_commands():
    from smartsight.voice_commands import GRAMMAR
    assert all(match_command(p) is not None for p in GRAMMAR)


def test_parse_command_with_arguments_and_new_commands():
    from smartsight.voice_commands import parse_command
    assert parse_command("Find my cup") == (C.FIND, "cup")
    assert parse_command("find the red water bottle") == (C.FIND, "red water bottle")
    assert parse_command("find this online") == (C.ASK_AI, None)
    for phrase, cmd in [("more", C.MORE), ("tell me more", C.MORE), ("read text", C.READ_TEXT), ("scan", C.SCENE),
                        ("what's around me", C.SCENE), ("category", C.CATEGORY), ("faster", C.FASTER),
                        ("slower", C.SLOWER), ("verbosity", C.VERBOSITY), ("help", C.HELP), ("quit", C.QUIT),
                        ("scan again", C.SCAN_AGAIN), ("what is this", C.IDENTIFY)]:
        assert parse_command(phrase) == (cmd, None), phrase
