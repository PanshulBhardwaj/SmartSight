import time
from smartsight.speech_engine import ConsoleBackend, Priority, SpeechController
from tests.fakes import FakeBackend, FakeClock


def test_speaks_in_order_without_overlap():
    b = FakeBackend(); s = SpeechController(b)
    for i in range(3):
        s.say(f"msg {i}", Priority.USER)
    assert s.wait_idle(2) and b.spoken == ["msg 0", "msg 1", "msg 2"]
    s.shutdown()


def test_user_priority_beats_queued_auto():
    b = FakeBackend(delay=0.3); s = SpeechController(b, max_pending_auto=5)
    s.say("first", Priority.AUTO)
    time.sleep(0.05)
    s.say("auto 2", Priority.AUTO); s.say("auto 3", Priority.AUTO)
    s.say("user answer", Priority.USER)
    assert s.wait_idle(3)
    assert b.spoken.index("user answer") < b.spoken.index("auto 2")
    s.shutdown()


def test_user_request_interrupts_current_auto():
    b = FakeBackend(delay=2.0, slow_word="long"); s = SpeechController(b)
    s.say("long auto announcement", Priority.AUTO)
    time.sleep(0.1)
    t0 = time.time(); s.say("answer", Priority.USER)
    assert s.wait_idle(3) and time.time() - t0 < 1.5 and b.stops >= 1
    s.shutdown()


def test_auto_duplicates_dropped():
    clock = FakeClock(); b = FakeBackend(); s = SpeechController(b, dedupe_seconds=5, clock=clock)
    assert s.say("I can see a bottle.") is True; s.wait_idle(1)
    assert s.say("I can see a bottle.") is False
    clock.advance(6)
    assert s.say("I can see a bottle.") is True
    s.shutdown()


def test_user_repeat_not_deduped():
    b = FakeBackend(); s = SpeechController(b)
    s.say("price is ten", Priority.USER); s.wait_idle(1)
    assert s.repeat() and s.wait_idle(1)
    assert b.spoken == ["price is ten", "price is ten"]
    s.shutdown()


def test_auto_backlog_capped():
    b = FakeBackend(delay=0.3); s = SpeechController(b, max_pending_auto=2)
    results = [s.say(f"a{i}") for i in range(6)]
    assert results.count(True) <= 3     # one speaking + at most 2 pending
    s.shutdown()


def test_stop_clears_queue_and_cuts_speech():
    b = FakeBackend(delay=2.0); s = SpeechController(b, max_pending_auto=9)
    for i in range(5):
        s.say(f"m{i}", Priority.USER)
    time.sleep(0.1)
    s.stop()
    assert s.wait_idle(2) and len(b.spoken) == 1 and b.stops >= 1
    s.shutdown()


def test_backend_exception_does_not_kill_worker():
    class Bad(FakeBackend):
        def speak(self, text):
            if text == "boom": raise RuntimeError("tts failed")
            super().speak(text)
    b = Bad(); s = SpeechController(b)
    s.say("boom", Priority.USER); s.say("ok", Priority.USER)
    assert s.wait_idle(2) and b.spoken == ["ok"]
    s.shutdown()


def test_empty_text_ignored_and_closed_controller_rejects():
    s = SpeechController(FakeBackend())
    assert s.say("  ") is False
    s.shutdown()
    assert s.say("late") is False


def test_repeat_with_nothing_spoken():
    assert SpeechController(FakeBackend()).repeat() is False


def test_console_backend_prints(capsys):
    ConsoleBackend().speak("hi")
    assert "hi" in capsys.readouterr().out
