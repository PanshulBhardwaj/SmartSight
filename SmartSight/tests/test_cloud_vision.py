import json
import numpy as np
from config.settings import Settings
from smartsight.cloud_vision import CloudVision, encode_frame, normalize
from smartsight.narration import NarrationEngine
from smartsight.utils import InfoState
from smartsight.voice_commands import Command as C, match_command
from tests.test_pipeline import FakeCamera, FakeDetector, FakeOcr, NoBarcode, GOOD
from smartsight.pipeline import Pipeline
from smartsight.product_lookup import LocalProductStore
from smartsight.speech_engine import SpeechController
from tests.fakes import FakeBackend

GOOD_JSON = {"can_identify": True, "item": "steel water bottle", "brand": None, "category": "drinkware",
             "description": "A reusable insulated bottle.", "typical_uses": "It keeps drinks hot or cold.",
             "approx_price_inr": {"low": 400, "high": 900, "basis": "typical insulated bottles"}}


def reply(obj_text, status=200):
    body = json.dumps({"content": [{"type": "text", "text": obj_text}]}).encode()
    return lambda url, h, b, t: (status, body)


def test_ok_and_fenced_json_and_request_shape():
    seen = {}
    def tr(url, headers, body, timeout):
        seen.update(headers=headers, body=json.loads(body))
        return 200, json.dumps({"content": [{"type": "text", "text": "```json\n" + json.dumps(GOOD_JSON) + "\n```"}]}).encode()
    r = CloudVision("k", transport=tr).describe(b"jpeg")
    assert r.status == "ok" and r.data["item"] == "steel water bottle"
    assert seen["headers"]["x-api-key"] == "k" and "tools" not in seen["body"]
    assert seen["body"]["messages"][0]["content"][0]["source"]["media_type"] == "image/jpeg"


def test_web_search_tool_added_only_when_enabled():
    seen = {}
    def tr(u, h, b, t): seen["b"] = json.loads(b); return 200, json.dumps({"content": [{"type": "text", "text": json.dumps(GOOD_JSON)}]}).encode()
    CloudVision("k", web_search=True, transport=tr).describe(b"x")
    assert seen["b"]["tools"][0]["name"] == "web_search"


def test_failures_map_to_statuses():
    assert CloudVision(None).describe(b"x").status == "no_key"
    assert CloudVision("k", transport=reply("", 401)).describe(b"x").status == "unauthorized"
    assert CloudVision("k", transport=reply("", 429)).describe(b"x").status == "rate_limited"
    assert CloudVision("k", transport=reply("", 500)).describe(b"x").status == "error"
    assert CloudVision("k", transport=reply("not json at all")).describe(b"x").status == "invalid_response"
    def boom(*a): raise OSError("offline")
    assert CloudVision("k", transport=boom).describe(b"x").status == "network_error"


def test_normalize_rejects_silly_prices_and_empty_item():
    d = dict(GOOD_JSON, approx_price_inr={"low": 10, "high": 5000, "basis": "?"})
    assert normalize(d)["price"] is None                      # 500x range is not an estimate
    assert normalize(dict(GOOD_JSON, item=""))["can_identify"] is False
    assert normalize({"item": "x"}) is None


def test_narration_labels_price_as_approximate_not_printed():
    from smartsight.cloud_vision import CloudResult
    n = NarrationEngine().cloud(CloudResult("ok", normalize(GOOD_JSON)))
    assert n.state == InfoState.TENTATIVE
    assert "steel water bottle" in n.text and "not a printed price" in n.text and "four hundred" in n.text
    unk = NarrationEngine().cloud(CloudResult("ok", normalize(dict(GOOD_JSON, can_identify=False))))
    assert unk.state == InfoState.UNKNOWN


def test_encode_frame_downscales():
    jpg = encode_frame(np.zeros((2000, 3000, 3), np.uint8), max_side=1024)
    import cv2
    assert max(cv2.imdecode(np.frombuffer(jpg, np.uint8), 1).shape[:2]) == 1024


def test_voice_phrases():
    for p in ("ask AI", "search online", "find this online", "look it up", "online price"):
        assert match_command(p) == C.ASK_AI, p
    assert match_command("how much is it") == C.PRICE          # printed-price command unaffected


def make(cloud, **flags):
    s = Settings.load(path="/none", env={}); s.capture_samples = 1
    for k, v in flags.items(): setattr(s, k, v)
    b = FakeBackend(); cam = FakeCamera(); cam.started = True
    p = Pipeline(s, cam, FakeDetector(), FakeOcr([]), NoBarcode(), LocalProductStore(),
                 SpeechController(b), sleep=lambda t: None, cloud=cloud)
    return p, b


class Spy(CloudVision):
    calls = 0
    def describe(self, jpeg, hints=None):
        Spy.calls += 1
        Spy.hints = hints
        return super().describe(jpeg, hints)


def test_nothing_is_sent_without_consent_or_when_paused():
    Spy.calls = 0
    spy = Spy("k", transport=reply(json.dumps(GOOD_JSON)))
    for flags, phrase in (({}, "permission"), ({"cloud_ai": False, "cloud_consent": True}, "turned off")):
        p, b = make(spy, **flags)
        p.process_command(C.ASK_AI); p.speech.wait_idle(2)
        assert phrase in b.spoken[-1]
    assert Spy.calls == 0


def test_no_key_sends_nothing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p, b = make(None, cloud_ai=True, cloud_consent=True)
    p.process_command(C.ASK_AI); p.speech.wait_idle(2)
    assert "no API key" in b.spoken[-1]


def test_dark_frame_not_uploaded():
    Spy.calls = 0
    p, b = make(Spy("k", transport=reply(json.dumps(GOOD_JSON))), cloud_ai=True, cloud_consent=True)
    p.camera.frame = np.zeros((240, 320, 3), np.uint8)
    p.process_command(C.ASK_AI); p.speech.wait_idle(2)
    assert Spy.calls == 0 and "too dark" in b.spoken[-1]


def test_full_flow_announces_notice_then_result():
    p, b = make(CloudVision("k", transport=reply(json.dumps(GOOD_JSON))), cloud_ai=True, cloud_consent=True)
    r = p.process_command(C.ASK_AI); p.speech.wait_idle(2)
    assert "Looking that up online" in b.spoken[0] and "steel water bottle" in b.spoken[1]
    assert r.state == InfoState.TENTATIVE


def test_cloud_is_core_by_default_but_consent_gated():
    s = Settings()
    assert s.cloud_ai is True and s.cloud_auto is True and s.cloud_web_search is True
    assert s.cloud_consent is False          # nothing is sent until the user agrees once


# ---------------- integrated (automatic) behaviour ----------------
WITH_SOURCES = json.dumps({"content": [
    {"type": "web_search_tool_result", "content": [{"url": "https://www.amazon.in/x", "title": "t"},
                                                    {"url": "https://flipkart.com/y"}]},
    {"type": "text", "text": json.dumps(dict(GOOD_JSON, similar_items=["glass bottle", "thermos"]))}]}).encode()


def test_sources_and_similar_items_parsed_and_spoken():
    r = CloudVision("k", web_search=True, transport=lambda *a: (200, WITH_SOURCES)).describe(b"x")
    assert r.data["sources"] == ["amazon.in", "flipkart.com"] and r.data["similar_items"] == ["glass bottle", "thermos"]
    text = NarrationEngine("detailed").cloud(r).text
    assert "Similar items include glass bottle, thermos" in text and "came from amazon.in, flipkart.com" in text


def test_hints_sent_to_ai_and_search_prompt_added():
    seen = {}
    def tr(u, h, b, t): seen["b"] = json.loads(b); return 200, WITH_SOURCES
    CloudVision("k", web_search=True, transport=tr).describe(b"x", ["bottle"])
    assert "bottle" in seen["b"]["messages"][0]["content"][1]["text"] and "web search tool" in seen["b"]["system"]


def auto(transport=None, **flags):
    flags = dict(cloud_ai=True, cloud_consent=True, stable_frames=2, cloud_min_interval=10.0, **flags)
    spy = Spy("k", transport=transport or reply(json.dumps(GOOD_JSON)))
    p, b = make(spy, **flags)
    return p, b, spy


def drain(p, cycles=6):
    for _ in range(cycles):
        p.step()
        try:
            while True: p.process_command(p._cmds.get_nowait())
        except Exception: pass
    p.speech.wait_idle(2)


def test_camera_sees_object_then_ai_identifies_automatically():
    Spy.calls = 0
    p, b, spy = auto()
    drain(p)
    assert b.spoken[0].startswith("Household item, bottle")       # instant local announcement (with category)
    assert any("steel water bottle" in t for t in b.spoken[1:])    # AI follow-up, no button pressed
    assert Spy.calls == 1 and Spy.hints == ["bottle"]


def test_stationary_object_is_looked_up_once_and_rate_limited():
    Spy.calls = 0
    p, b, spy = auto()
    drain(p, 60)
    assert Spy.calls == 1


def test_auto_lookup_never_runs_without_consent():
    Spy.calls = 0
    spy = Spy("k", transport=reply(json.dumps(GOOD_JSON)))
    p, b = make(spy, cloud_ai=True, cloud_consent=False, stable_frames=2)
    drain(p)
    assert Spy.calls == 0 and len(b.spoken) == 1 and b.spoken[0].startswith("Household item, bottle")


def test_auto_lookup_can_be_switched_off():
    Spy.calls = 0
    p, b, spy = auto(cloud_auto=False)
    drain(p)
    assert Spy.calls == 0


def test_auto_lookup_silent_on_failure():
    p, b, spy = auto(transport=reply("", 500))
    drain(p)
    assert len(b.spoken) == 1 and b.spoken[0].startswith("Household item, bottle")      # failure stays silent


def test_identify_uses_ai_when_local_cannot_confirm():
    Spy.calls = 0
    p, b, spy = auto()
    p.process_command(C.IDENTIFY); p.speech.wait_idle(2)
    assert Spy.calls == 1 and "steel water bottle" in b.spoken[-1]


def test_describe_always_uses_ai_and_offline_falls_back_to_local():
    def boom(*a): raise OSError("offline")
    p, b, spy = auto(transport=boom)
    p.ocr.lines = []
    p.process_command(C.DESCRIBE); p.speech.wait_idle(2)
    assert "could not reach the online AI" in b.spoken[-1] and "bottle" in b.spoken[-1]


def test_same_scene_answer_is_cached_for_user_requests():
    Spy.calls = 0
    p, b, spy = auto()
    p.process_command(C.IDENTIFY); p.process_command(C.IDENTIFY)
    assert Spy.calls == 1


def test_confirmed_local_product_skips_ai_on_identify():
    from smartsight.product_lookup import ProductRecord
    Spy.calls = 0
    p, b, spy = auto()
    p.store.add(ProductRecord("1", "Peppy Crisps", "Peppy"))
    p.barcode = type("B", (), {"read": lambda self, f: ["1"], "reset": lambda self: None})()
    p.ocr.lines = [__import__("smartsight.utils", fromlist=["OcrLine"]).OcrLine(t, 0.9, (0, 0, 9, 20)) for t in ("PEPPY", "CRISPS")]
    p.process_command(C.IDENTIFY); p.speech.wait_idle(2)
    assert Spy.calls == 0 and "Peppy" in b.spoken[-1]


def test_status_text():
    p, b, spy = auto()
    assert p.cloud_status().startswith("ON")
    p2, _ = make(Spy("k"), cloud_ai=True, cloud_consent=False)
    assert "permission" in p2.cloud_status()


# ---------------- free providers: Gemini (free tier) and Ollama (local) ----------------
def gem_reply(text, chunks=None, status=200):
    cand = {"content": {"parts": [{"text": text}]}}
    if chunks:
        cand["groundingMetadata"] = {"groundingChunks": [{"web": {"uri": "https://vertexaisearch.cloud.google.com/x", "title": t}} for t in chunks]}
    return lambda u, h, b, t: (status, json.dumps({"candidates": [cand]}).encode())


def test_provider_is_chosen_by_model_name():
    assert CloudVision("k", "claude-sonnet-5-5").provider == "anthropic"
    assert CloudVision("k", "gemini-3.1-flash-lite").provider == "gemini"
    assert CloudVision(None, "ollama:qwen2.5vl:3b").provider == "ollama"


def test_gemini_request_shape_and_parse_with_search_sources():
    seen = {}
    def tr(url, headers, body, timeout):
        seen.update(url=url, headers=headers, body=json.loads(body))
        return gem_reply(json.dumps(dict(GOOD_JSON, similar_items=["thermos"])), ["amazon.in", "Some Page Title", "www.flipkart.com"])(url, headers, body, timeout)
    r = CloudVision("gk", "gemini-3.1-flash-lite", web_search=True, transport=tr).describe(b"jpeg", ["bottle"])
    assert r.status == "ok" and r.data["item"] == "steel water bottle" and r.data["similar_items"] == ["thermos"]
    assert r.data["sources"] == ["amazon.in", "flipkart.com"]                      # real site titles kept, free-text titles dropped
    assert seen["url"].endswith("/models/gemini-3.1-flash-lite:generateContent") and seen["headers"]["x-goog-api-key"] == "gk"
    assert "x-api-key" not in seen["headers"] and seen["body"]["tools"] == [{"google_search": {}}]
    parts = seen["body"]["contents"][0]["parts"]
    assert parts[0]["inline_data"]["mime_type"] == "image/jpeg" and "bottle" in parts[1]["text"]
    assert "web search tool" in seen["body"]["system_instruction"]["parts"][0]["text"]


def test_gemini_without_search_has_no_tools_and_reads_key_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    seen = {}
    def tr(u, h, b, t): seen.update(h=h, b=json.loads(b)); return gem_reply(json.dumps(GOOD_JSON))(u, h, b, t)
    c = CloudVision(None, "gemini-3.1-flash-lite", web_search=False, transport=tr)
    assert c.configured and c.describe(b"x").status == "ok"
    assert seen["h"]["x-goog-api-key"] == "from-env" and "tools" not in seen["b"]


def test_gemini_errors_map_to_statuses(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False); monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert CloudVision(None, "gemini-3.1-flash-lite").describe(b"x").status == "no_key"
    for code, want in ((403, "unauthorized"), (429, "rate_limited"), (500, "error")):
        assert CloudVision("k", "gemini-3.1-flash-lite", transport=lambda *a, c=code: (c, b"{}")).describe(b"x").status == want
    bad_key = lambda *a: (400, b'{"error":{"message":"API key not valid. Please pass a valid API key."}}')
    assert CloudVision("k", "gemini-3.1-flash-lite", transport=bad_key).describe(b"x").status == "unauthorized"
    assert CloudVision("k", "gemini-3.1-flash-lite", transport=lambda *a: (200, b'{"candidates": []}')).describe(b"x").status == "invalid_response"
    blocked = lambda *a: (200, json.dumps({"promptFeedback": {"blockReason": "SAFETY"}}).encode())
    assert CloudVision("k", "gemini-3.1-flash-lite", transport=blocked).describe(b"x").status == "invalid_response"


def test_ollama_needs_no_key_never_uses_web_search_and_stays_local():
    seen = {}
    def tr(url, headers, body, timeout):
        seen.update(url=url, headers=headers, body=json.loads(body))
        return 200, json.dumps({"message": {"content": "Sure! " + json.dumps(GOOD_JSON)}}).encode()
    c = CloudVision(None, "ollama:qwen2.5vl:3b", web_search=True, transport=tr)
    assert c.configured
    r = c.describe(b"jpeg", ["cup"])
    assert r.status == "ok" and r.data["item"] == "steel water bottle" and r.data["sources"] == []
    assert seen["url"] == "http://localhost:11434/api/chat" and seen["body"]["model"] == "qwen2.5vl:3b"
    assert "tools" not in seen["body"] and "web search" not in seen["body"]["messages"][0]["content"].lower()
    assert seen["body"]["messages"][1]["images"] and "x-api-key" not in seen["headers"] and seen["body"]["stream"] is False


def test_ollama_not_running_gives_a_helpful_message(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
    def refused(url, h, b, t):
        assert url.startswith("http://127.0.0.1:11434"); raise ConnectionRefusedError("refused")
    r = CloudVision(None, "ollama:llava", transport=refused).describe(b"x")
    assert r.status == "network_error" and "Is Ollama running" in r.message


def test_ai_text_from_any_provider_is_still_sanitised_in_the_card():
    from smartsight.card import CardBuilder
    from smartsight.knowledge import KnowledgeBase
    from smartsight.taxonomy import Vocabulary
    from smartsight.fusion import Recognition
    r = CloudVision("k", "gemini-3.1-flash-lite",
                    transport=gem_reply(json.dumps(dict(GOOD_JSON, description="A bottle. It is safe to drink from.")))).describe(b"x")
    card = CardBuilder(KnowledgeBase(), Vocabulary.empty()).build(Recognition("cup", "Household Item", "Kitchenware", 0.9, "detector"), ai=r.data)
    assert card.ai["description"] == "A bottle."
