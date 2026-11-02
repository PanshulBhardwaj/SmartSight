import numpy as np
import pytest
from config.settings import Settings, ROOT
from smartsight.detector import Detection
from smartsight.fusion import fuse
from smartsight.recognition_manager import RecognitionManager
from smartsight.recognizer import Candidate, ClipRecognizer, NullRecognizer, center_box, crop_box, make_recognizer
from smartsight.taxonomy import Taxonomy, Vocabulary
from smartsight.tracker import IouTracker

VOCAB = Vocabulary.from_file(ROOT / "data" / "vocabulary.json")
TAX = Taxonomy.from_file(ROOT / "data" / "taxonomy.json")
REQUIRED = {"Food and Drink", "Packaged Goods", "Medicine and Health", "Machine and Appliance", "Electronics", "Tool", "Vehicle",
            "Furniture", "Clothing and Accessories", "Plant", "Animal", "Person", "Document and Text", "Currency", "Household Item",
            "Stationery", "Other"}


# ---------- data files ----------
def test_taxonomy_has_all_required_categories_and_vocab_is_valid():
    assert REQUIRED <= set(TAX.categories)
    assert len(VOCAB.entries) >= 300
    for e in VOCAB.entries:
        assert TAX.is_valid(e.category, e.subcategory), e


def test_every_coco_class_is_mapped_to_a_valid_category():
    coco = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign parking_meter bench bird cat dog horse "
            "sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball kite baseball_bat "
            "baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli "
            "carrot hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard cell_phone microwave "
            "oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier toothbrush").split()
    assert len(coco) == 80
    for c in coco:
        m = TAX.map_detector(c.replace("_", " "))
        assert m.category != "Other" or c in ("traffic_light", "fire_hydrant", "stop_sign", "parking_meter"), c
        assert TAX.is_valid(m.category, m.subcategory), (c, m)
    assert TAX.map_detector("flux capacitor").category == "Other"          # unknown labels degrade gracefully


def test_vocabulary_find_handles_free_text():
    assert VOCAB.find("cup").name == "cup"
    assert VOCAB.find("cups").name == "cup"
    assert VOCAB.find("my red water bottle").name in ("reusable water bottle", "plastic water bottle")
    assert VOCAB.find("shampoo").name == "shampoo bottle"
    assert VOCAB.find("TV").name == "television"
    assert VOCAB.find("xyzzy plugh") is None


# ---------- zero-shot recognizer with a fake embedder ----------
class FakeEmbedder:
    """Text vectors are seeded by the prompt's object name; an image encodes its true entry index in pixel (0,0)."""
    key = "fake"
    def __init__(self, vocab): self.vocab, self.text_calls = vocab, 0
    def _vec(self, name):
        r = np.random.default_rng(abs(hash(name)) % (2 ** 32)); v = r.normal(size=32); return v / np.linalg.norm(v)
    def encode_texts(self, texts):
        self.text_calls += 1
        out = []
        for t in texts:
            name = t.split(" a ", 1)[-1].replace(".", "").replace("photo of ", "").replace("close-up photo of ", "")
            name = t
            for e in self.vocab.entries:
                for n in (e.name, *e.aliases):
                    if t.startswith(("a photo of a ", "a close-up photo of a ", "a ")) and n in t:
                        name = n
            out.append(self._vec(name))
        return np.stack(out).astype(np.float32)
    def encode_images(self, imgs):
        idx = int(imgs[0][0, 0, 0]) * 256 + int(imgs[0][0, 0, 1])
        return np.stack([self._vec(self.vocab.entries[idx].name)]).astype(np.float32)


def img_for(vocab, name):
    i = vocab.entries.index(vocab.get(name))
    im = np.zeros((40, 40, 3), np.uint8); im[0, 0, 0], im[0, 0, 1] = divmod(i, 256); return im


def test_clip_recognizer_ranks_correct_label_and_scores_sum_to_one(tmp_path):
    emb = FakeEmbedder(VOCAB)
    rec = ClipRecognizer(emb, VOCAB, tmp_path)
    top = rec.classify(img_for(VOCAB, "stapler"), top_k=3)
    assert top[0].label == "stapler" and top[0].category == "Stationery" and top[0].score > top[1].score
    allp = rec.classify(img_for(VOCAB, "stapler"), top_k=len(VOCAB.entries))
    assert abs(sum(c.score for c in allp) - 1.0) < 1e-4


def test_text_embeddings_cached_on_disk(tmp_path):
    emb = FakeEmbedder(VOCAB)
    ClipRecognizer(emb, VOCAB, tmp_path).classify(img_for(VOCAB, "hammer"))
    assert emb.text_calls == 1 and list(tmp_path.glob("clip_text_*.npy"))
    emb2 = FakeEmbedder(VOCAB)
    ClipRecognizer(emb2, VOCAB, tmp_path).classify(img_for(VOCAB, "hammer"))
    assert emb2.text_calls == 0                                              # reused from disk


def test_null_and_degenerate_inputs():
    assert NullRecognizer().classify(None) == []
    assert ClipRecognizer(FakeEmbedder(VOCAB), VOCAB).classify(np.zeros((0, 0, 3), np.uint8)) == []


def test_make_recognizer_degrades_gracefully_without_open_clip():
    s = Settings.load(path="/none", env={})
    rec, msg = make_recognizer(s, VOCAB)
    assert rec is None and "requirements-recognition.txt" in msg            # open_clip is not installed in the sandbox
    s.zero_shot_enabled = False
    assert make_recognizer(s, VOCAB)[0] is None


def test_crop_helpers():
    f = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
    assert crop_box(f, (50, 20, 100, 60)).shape[:2] == (48, 60)
    assert crop_box(f, (500, 500, 600, 600)) is None
    assert center_box(f) == (40, 20, 160, 80)


# ---------- fusion ----------
M = TAX.map_detector
C = lambda l, s, c, sub: Candidate(l, s, c, sub)


def test_zero_shot_refines_generic_detector_class():
    r = fuse(M("bottle"), 0.9, [C("shampoo bottle", 0.7, "Packaged Goods", "Personal Care Product")])
    assert r.label == "shampoo bottle" and r.category == "Packaged Goods" and "zero-shot model" in r.source


def test_agreement_boosts_confidence_and_lists_both_sources():
    r = fuse(M("apple"), 0.6, [C("apple", 0.7, "Food and Drink", "Fruit")])
    assert r.confidence > 0.7 and "detector" in r.source and "zero-shot model" in r.source


def test_generic_detector_only_is_flagged_generic():
    r = fuse(M("bottle"), 0.9, [])
    assert r.generic and r.label == "bottle" and r.category == "Household Item"
    assert not fuse(M("apple"), 0.9, []).generic


def test_below_threshold_zero_shot_is_ignored_but_kept_as_alternative():
    r = fuse(M("cup"), 0.8, [C("mug", 0.2, "Household Item", "Kitchenware"), C("glass", 0.1, "Household Item", "Kitchenware")], 0.30)
    assert r.label == "cup" and ("mug", 0.2) in r.alternatives


def test_person_is_never_refined_or_named():
    r = fuse(M("person"), 0.9, [C("doll", 0.9, "Household Item", "Decor and Miscellaneous")])
    assert r.label == "person" and r.category == "Person" and r.source == "detector"


def test_ocr_and_barcode_evidence():
    e = VOCAB.find("shampoo")
    r = fuse(M("bottle"), 0.9, [], 0.3, ocr=(e, 0.85))
    assert r.label == "shampoo bottle" and "OCR text" in r.source
    r = fuse(M("bottle"), 0.5, [], 0.3, barcode=("Peppy Crisps", "Packaged Goods", "Packaged Snack", 0.95))
    assert r.label == "Peppy Crisps" and r.specific_product


def test_disagreeing_strong_sources_lower_confidence():
    solo = fuse(M("apple"), 0.8, [])
    clash = fuse(M("apple"), 0.8, [C("stapler", 0.9, "Stationery", "Office Supplies")])
    assert clash.label == "stapler" and clash.confidence < 0.9 and ("apple", 0.8) in clash.alternatives
    assert solo.confidence == 0.8


def test_nothing_known_is_unknown_other():
    r = fuse(None, 0.0, [])
    assert r.label == "unknown object" and r.category == "Other" and r.confidence == 0.0


# ---------- manager: async, stable-only, cached per track ----------
class CountingRec:
    name = "counting"
    def __init__(self, score=0.8): self.calls, self.score = 0, score
    def classify(self, crop, top_k=3):
        self.calls += 1
        return [Candidate("shampoo bottle", self.score, "Packaged Goods", "Personal Care Product")]


def run_frames(mgr, tr, n, label="bottle", box=(10, 10, 110, 150)):
    frame = np.full((200, 200, 3), 120, np.uint8)
    for _ in range(n):
        tracks = tr.update([Detection(label, 0.9, box)])
        mgr.update(frame, tracks)
    return tracks


def settings(**kw):
    s = Settings.load(path="/none", env={}); s.stable_frames = 3
    for k, v in kw.items(): setattr(s, k, v)
    return s


def test_tier2_waits_for_stability_then_runs_once_per_track():
    rec, tr = CountingRec(), IouTracker()
    mgr = RecognitionManager(rec, TAX, VOCAB, settings(), async_tiers=False)
    run_frames(mgr, tr, 2); assert rec.calls == 0                     # not stable yet
    run_frames(mgr, tr, 30); assert rec.calls == 1                    # cached: never re-analysed every frame


def test_async_result_arrives_without_blocking():
    import time
    class SlowRec(CountingRec):
        def classify(self, crop, top_k=3): time.sleep(0.3); return super().classify(crop, top_k)
    rec, tr = SlowRec(), IouTracker()
    mgr = RecognitionManager(rec, TAX, VOCAB, settings(), async_tiers=True)
    t0 = time.perf_counter(); tracks = run_frames(mgr, tr, 4)
    assert time.perf_counter() - t0 < 0.2                              # update() returned while Tier 2 still running
    assert mgr.recognition(tracks[0]).label == "bottle"                # until the result lands, detector label is used
    time.sleep(0.5)
    assert mgr.recognition(tracks[0]).label == "shampoo bottle" and mgr.zs_ms.summary()["n"] == 1
    mgr.shutdown()


def test_low_confidence_gets_one_retry_only_after_delay():
    clock = [0.0]
    rec, tr = CountingRec(score=0.1), IouTracker()
    mgr = RecognitionManager(rec, TAX, VOCAB, settings(), async_tiers=False, clock=lambda: clock[0])
    run_frames(mgr, tr, 5); assert rec.calls == 1
    clock[0] = 1.0; run_frames(mgr, tr, 3); assert rec.calls == 1      # too soon
    clock[0] = 5.0; run_frames(mgr, tr, 3); assert rec.calls == 2      # one retry
    clock[0] = 50.0; run_frames(mgr, tr, 3); assert rec.calls == 2     # capped


def test_people_are_never_sent_to_tier2():
    rec, tr = CountingRec(), IouTracker()
    mgr = RecognitionManager(rec, TAX, VOCAB, settings(), async_tiers=False)
    run_frames(mgr, tr, 10, label="person"); assert rec.calls == 0


def test_dead_tracks_are_pruned_and_new_track_is_analysed():
    rec, tr = CountingRec(), IouTracker(max_misses=1)
    mgr = RecognitionManager(rec, TAX, VOCAB, settings(), async_tiers=False)
    run_frames(mgr, tr, 5)
    for _ in range(4): mgr.update(np.zeros((200, 200, 3), np.uint8), tr.update([]))
    run_frames(mgr, tr, 5, box=(10, 10, 110, 150)); assert rec.calls - mgr._center.attempts == 2   # old track pruned, new one analysed


def test_center_fallback_when_nothing_is_detected():
    rec, tr = CountingRec(), IouTracker()
    clock = [0.0]
    mgr = RecognitionManager(rec, TAX, VOCAB, settings(), async_tiers=False, clock=lambda: clock[0])
    rng = np.random.default_rng(0)
    for i in range(6):
        clock[0] += 3.0
        mgr.update(rng.integers(0, 255, (200, 200, 3), dtype=np.uint8), tr.update([]))
    r = mgr.center_recognition()
    assert r is not None and r.label == "shampoo bottle" and r.track_id == -1


# ---------- eval script plumbing ----------
def test_eval_script_reports_per_category_accuracy():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("evalrec", ROOT / "scripts" / "eval_recognition.py")
    m = importlib.util.module_from_spec(spec); sys.modules["evalrec"] = m; spec.loader.exec_module(m)
    samples = [("a", "apple"), ("b", "stapler"), ("c", "cup"), ("d", "cups"), ("e", "water bottle")]
    preds = {"a": ("apple", "Food and Drink"), "b": ("pen", "Stationery"), "c": ("mug", "Household Item"), "d": ("cup", "x"), "e": None}
    rep = m.evaluate(samples, lambda x: preds[x], VOCAB)
    assert rep["n"] == 5 and abs(rep["top1"] - 0.4) < 1e-9
    assert rep["per_category"]["Food and Drink"]["top1"] == 1.0 and rep["per_category"]["Stationery"]["top1"] == 0.0
    assert ("stapler", "pen") in rep["wrong"]
