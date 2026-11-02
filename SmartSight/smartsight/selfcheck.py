"""Start-up self-check: which engine was actually selected, spoken and printed."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SelfCheck:
    items: list = field(default_factory=list)      # (name, ok, message)

    def add(self, name: str, ok: bool, msg: str) -> None:
        self.items.append((name, ok, msg))

    @property
    def spoken(self) -> str:
        return "Self check. " + " ".join(m if m.endswith(".") else m + "." for _, _, m in self.items)

    @property
    def printed(self) -> str:
        return "\n".join(f"  [{'OK' if ok else '!!'}] {n}: {m}" for n, ok, m in self.items)


def run_selfcheck(pipe, status: dict | None = None) -> SelfCheck:
    st = status or getattr(pipe, "component_status", {}) or {}
    sc = SelfCheck()
    sc.add("camera", pipe.camera_status == "running", "Camera is ready" if pipe.camera_status == "running" else "The camera is not running")
    d = st.get("detector", "")
    sc.add("detector", pipe.detector is not None, "Object detection is ready" if pipe.detector is not None
           else "Object detection is not available, so I can only read text")
    o = st.get("ocr", "")
    sc.add("ocr", pipe.ocr is not None, f"Text reading uses {o.split('(')[-1].rstrip(')')}" if pipe.ocr is not None and "(" in o
           else ("Text reading is ready" if pipe.ocr is not None else "Text reading is not available"))
    z = st.get("zero_shot", "")
    has_zs = pipe.recog.recognizer is not None
    sc.add("recognition", has_zs, "Open vocabulary recognition is on" if has_zs
           else "Open vocabulary recognition is not installed, so I will name only common objects")
    sp = st.get("speech", "")
    sc.add("speech", "Console" not in sp, "Speech is ready" if "Console" not in sp else "No voice engine found. Speech will only be printed")
    cloud = pipe.cloud_status() if hasattr(pipe, "cloud_status") else ""
    if cloud.startswith("ON"):
        sc.add("online_ai", True, "Online AI is on. Camera photos may be sent when I need help identifying something")
    elif "permission" in cloud:
        sc.add("online_ai", False, "Online AI is waiting for your permission, so I am fully offline")
    elif "paused" in cloud:
        sc.add("online_ai", True, "Online AI is paused. Nothing is sent")
    else:
        sc.add("online_ai", False, "Online AI has no API key, so I am fully offline")
    for w in getattr(pipe.s, "load_warnings", [])[:2]:
        sc.add("config", False, f"Settings problem: {w}")
    return sc
