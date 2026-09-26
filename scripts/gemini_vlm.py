"""Path 3: a commercial VLM (Google Gemini, free tier) as a drop-in for the
local Qwen in assess_risk.py. Same interface as assess_risk.VLM.ask(), so the
identification prompt, crops, physics check, risk step and context gate are
exactly the ones paths 1 and 2 use; only the model changes.

The key is read from .secrets/gemini_api_key.txt (or GEMINI_API_KEY) and never
printed. Every answer is cached on disk by a hash of the model, prompt and
image bytes, so an interrupted run (rate limit, daily quota) resumes without
spending quota twice. Token usage is logged per call so a paid-tier cost can be
estimated from it.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = REPO_ROOT / ".secrets" / "gemini_api_key.txt"
CACHE = REPO_ROOT / "results" / "risk-assessment" / "path3_commercial_vlm" / "api_cache"
DEFAULT_MODEL = "gemini-3.8-flash"


class QuotaExhausted(RuntimeError):
    """The free tier's daily quota is used up; resume the run tomorrow."""


class GeminiVLM:
    def __init__(self, model: str = DEFAULT_MODEL, min_interval_s: float = 4.0):
        from google import genai
        from google.genai import types
        key = os.environ.get("GEMINI_API_KEY") or KEY_FILE.read_text(encoding="utf-8").strip()
        self.client = genai.Client(api_key=key)
        self.types = types
        self.model = model
        self.name = model
        self.min_interval_s = min_interval_s
        self._last = 0.0
        self.calls = 0
        self.cached_hits = 0
        self.usage = {"prompt": 0, "output": 0, "thoughts": 0}
        CACHE.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _jpeg(im: Image.Image) -> bytes:
        b = io.BytesIO()
        im.convert("RGB").save(b, format="JPEG", quality=92)
        return b.getvalue()

    def ask(self, images: list[Image.Image], text: str, max_new_tokens: int,
            prefix: str = "", thinking: bool = False) -> str:
        """`prefix` and `thinking` exist for interface parity with the local
        VLM (which needs a prefilled answer to stay in JSON); here JSON is
        enforced with response_mime_type instead, and the model's own default
        thinking is left on, since accuracy is the point of path 3."""
        blobs = [self._jpeg(im) for im in images]
        h = hashlib.sha256(self.model.encode() + text.encode())
        for b in blobs:
            h.update(b)
        cf = CACHE / f"{h.hexdigest()[:40]}.json"
        if cf.is_file():
            self.cached_hits += 1
            return json.loads(cf.read_text(encoding="utf-8"))["text"]
        types = self.types
        parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in blobs] + [text]
        config = types.GenerateContentConfig(response_mime_type="application/json",
                                             max_output_tokens=4096)
        for attempt in range(12):
            wait = self.min_interval_s - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                t0 = time.time()
                resp = self.client.models.generate_content(model=self.model, contents=parts,
                                                           config=config)
                dt = time.time() - t0
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    if re.search(r"PerDay|per day|daily", msg, flags=re.I):
                        raise QuotaExhausted(msg[:400]) from exc
                    m = re.search(r"retry(?:Delay)?['\"]?\s*[:=]\s*['\"]?(\d+)", msg)
                    time.sleep(int(m.group(1)) + 1 if m else 20)
                    continue
                if "500" in msg or "503" in msg or "UNAVAILABLE" in msg:
                    time.sleep(10)
                    continue
                raise
        else:
            raise RuntimeError("Gemini call failed after retries")
        out = resp.text or ""
        u = resp.usage_metadata
        rec = {"text": out, "model": self.model, "seconds": round(dt, 2),
               "prompt_tokens": getattr(u, "prompt_token_count", None),
               "output_tokens": getattr(u, "candidates_token_count", None),
               "thought_tokens": getattr(u, "thoughts_token_count", None)}
        for k, f in (("prompt", "prompt_tokens"), ("output", "output_tokens"),
                     ("thoughts", "thought_tokens")):
            self.usage[k] += rec[f] or 0
        self.calls += 1
        cf.write_text(json.dumps(rec), encoding="utf-8")
        return out
