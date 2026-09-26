"""Path 3: a commercial VLM as a drop-in for the local Qwen in assess_risk.py.

Same interface as assess_risk.VLM.ask(), so the identification prompt, crops,
physics check, risk step and context gate are exactly the ones paths 1 and 2
use; only the model changes. Two providers:

  openai   OpenAI Chat Completions (default), key in .secrets/openai_api_key.txt
           or OPENAI_API_KEY
  gemini   Google Gemini, key in .secrets/gemini_api_key.txt or GEMINI_API_KEY.
           The free tier allows 20 requests per model per day, too few for a
           run (about 520 calls); it needs billing enabled.

Keys are never printed. Every answer is cached on disk by a hash of provider,
model, prompt and image bytes, so an interrupted run resumes without paying
twice. Token usage is summed per run so cost can be computed.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRETS = REPO_ROOT / ".secrets"
CACHE = REPO_ROOT / "results" / "risk-assessment" / "path3_commercial_vlm" / "api_cache"


class QuotaExhausted(RuntimeError):
    """A daily or billing quota is used up; resume later."""


def _key(provider: str) -> str:
    env = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}[provider]
    return os.environ.get(env) or (SECRETS / f"{provider}_api_key.txt").read_text(
        encoding="utf-8").strip()


class ApiVLM:
    def __init__(self, provider: str = "openai", model: str = "gpt-5.4",
                 min_interval_s: float = 0.5):
        self.provider, self.model = provider, model
        self.name = f"{provider}/{model}"
        self.min_interval_s = min_interval_s
        self._last = 0.0
        self.calls = 0
        self.cached_hits = 0
        self.usage = {"prompt": 0, "output": 0, "reasoning": 0}
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=_key("openai"), max_retries=0)
        elif provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=_key("gemini"))
        else:
            raise ValueError(provider)
        CACHE.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _jpeg(im: Image.Image) -> bytes:
        b = io.BytesIO()
        im.convert("RGB").save(b, format="JPEG", quality=92)
        return b.getvalue()

    # -------------------------------------------------------------- providers
    def _openai(self, blobs, text):
        content = [{"type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode(),
                                  "detail": "high"}} for b in blobs]
        content.append({"type": "text", "text": text})
        r = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"}, max_completion_tokens=4096)
        u = r.usage
        det = getattr(u, "completion_tokens_details", None)
        return (r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens,
                getattr(det, "reasoning_tokens", 0) or 0)

    def _gemini(self, blobs, text):
        from google.genai import types
        parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in blobs] + [text]
        r = self.client.models.generate_content(
            model=self.model, contents=parts,
            config=types.GenerateContentConfig(response_mime_type="application/json",
                                               max_output_tokens=4096))
        u = r.usage_metadata
        return (r.text or "", getattr(u, "prompt_token_count", 0) or 0,
                getattr(u, "candidates_token_count", 0) or 0,
                getattr(u, "thoughts_token_count", 0) or 0)

    # -------------------------------------------------------------- interface
    def ask(self, images: list[Image.Image], text: str, max_new_tokens: int,
            prefix: str = "", thinking: bool = False) -> str:
        """`prefix` and `thinking` exist for interface parity with the local
        VLM (which needs a prefilled answer to stay in JSON). Here JSON is
        enforced by the provider's JSON mode, and the model's own default
        reasoning is left on: accuracy is the point of path 3."""
        blobs = [self._jpeg(im) for im in images]
        h = hashlib.sha256(f"{self.provider}/{self.model}".encode() + text.encode())
        for b in blobs:
            h.update(b)
        cf = CACHE / f"{h.hexdigest()[:40]}.json"
        if cf.is_file():
            self.cached_hits += 1
            return json.loads(cf.read_text(encoding="utf-8"))["text"]
        call = self._openai if self.provider == "openai" else self._gemini
        for attempt in range(12):
            wait = self.min_interval_s - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                t0 = time.time()
                out, p, o, rsn = call(blobs, text)
                dt = time.time() - t0
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if "insufficient_quota" in msg or re.search(r"PerDay", msg):
                    raise QuotaExhausted(msg[:300]) from exc
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate_limit" in msg:
                    m = re.search(r"(?:retry(?:Delay)?|try again in)['\"]?\s*[:=]?\s*['\"]?([\d.]+)",
                                  msg, flags=re.I)
                    time.sleep(float(m.group(1)) + 1 if m else 20)
                    continue
                if any(c in msg for c in ("500", "502", "503", "UNAVAILABLE", "timed out")):
                    time.sleep(10)
                    continue
                raise
        else:
            raise RuntimeError(f"{self.name}: call failed after retries")
        self.calls += 1
        self.usage["prompt"] += p
        self.usage["output"] += o
        self.usage["reasoning"] += rsn
        cf.write_text(json.dumps({"text": out, "provider": self.provider, "model": self.model,
                                  "seconds": round(dt, 2), "prompt_tokens": p,
                                  "output_tokens": o, "reasoning_tokens": rsn}),
                      encoding="utf-8")
        return out
