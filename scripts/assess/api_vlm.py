"""Architectures b and c: a commercial VLM as a drop-in for the local Qwen in
assess_risk.py.

Same interface as assess_risk.VLM.ask(), so the identification prompt, crops,
physics check, risk step and context gate are exactly the ones architecture
a uses; only the model changes. Two providers:

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

REPO_ROOT = Path(__file__).resolve().parents[2]
SECRETS = REPO_ROOT / ".secrets"
CACHE = REPO_ROOT / ".cache" / "api_vlm"


class QuotaExhausted(RuntimeError):
    """A daily or billing quota is used up; resume later."""


def _key(provider: str) -> str:
    env = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}[provider]
    return os.environ.get(env) or (SECRETS / f"{provider}_api_key.txt").read_text(
        encoding="utf-8").strip()


class ApiVLM:
    def __init__(self, provider: str = "openai", model: str = "gpt-5.4",
                 min_interval_s: float = 0.5, cache: bool = True, hedge_after_s: float = 0.0):
        """`cache=False` neither reads nor writes the answer cache, for timing
        runs, where a cached answer would hide the network time. Calls may be
        made from several threads (the fast architecture asks about its
        candidates in parallel); counters are guarded by a lock.

        `hedge_after_s > 0` hedges slow calls: if an answer has not arrived
        after that many seconds, the same request is sent again and the first
        answer wins. A few gpt-5.4 calls took 12-13 s where most take 2 s; a
        live frame should not wait for them. Tokens of both requests count."""
        import threading
        self.provider, self.model = provider, model
        self.name = f"{provider}/{model}"
        self.min_interval_s = min_interval_s
        self.use_cache = cache
        self.hedge_after_s = hedge_after_s
        self.hedged = 0
        self._lock = threading.Lock()
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(16)
        self._last = 0.0
        self.calls = 0
        self.cached_hits = 0
        self.usage = {"prompt": 0, "output": 0, "reasoning": 0}
        self.errors: list[str] = []  # transient failures that were retried
        self.durations: list[float] = []  # seconds per completed call
        if provider == "openai":
            from openai import OpenAI
            import httpx
            # Keep connections open between frames. Each new connection costs
            # a DNS lookup, and this laptop's resolver takes ~11 s on a cache
            # miss (curl: 11.3 s, then 0.2 s); httpx's default 5 s keep-alive
            # made about one frame in seven wait for it.
            http = httpx.Client(limits=httpx.Limits(max_connections=16, max_keepalive_connections=16,
                                                    keepalive_expiry=600),
                                timeout=httpx.Timeout(120, connect=30))
            self.client = OpenAI(api_key=_key("openai"), max_retries=0, http_client=http)
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

    def _count(self, fut):
        """Done-callback: count every completed request, the losing hedge too."""
        if fut.exception() is None:
            _, p, o, rsn = fut.result()
            with self._lock:
                self.calls += 1
                self.usage["prompt"] += p
                self.usage["output"] += o
                self.usage["reasoning"] += rsn

    def _hedged(self, call, blobs, text) -> str:
        from concurrent.futures import FIRST_COMPLETED, wait
        futs = [self._pool.submit(call, blobs, text)]
        futs[0].add_done_callback(self._count)
        if self.hedge_after_s:
            done, _ = wait(futs, timeout=self.hedge_after_s)
            if not done:
                with self._lock:
                    self.hedged += 1
                futs.append(self._pool.submit(call, blobs, text))
                futs[1].add_done_callback(self._count)
        pending = set(futs)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                if f.exception() is None:
                    return f.result()[0]
        raise futs[0].exception()

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
        if self.use_cache and cf.is_file():
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
                out = self._hedged(call, blobs, text)
                dt = time.time() - t0
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                with self._lock:
                    self.errors.append(f"{type(exc).__name__}: {msg[:160]}")
                if "insufficient_quota" in msg or re.search(r"PerDay", msg):
                    raise QuotaExhausted(msg[:300]) from exc
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate_limit" in msg:
                    m = re.search(r"(?:retry(?:Delay)?|try again in)['\"]?\s*[:=]?\s*['\"]?([\d.]+)",
                                  msg, flags=re.I)
                    time.sleep(float(m.group(1)) + 1 if m else 20)
                    continue
                if any(c in msg for c in ("500", "502", "503", "UNAVAILABLE", "timed out")) or \
                        type(exc).__name__ in ("APIConnectionError", "APITimeoutError",
                                               "InternalServerError"):
                    # Exponential backoff from 0.5 s: a fixed 10 s wait cost a live
                    # frame 12 s for one transient server error.
                    time.sleep(min(8.0, 0.5 * 2 ** attempt))
                    continue
                raise
        else:
            raise RuntimeError(f"{self.name}: call failed after retries")
        with self._lock:
            self.durations.append(round(dt, 2))
        if self.use_cache:
            cf.write_text(json.dumps({"text": out, "provider": self.provider, "model": self.model,
                                      "seconds": round(dt, 2)}), encoding="utf-8")
        return out
