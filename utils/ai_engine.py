"""Groq AI engine for SmartNotes AI."""

import os
import time

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Pehla model pehle try hota hai, fail ho to agla.
DEFAULT_MODELS = {
    "fast": "openai/gpt-oss-20b",
    "smart": "openai/gpt-oss-120b",
    "default": "openai/gpt-oss-20b",
}
FALLBACK_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile"]


class AIEngineError(Exception):
    """AI call fail ho to yeh error aata hai (asal wajah message mein hoti hai)."""


def _get_key(api_key=None):
    if api_key:
        return api_key
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


class AIEngine:
    def __init__(self, api_key=None, model=None, *args, **kwargs):
        self.api_key = _get_key(api_key if isinstance(api_key, str) else None)
        if not self.api_key:
            raise AIEngineError("GROQ_API_KEY set nahi hai.")
        if isinstance(model, str) and model:
            self.models = [model] + [m for m in FALLBACK_MODELS if m != model]
        else:
            self.models = list(FALLBACK_MODELS)
        self.last_truncated = False
        self.last_model = None

    def generate(self, prompt, system_prompt="You are a helpful study assistant.",
                 max_tokens=1500, temperature=0.3, **kwargs):
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": str(prompt)}]

        errors = []
        for model in self.models:
            for attempt in range(2):
                try:
                    r = requests.post(
                        GROQ_URL,
                        headers={"Authorization": f"Bearer {self.api_key}",
                                 "Content-Type": "application/json"},
                        json={"model": model, "messages": messages,
                              "max_tokens": max_tokens, "temperature": temperature},
                        timeout=90,
                    )
                except requests.RequestException as e:
                    errors.append(f"{model}: network error {e}")
                    break
                if r.status_code == 200:
                    data = r.json()
                    choice = data["choices"][0]
                    self.last_truncated = choice.get("finish_reason") == "length"
                    self.last_model = model
                    return choice["message"].get("content") or ""
                if r.status_code == 429 and attempt == 0:
                    time.sleep(5)  # rate limit: thora ruk ke dobara
                    continue
                errors.append(f"{model}: HTTP {r.status_code} {r.text[:200]}")
                if r.status_code == 401:
                    raise AIEngineError("Groq API key galat hai (401). " + r.text[:200])
                break
        raise AIEngineError(" | ".join(errors))

    # Alag alag naam, agar app.py inme se koi use karti ho
    complete = chat = ask = run = summarize = generate


def was_truncated(obj=None):
    """True agar jawab max_tokens par kat gaya ho."""
    if isinstance(obj, AIEngine):
        return obj.last_truncated
    return bool(obj)
