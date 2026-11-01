"""AI engine for SmartNotes AI (uses the free Groq API)."""

import json
import random
import re
import time

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# The first model is tried first. If it fails, the next one is used.
DEFAULT_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
]

# Longer notes are sampled from across the whole document to stay inside the free limits.
MAX_CHARS = 12000

SYSTEM_PROMPT = (
    "You are SmartNotes AI, a friendly study assistant. Use ONLY the student's notes "
    "that are given to you. Write in simple, clear English."
)


class AIEngineError(Exception):
    """Raised when the AI cannot complete a request. The message is safe to show to the student."""


def was_truncated(text):
    """True when the notes are too long and only samples of them will be sent to the AI."""
    return len(text or "") > MAX_CHARS


def prepare_text(text, limit=MAX_CHARS):
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    parts = 6
    size = limit // parts
    step = (len(text) - size) / (parts - 1)
    chunks = [text[int(i * step): int(i * step) + size] for i in range(parts)]
    return "\n\n[...]\n\n".join(chunks)


def _parse_json(raw):
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except ValueError:
        match = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


class AIEngine:
    def __init__(self, api_key, models=None):
        key = (api_key or "").strip().strip('"').strip("'")
        if not key:
            raise AIEngineError("Missing GROQ_API_KEY.")
        self.api_key = key
        self.models = list(models) if models else list(DEFAULT_MODELS)

    # ------------------------------------------------------------------
    # Core call to Groq
    # ------------------------------------------------------------------
    def _chat(self, messages, max_tokens=2000, temperature=0.3):
        problem = ""
        rate_limited = False

        for model in self.models:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if "gpt-oss" in model:
                payload["reasoning_effort"] = "low"

            for attempt in range(2):
                try:
                    response = requests.post(
                        GROQ_URL,
                        headers={"Authorization": f"Bearer {self.api_key}",
                                 "Content-Type": "application/json"},
                        json=payload,
                        timeout=90,
                    )
                except requests.RequestException:
                    raise AIEngineError(
                        "Could not reach the AI service. Please check your internet and try again."
                    )

                if response.status_code == 200:
                    try:
                        content = response.json()["choices"][0]["message"].get("content") or ""
                    except Exception:
                        content = ""
                    if content.strip():
                        return content.strip()
                    problem = f"{model}: empty answer"
                    break

                if response.status_code == 401:
                    raise AIEngineError(
                        "Groq rejected the API key. Please check GROQ_API_KEY in the Secrets box."
                    )

                if response.status_code == 429:
                    rate_limited = True
                    if attempt == 0:
                        time.sleep(6)
                        continue
                    problem = f"{model}: rate limit"
                    break

                problem = f"{model}: HTTP {response.status_code} {response.text[:160]}"
                break

        if rate_limited:
            raise AIEngineError(
                "The free Groq limit was reached for now. Please wait a minute and try again."
            )
        raise AIEngineError(f"The AI could not process this request. ({problem})")

    def _ask(self, instruction, text, max_tokens=2000):
        notes = prepare_text(text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{instruction}\n\nSTUDENT NOTES:\n\"\"\"\n{notes}\n\"\"\""},
        ]
        return self._chat(messages, max_tokens=max_tokens)

    def _json_list(self, instruction, text, max_tokens):
        last_error = None
        for _ in range(2):
            raw = self._ask(instruction, text, max_tokens=max_tokens)
            try:
                data = _parse_json(raw)
                if isinstance(data, dict):
                    data = next((v for v in data.values() if isinstance(v, list)), None)
                if isinstance(data, list) and data:
                    return data
            except ValueError as error:
                last_error = error
        raise AIEngineError("The AI answer could not be read. Please press the button again.")

    # ------------------------------------------------------------------
    # Features used by app.py
    # ------------------------------------------------------------------
    def summarize(self, text, length="Medium"):
        styles = {
            "Short": "Write a SHORT summary: a quick overview in about 5 to 7 sentences.",
            "Medium": "Write a MEDIUM summary: cover the main topics with short headings and bullet points.",
            "Detailed": (
                "Write a DETAILED summary: explain every main topic under clear headings, "
                "then finish with a short recap."
            ),
        }
        instruction = (
            f"{styles.get(length, styles['Medium'])}\n"
            "Use Markdown headings and bullet points. Do not add information that is not in the notes."
        )
        tokens = {"Short": 700, "Medium": 1400, "Detailed": 2200}.get(length, 1400)
        return self._ask(instruction, text, max_tokens=tokens)

    def important_points(self, text):
        instruction = (
            "Pull out the most important points from the notes. Use Markdown with these sections:\n"
            "## Key concepts\n## Definitions\n## Key terms\n"
            "Use short bullet points. Skip a section if the notes have nothing for it."
        )
        return self._ask(instruction, text, max_tokens=1800)

    def generate_mcqs(self, text, count=5):
        instruction = (
            f"Create exactly {count} multiple-choice questions from the notes.\n"
            "Return ONLY a JSON array, with no other text. Each item must look like:\n"
            '{"question": "...", "options": ["...", "...", "...", "..."], '
            '"answer_index": 0, "explanation": "one short sentence"}\n'
            "Each question has exactly 4 options. answer_index is the number (0 to 3) of the correct option."
        )
        raw_items = self._json_list(instruction, text, max_tokens=min(4000, 500 + 170 * count))

        questions = []
        for item in raw_items:
            try:
                options = [str(o).strip() for o in item["options"]]
                if len(options) != 4:
                    continue
                index = item.get("answer_index")
                if isinstance(index, str) and index.strip().upper() in ("A", "B", "C", "D"):
                    index = "ABCD".index(index.strip().upper())
                index = int(index)
                if not 0 <= index <= 3:
                    continue
                correct = options[index]
                random.shuffle(options)  # so the right answer is not always in the same place
                questions.append({
                    "question": str(item["question"]).strip(),
                    "options": options,
                    "answer_index": options.index(correct),
                    "explanation": str(item.get("explanation", "")).strip(),
                })
            except Exception:
                continue

        if not questions:
            raise AIEngineError("The AI could not make valid questions. Please try again.")
        return questions[:count]

    def generate_flashcards(self, text, count=10):
        instruction = (
            f"Create exactly {count} flashcards from the notes.\n"
            "Return ONLY a JSON array, with no other text. Each item must look like:\n"
            '{"question": "...", "answer": "a short, clear answer"}'
        )
        raw_items = self._json_list(instruction, text, max_tokens=min(4000, 500 + 120 * count))

        cards = []
        for item in raw_items:
            try:
                question = str(item["question"]).strip()
                answer = str(item["answer"]).strip()
                if question and answer:
                    cards.append({"question": question, "answer": answer})
            except Exception:
                continue

        if not cards:
            raise AIEngineError("The AI could not make valid flashcards. Please try again.")
        return cards[:count]

    def answer_question(self, text, question, history=None):
        notes = prepare_text(text)
        messages = [{
            "role": "system",
            "content": (
                SYSTEM_PROMPT
                + " Answer the student's question using ONLY the notes below. "
                "If the answer is not in the notes, say that the notes do not cover it.\n\n"
                f"STUDENT NOTES:\n\"\"\"\n{notes}\n\"\"\""
            ),
        }]
        for message in (history or [])[-6:]:
            if message.get("role") in ("user", "assistant"):
                messages.append({"role": message["role"], "content": message["content"]})
        messages.append({"role": "user", "content": question})
        return self._chat(messages, max_tokens=1200)
