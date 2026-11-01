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
# The bigger model writes better summaries, so it is tried first for summaries.
BEST_MODEL = "openai/gpt-oss-120b"

# Notes up to this size go to the AI in one piece.
MAX_CHARS = 16000
# Size of one part when long notes are split up.
PART_CHARS = 9000

SYSTEM_PROMPT = (
    "You are SmartNotes AI, a friendly study assistant. Use ONLY the student's notes "
    "that are given to you. Write in simple, clear English."
)


class AIEngineError(Exception):
    """Raised when the AI cannot complete a request. The message is safe to show to the student."""


def was_truncated(text):
    """True when the notes are very long and only part of them goes to the AI in one call."""
    return len(text or "") > MAX_CHARS


def prepare_text(text, limit=MAX_CHARS):
    """If text is longer than limit, take samples from across the whole text."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    parts = 6
    size = limit // parts
    step = (len(text) - size) / (parts - 1)
    chunks = [text[int(i * step): int(i * step) + size] for i in range(parts)]
    return "\n\n[...]\n\n".join(chunks)


def split_parts(text, n, limit=PART_CHARS):
    """Cut the text into n consecutive parts (each at most `limit` characters)."""
    text = (text or "").strip()
    size = len(text) / n
    return [prepare_text(text[int(i * size): int((i + 1) * size)], limit) for i in range(n)]


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


def _clean_mcqs(raw_items):
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
    return questions


def _clean_cards(raw_items):
    cards = []
    for item in raw_items:
        try:
            question = str(item["question"]).strip()
            answer = str(item["answer"]).strip()
            if question and answer:
                cards.append({"question": question, "answer": answer})
        except Exception:
            continue
    return cards


# ----------------------------------------------------------------------
# Summary options
# ----------------------------------------------------------------------
SUMMARY_LENGTHS = {
    "Short": (
        "Write a SHORT summary: one overview sentence, then 5 to 8 bullet points with the most "
        "important ideas.",
        900,
    ),
    "Medium": (
        "Write a MEDIUM summary. Start with a 2-sentence overview under '## Big picture'. Then go "
        "through the topics in order, each with a ## heading and 3 to 6 bullet points covering the "
        "key ideas, definitions and examples.",
        1900,
    ),
    "Lengthy": (
        "Write a LENGTHY, detailed summary. Start with a short overview under '## Big picture'. "
        "Then explain EVERY topic in order under its own ## heading, with clear explanations, "
        "definitions, steps, formulas and examples from the notes. Finish with a '## Quick recap' "
        "section with the 5 to 10 points a student must remember.",
        3500,
    ),
}

SUMMARY_LEVELS = {
    "Easy": (
        "LANGUAGE LEVEL: EASY. Use very simple everyday words and short sentences, as if "
        "explaining to a beginner. Explain every technical term in plain words and add a small "
        "everyday example where it helps."
    ),
    "Normal": (
        "LANGUAGE LEVEL: NORMAL. Use clear student-level English. Keep the important technical "
        "terms, each with a short explanation."
    ),
    "Hard": (
        "LANGUAGE LEVEL: HARD. Use precise technical and academic language. Keep the exact "
        "terminology, formulas and details. Assume the reader already knows the basics."
    ),
}

SUMMARY_RULES = (
    "Quality rules: make it exam-ready; put key terms in **bold**; keep the order of the "
    "document; cover the WHOLE document, not just the beginning; no filler and no repetition; "
    "do not invent facts; keep important formulas or code if the notes have them. Use Markdown."
)


# ----------------------------------------------------------------------
# Explained notes (whole document, part by part)
# ----------------------------------------------------------------------
EXPLAIN_PART_CHARS = 7000   # one part is explained at a time
MAX_EXPLAIN_PARTS = 15

EXPLAIN_STYLES = {
    "Mixed (best)": (
        "EXPLANATION STYLE: MIXED. Explain each topic properly, do not just list it. For every "
        "topic use a DIFFERENT teaching technique where it fits, and change the technique from "
        "topic to topic: a plain-words explanation, a real-life example, an everyday analogy, a "
        "step-by-step walk-through, a small comparison table, or a mini question-and-answer. "
        "End each topic with a short line starting with '🧠 Remember:' that holds the key point."
    ),
    "Simple words": (
        "EXPLANATION STYLE: SIMPLE WORDS. Explain every topic in a few short paragraphs using the "
        "simplest words possible, as if talking to a friend."
    ),
    "With real-life examples": (
        "EXPLANATION STYLE: REAL-LIFE EXAMPLES. Explain each idea clearly, then give at least one "
        "real-life example for every topic."
    ),
    "Step by step": (
        "EXPLANATION STYLE: STEP BY STEP. Break every topic into numbered steps, and explain what "
        "happens in each step and why."
    ),
    "Analogies": (
        "EXPLANATION STYLE: ANALOGIES. Compare every idea with something from everyday life, then "
        "link the comparison back to what the notes say."
    ),
    "Question and answer": (
        "EXPLANATION STYLE: QUESTION AND ANSWER. Write each topic as 2 to 4 questions a student "
        "might ask, each followed by a clear answer."
    ),
    "Exam focused": (
        "EXPLANATION STYLE: EXAM FOCUSED. For every topic give: the explanation, the key points to "
        "memorise, and 1 or 2 likely exam questions with short model answers."
    ),
}


def split_sequential(text, size=EXPLAIN_PART_CHARS):
    """Cut the text into consecutive parts of about `size` characters, breaking at line ends."""
    parts, current = [], ""
    for line in (text or "").splitlines(keepends=True):
        while len(line) > size:  # a very long line
            if current:
                parts.append(current)
                current = ""
            parts.append(line[:size])
            line = line[size:]
        if current and len(current) + len(line) > size:
            parts.append(current)
            current = ""
        current += line
    if current.strip():
        parts.append(current)
    return [p.strip() for p in parts if p.strip()]


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
    def _chat(self, messages, max_tokens=2000, temperature=0.3, models=None):
        problem = ""
        rate_limited = False

        for model in (models or self.models):
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
                        time.sleep(8)
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

    def _ask(self, instruction, text, max_tokens=2000, models=None):
        notes = prepare_text(text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{instruction}\n\nSTUDENT NOTES:\n\"\"\"\n{notes}\n\"\"\""},
        ]
        return self._chat(messages, max_tokens=max_tokens, models=models)

    def _json_list(self, instruction, text, max_tokens):
        for _ in range(2):
            raw = self._ask(instruction, text, max_tokens=max_tokens)
            try:
                data = _parse_json(raw)
                if isinstance(data, dict):
                    data = next((v for v in data.values() if isinstance(v, list)), None)
                if isinstance(data, list) and data:
                    return data
            except ValueError:
                pass
        raise AIEngineError("The AI answer could not be read. Please press the button again.")

    def _generate_many(self, text, count, per_batch, make_instruction, tokens_for, clean):
        """Make many items by splitting the notes into parts, so every part of the notes is used."""
        n = max(1, -(-count // per_batch))
        parts = split_parts(text, n) if n > 1 else [prepare_text(text)]
        items = []
        for i, part in enumerate(parts):
            want = count // n + (1 if i < count % n else 0)
            if want == 0:
                continue
            try:
                raw = self._json_list(make_instruction(want), part, tokens_for(want))
                items.extend(clean(raw)[:want])
            except AIEngineError:
                if items:  # keep what we already have
                    break
                raise
            if i < len(parts) - 1:
                time.sleep(3)  # be gentle with the free limits
        if not items:
            raise AIEngineError("The AI could not make valid results. Please try again.")
        return items

    # ------------------------------------------------------------------
    # Features used by app.py
    # ------------------------------------------------------------------
    def summarize(self, text, length="Medium", level="Normal"):
        length = {"Detailed": "Lengthy", "Long": "Lengthy"}.get(length, length)
        style, tokens = SUMMARY_LENGTHS.get(length, SUMMARY_LENGTHS["Medium"])
        level_text = SUMMARY_LEVELS.get(level, SUMMARY_LEVELS["Normal"])
        instruction = f"{style}\n{level_text}\n{SUMMARY_RULES}"
        best = [BEST_MODEL] + [m for m in self.models if m != BEST_MODEL]

        text = (text or "").strip()
        if len(text) <= MAX_CHARS:
            return self._ask(instruction, text, max_tokens=tokens, models=best)

        # Long notes: condense each part first, then write the final summary from all parts.
        n = min(10, -(-len(text) // PART_CHARS))
        notes = []
        for i, part in enumerate(split_parts(text, n), start=1):
            notes.append(f"PART {i}:\n" + self._ask(
                "Condense this part of the notes into compact bullet points. Keep every key term, "
                "definition, number and example.",
                part,
                max_tokens=500,
            ))
            time.sleep(3)
        combined = "\n\n".join(notes)
        return self._ask(
            "These are condensed notes from consecutive parts of ONE document.\n" + instruction,
            combined,
            max_tokens=tokens,
            models=best,
        )

    def explain_notes(self, text, style="Mixed (best)", level="Normal", progress=None):
        """Explain the WHOLE document, one part after the other, so nothing is skipped."""
        text = (text or "").strip()
        size = max(EXPLAIN_PART_CHARS, -(-len(text) // MAX_EXPLAIN_PARTS))
        parts = split_sequential(text, size)
        if not parts:
            raise AIEngineError("There is no text to explain.")

        style_text = EXPLAIN_STYLES.get(style, EXPLAIN_STYLES["Mixed (best)"])
        level_text = SUMMARY_LEVELS.get(level, SUMMARY_LEVELS["Normal"])
        best = [BEST_MODEL] + [m for m in self.models if m != BEST_MODEL]

        output = []
        seen = []
        total = len(parts)
        for i, part in enumerate(parts, start=1):
            earlier = ""
            if seen:
                earlier = (
                    "Topics already explained in earlier parts (do not repeat them): "
                    + "; ".join(seen[-15:]) + "\n"
                )
            instruction = (
                f"This is part {i} of {total} of the student's notes. Explain EVERYTHING in this "
                "part thoroughly, in the order it appears. Do not skip any topic.\n"
                f"{earlier}{style_text}\n{level_text}\n"
                "Use a ## heading for each topic and put key terms in **bold**. Do not invent facts. "
                "Do not write an introduction or a conclusion for the whole document."
            )
            try:
                result = self._ask(instruction, part, max_tokens=3000, models=best)
            except AIEngineError as error:
                if output:
                    output.append(
                        f"> ⚠️ Part {i} of {total} could not be explained ({error}). "
                        "Press the button again to retry."
                    )
                    break
                raise
            output.append(result)
            seen += re.findall(r"^#{1,3}\s+(.+)$", result, re.MULTILINE)
            if progress:
                progress(i, total, "\n\n".join(output))
            if i < total:
                time.sleep(6)  # be gentle with the free limits
        return "\n\n---\n\n".join(output)

    def important_points(self, text):
        instruction = (
            "Pull out the most important points from the notes. Use Markdown with these sections:\n"
            "## Key concepts\n## Definitions\n## Key terms\n"
            "Use short bullet points. Skip a section if the notes have nothing for it."
        )
        return self._ask(instruction, text, max_tokens=2000)

    def generate_mcqs(self, text, count=5):
        def instruction(want):
            return (
                f"Create exactly {want} multiple-choice questions from the notes.\n"
                "Return ONLY a JSON array, with no other text. Each item must look like:\n"
                '{"question": "...", "options": ["...", "...", "...", "..."], '
                '"answer_index": 0, "explanation": "one short sentence"}\n'
                "Each question has exactly 4 options. answer_index is the number (0 to 3) of the "
                "correct option. Do not repeat questions."
            )

        return self._generate_many(
            text, count, per_batch=10,
            make_instruction=instruction,
            tokens_for=lambda want: min(4000, 600 + 180 * want),
            clean=_clean_mcqs,
        )

    def generate_flashcards(self, text, count=10):
        def instruction(want):
            return (
                f"Create exactly {want} flashcards from the notes.\n"
                "Return ONLY a JSON array, with no other text. Each item must look like:\n"
                '{"question": "...", "answer": "a short, clear answer"}\n'
                "Do not repeat questions."
            )

        return self._generate_many(
            text, count, per_batch=15,
            make_instruction=instruction,
            tokens_for=lambda want: min(4000, 600 + 130 * want),
            clean=_clean_cards,
        )

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
