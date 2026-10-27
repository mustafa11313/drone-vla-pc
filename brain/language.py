"""
language.py  -  commands in Arabic or English end up as the same instruction (idea V8).

The model was trained on English sentences. When the phone sends Arabic (MIC with "lang ar"),
the bridge turns it into the English training sentence before the model sees it:

  1. Word list (offline, instant): Arabic words for land / pad / red  ->  the training
     sentence. Covers "انزل على المنصة الحمراء", "اهبط على الهدف الأحمر", ...
  2. Otherwise, if ANTHROPIC_API_KEY is set on the laptop, Claude translates the command into
     one short English instruction (pip install anthropic).
  3. Otherwise the training sentence is used and the event log says so.

The log keeps both the words heard and the instruction used, so the thesis can show that
Arabic and English commands lead to the same behaviour.
"""
import re

TRAINING_SENTENCE = "Take off, locate the red landing pad, and land on it."
ARABIC = re.compile(r"[؀-ۿ]")

# Arabic stems: land / descend / go down, pad / target / platform / mat, red
LAND = ["هبط", "اهبط", "انزل", "نزول", "هبوط", "حط", "ينزل", "يهبط"]
PAD = ["منصة", "المنصة", "منصه", "هدف", "الهدف", "مربع", "سجادة", "علامة"]
RED = ["احمر", "أحمر", "الأحمر", "الاحمر", "حمراء", "الحمراء"]
MODEL = "claude-opus-5-5"


def is_arabic(text):
    return bool(text and ARABIC.search(text))


def from_word_list(text):
    """Training sentence when the Arabic command is clearly 'land on the (red) pad', else None."""
    t = text.replace("ـ", "")
    land = any(w in t for w in LAND)
    target = any(w in t for w in PAD) or any(w in t for w in RED)
    return TRAINING_SENTENCE if land and target else None


def with_claude(text):
    """One short English imperative, or None when the API is not available."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        r = client.beta.messages.create(
            model=MODEL, max_tokens=2000,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",   # another model answers if this one declines
            system="You translate spoken drone commands into one short English imperative sentence "
                   "for a drone landing model. Answer with the sentence only.",
            messages=[{"role": "user", "content": text}])
        if r.stop_reason == "refusal":
            return None
        out = " ".join(b.text for b in r.content if b.type == "text").strip()
        return out or None
    except Exception:
        return None


def to_instruction(text):
    """(instruction for the model, how it was made). English passes through unchanged."""
    if not is_arabic(text):
        return text, "as said"
    hit = from_word_list(text)
    if hit:
        return hit, "arabic word list"
    translated = with_claude(text)
    if translated:
        return translated, "translated by Claude"
    return TRAINING_SENTENCE, "not understood, training sentence used"
