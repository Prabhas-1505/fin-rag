import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

REFUSAL_MESSAGE = (
    "I can't complete that request - it looks like an attempt to override my "
    "instructions rather than a question I can answer from the ingested documents."
)

UNTRUSTED_OPEN = "<<RETRIEVED_DOCUMENT_DATA>>"
UNTRUSTED_CLOSE = "<<END_RETRIEVED_DOCUMENT_DATA>>"

# Four attack categories. Patterns are intentionally broad (regex over
# lowercased text) - this is a heuristic screen, not a classifier, and is
# meant to catch the obvious cases cheaply before anything reaches an LLM
# call or gets returned to the user.
_INJECTION_PATTERNS: List[Tuple[str, str]] = [
    ("instruction_override", r"ignor\w*\b.{0,20}\binstructions\b"),
    ("instruction_override", r"disregard\w* (all |any |the )?(previous|prior|above)"),
    ("instruction_override", r"\bnew instructions\b"),
    ("instruction_override", r"forget (everything|all)(\s+you (know|were told))?\s*(above|before)"),
    ("instruction_override", r"\byou are now\b"),
    ("instruction_override", r"\[\s*system\s*\]"),
    ("instruction_override", r"<\|?\s*(im_start|system)\s*\|?>"),
    ("instruction_override", r"^\s*system\s*:"),

    ("jailbreak_roleplay", r"\bact as\b.{0,40}\b(no (restrictions|rules|filters)|unrestricted)\b"),
    ("jailbreak_roleplay", r"\bpretend (you are|to be)\b"),
    ("jailbreak_roleplay", r"\bdeveloper mode\b"),
    ("jailbreak_roleplay", r"\bdan\b"),
    ("jailbreak_roleplay", r"\bjailbreak(ed|ing)?\b"),
    ("jailbreak_roleplay", r"\bno (longer )?bound by (your |the )?(rules|restrictions|guidelines)\b"),

    ("prompt_exfiltration", r"(repeat|print|reveal|show|output|leak)\s+(your |the )?(system prompt|instructions|initial prompt)"),
    ("prompt_exfiltration", r"what (are|were) your( \w+){0,2} (instructions|rules|system prompt)"),
    ("prompt_exfiltration", r"\brepeat (the|everything) (above|before this)\b"),
    ("prompt_exfiltration", r"\beverything\b.{0,20}\b(above|before this)\b"),
    ("prompt_exfiltration", r"\bverbatim\b.{0,30}\b(prompt|instructions)\b"),
    ("prompt_exfiltration", r"(print|output|show)\s+(it|them|that)\s+(verbatim|exactly|word for word)"),

    # "base64" mentioned near an actual-looking encoded blob (not a bare
    # mention - "what is base64 encoding?" shouldn't trip this).
    ("encoded_payload", r"base64(?=(?:.|\n){0,80}(?:[A-Za-z0-9+/]{20,}))"),
    ("encoded_payload", r"(?:[A-Za-z0-9+/]{4}){6,}={0,2}"),  # base64-looking blob, >=24 chars, standalone
    ("encoded_payload", r"\\x[0-9a-fA-F]{2}(\\x[0-9a-fA-F]{2}){3,}"),
]


@dataclass
class ScreeningResult:
    blocked: bool
    category: Optional[str] = None
    matched_pattern: Optional[str] = None


def screen_text(text: str) -> ScreeningResult:
    lowered = text.lower()
    for category, pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return ScreeningResult(blocked=True, category=category, matched_pattern=pattern)
    return ScreeningResult(blocked=False)


def screen_user_input(query: str) -> ScreeningResult:
    """Layer 1: screen the raw user query before any LLM call."""
    return screen_text(query)


def quarantine_chunks(chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Layer 2 (anti-poisoning): screens retrieved document chunks for injected
    instructions before they reach a prompt. Returns (clean, flagged).
    """
    clean, flagged = [], []
    for chunk in chunks:
        text = chunk.get("payload", {}).get("text", "")
        result = screen_text(text)
        if result.blocked:
            flagged.append({**chunk, "_screening": result})
        else:
            clean.append(chunk)
    return clean, flagged


def wrap_untrusted(text: str) -> str:
    """Delimits retrieved content as data, never instructions, for the prompt."""
    return f"{UNTRUSTED_OPEN}\n{text}\n{UNTRUSTED_CLOSE}"


def validate_output(answer: str, system_prompt_fragments: Optional[List[str]] = None) -> ScreeningResult:
    """
    Layer 3: screens the model's final answer for signs of a successful
    hijack - either echoing injection-style directives in its own voice, or
    leaking a recognizable fragment of our system prompt.
    """
    result = screen_text(answer)
    if result.blocked:
        return result

    for fragment in system_prompt_fragments or []:
        if len(fragment) > 20 and fragment.lower() in answer.lower():
            return ScreeningResult(blocked=True, category="prompt_exfiltration", matched_pattern="system_prompt_leak")

    return ScreeningResult(blocked=False)
