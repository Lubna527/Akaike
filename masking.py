"""PII masking module (no LLMs)"""
from __future__ import annotations

import re
from typing import List, Dict, Tuple

import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])

# --- regex patterns ---
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(\+?\d{1,3}[- ]?)?\d{10}\b")
DOB_RE = re.compile(r"\b(\d{2}[/-]\d{2}[/-]\d{2,4})\b")
AADHAR_RE = re.compile(r"\b\d{4} ?\d{4} ?\d{4}\b")
CARD_RE = re.compile(r"\b\d{13,19}\b")
CVV_RE = re.compile(r"(?i)(?<=cvv\s?)\d{3,4}\b")
EXP_RE = re.compile(r"\b(0[1-9]|1[0-2])/?\d{2,4}\b")

_regex_map = {
    "email": EMAIL_RE,
    "phone_number": PHONE_RE,
    "dob": DOB_RE,
    "aadhar_num": AADHAR_RE,
    "credit_debit_no": CARD_RE,
    "cvv_no": CVV_RE,
    "expiry_no": EXP_RE,
}

_spacy_labels = {"PERSON": "full_name"}


class Span:
    def __init__(self, start: int, end: int, label: str):
        self.start, self.end, self.label = start, end, label

    def __lt__(self, other):
        return self.start < other.start


def _regex_spans(text: str) -> List[Span]:
    spans = []
    for label, rgx in _regex_map.items():
        for m in rgx.finditer(text):
            spans.append(Span(m.start(), m.end(), label))
    return spans


def _spacy_spans(text: str) -> List[Span]:
    doc = nlp(text)
    spans = []
    for ent in doc.ents:
        if ent.label_ in _spacy_labels:
            spans.append(Span(ent.start_char, ent.end_char, _spacy_labels[ent.label_]))
    return spans


def mask(text: str) -> Tuple[str, List[Dict]]:
    spans = _regex_spans(text) + _spacy_spans(text)
    # resolve overlaps – keep earliest, longest
    spans.sort()
    cleaned = []
    current_end = -1
    dedup = []
    for s in spans:
        if s.start >= current_end:
            dedup.append(s)
            current_end = s.end
    spans = dedup

    parts = []
    cursor = 0
    entities = []
    for s in spans:
        parts.append(text[cursor:s.start])
        tag = f"[{s.label}]"
        parts.append(tag)
        entities.append({
            "position": [s.start, s.end],
            "classification": s.label,
            "entity": text[s.start:s.end],
        })
        cursor = s.end
    parts.append(text[cursor:])
    return "".join(parts), entities
