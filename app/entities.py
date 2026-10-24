"""Entity and relation extraction.

Tries spaCy's small English model (``en_core_web_sm``) for proper named
entity recognition when it is installed and its model data is available.
If spaCy or the model isn't present (common in minimal/offline CI/demo
environments) we fall back to a deterministic regex heuristic:

* Entities: capitalized word sequences ("Jane Doe", "Acme Corp") that are
  not the first word of a sentence-only stopword.
* Relations: a short list of lexical patterns such as
  "<Entity> works at <Entity>", "<Entity> founded <Entity>",
  "<Entity> is a <Entity>", "<Entity> acquired <Entity>".

Neither approach is a substitute for a production-grade IE pipeline; this
is intentionally simple so the demo has zero mandatory downloads.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_CAPWORD = r"[A-Z][a-zA-Z0-9&.'-]*"
_ENTITY_RE = re.compile(rf"(?:{_CAPWORD}(?:\s+{_CAPWORD}){{0,3}})")

_RELATION_PATTERNS = [
    (re.compile(rf"({_ENTITY_RE.pattern})\s+works? at\s+({_ENTITY_RE.pattern})"), "WORKS_AT"),
    (re.compile(rf"({_ENTITY_RE.pattern})\s+founded\s+({_ENTITY_RE.pattern})"), "FOUNDED"),
    (re.compile(rf"({_ENTITY_RE.pattern})\s+acquired\s+({_ENTITY_RE.pattern})"), "ACQUIRED"),
    (re.compile(rf"({_ENTITY_RE.pattern})\s+is a\s+({_ENTITY_RE.pattern})"), "IS_A"),
    (re.compile(rf"({_ENTITY_RE.pattern})\s+is the\s+({_ENTITY_RE.pattern})"), "IS_A"),
    (re.compile(rf"({_ENTITY_RE.pattern})\s+reports? to\s+({_ENTITY_RE.pattern})"), "REPORTS_TO"),
    (re.compile(rf"({_ENTITY_RE.pattern})\s+partners? with\s+({_ENTITY_RE.pattern})"), "PARTNERS_WITH"),
]

_STOPWORDS = {
    "The",
    "This",
    "That",
    "These",
    "Those",
    "It",
    "They",
    "We",
    "I",
    "A",
    "An",
}


@dataclass
class RelationTriple:
    source: str
    relation: str
    target: str


@dataclass
class ExtractionResult:
    entities: list[str] = field(default_factory=list)
    relations: list[RelationTriple] = field(default_factory=list)


def _regex_entities(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in _ENTITY_RE.finditer(text):
        candidate = match.group(0).strip().rstrip(".,;:!?")
        words = candidate.split()
        if not words:
            continue
        if words[0] in _STOPWORDS and len(words) == 1:
            continue
        if len(candidate) < 3:
            continue
        seen.setdefault(candidate, None)
    return list(seen.keys())


def _regex_relations(text: str) -> list[RelationTriple]:
    relations: list[RelationTriple] = []
    for pattern, relation_name in _RELATION_PATTERNS:
        for match in pattern.finditer(text):
            source = match.group(1).strip().rstrip(".,;:!?")
            target = match.group(2).strip().rstrip(".,;:!?")
            if source and target and source != target:
                relations.append(RelationTriple(source=source, relation=relation_name, target=target))
    return relations


_SPACY_NLP = None
_SPACY_LOAD_ATTEMPTED = False


def _try_load_spacy():
    global _SPACY_NLP, _SPACY_LOAD_ATTEMPTED
    if _SPACY_LOAD_ATTEMPTED:
        return _SPACY_NLP
    _SPACY_LOAD_ATTEMPTED = True
    try:
        import spacy  # type: ignore

        _SPACY_NLP = spacy.load("en_core_web_sm")
    except Exception:
        _SPACY_NLP = None
    return _SPACY_NLP


def extract_entities_and_relations(text: str, use_spacy: bool = True) -> ExtractionResult:
    """Extract entities (and, best-effort, relation triples) from ``text``.

    Relation extraction always runs the lexical-pattern pass regardless of
    whether spaCy is available, since spaCy's small model does not do
    relation extraction out of the box; spaCy (when available) only
    upgrades entity recognition quality.
    """
    nlp = _try_load_spacy() if use_spacy else None
    if nlp is not None:
        doc = nlp(text)
        entities = list(dict.fromkeys(ent.text.strip() for ent in doc.ents if ent.text.strip()))
        if not entities:
            entities = _regex_entities(text)
    else:
        entities = _regex_entities(text)

    relations = _regex_relations(text)
    return ExtractionResult(entities=entities, relations=relations)
