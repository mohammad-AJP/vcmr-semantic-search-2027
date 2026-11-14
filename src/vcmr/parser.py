"""Conservative, source-grounded dependency parser for retrieval experiments.

The parser is an engineering baseline, not a novelty claim. It does not observe
video or establish that an entity can be tracked. All spans are offsets into the
unaltered input, and confidence is deliberately not fabricated.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

PARSER_VERSION = "0.1.0"
MODEL_NAME = "en_core_web_sm"
PERSON_WORDS = {"person", "people", "man", "woman", "boy", "girl", "child", "kid", "someone", "somebody", "he", "she", "they", "adult", "gentleman", "lady"}
GARMENT_WORDS = {"suit", "shirt", "jacket", "coat", "dress", "skirt", "trouser", "pants", "jeans", "shorts", "sweater", "t-shirt", "uniform", "hat", "cap", "shoe", "boot", "sock", "hoodie", "vest", "blazer", "tie"}
STATE_VERBS = {"stand", "sit", "lie", "hold", "wear", "remain", "wait", "watch", "look"}
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
OBJECT_DEPS = {"dobj", "obj", "dative", "iobj", "attr", "oprd"}


@lru_cache(maxsize=1)
def _get_nlp():
    try:
        import spacy
        return spacy.load(MODEL_NAME, disable=["ner"])
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "The semantic parser requires spaCy and en_core_web_sm in the "
            "isolated experiment environment. No keyword-only fallback is used."
        ) from exc


def _span(text: str, start: int, end: int) -> dict[str, Any]:
    return {"text": text[start:end], "start": start, "end": end}


def _token_span(text: str, token) -> dict[str, Any]:
    return _span(text, token.idx, token.idx + len(token.text))


def _category(lemma: str) -> str:
    if lemma.lower() in PERSON_WORDS:
        return "person"
    if lemma.lower() in GARMENT_WORDS:
        return "garment"
    return "entity"


def parse_query(text: str) -> dict[str, Any]:
    """Return a JSON-serializable semantic program and <=6 retrieval phrases.

    Action arguments refer to entity IDs and intervals. An interval is a logical
    variable, not a predicted timestamp. Sequential clauses keep distinct
    intervals; an unmarked conjunction does not assert simultaneity. Phrase text
    is always an exact substring of the original query.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("query must be a nonempty string")
    nlp = _get_nlp()
    doc = nlp(text)
    limitations = [
        "Dependency parsing is fallible; predicates are hypotheses, not calibrated probabilities.",
        "Entity IDs express linguistic binding only; no visual tracks or visual identity evidence are supplied.",
    ]
    entities: list[dict[str, Any]] = []
    ent_by_head: dict[int, dict[str, Any]] = {}
    # noun_chunks keep determiners and modifiers while excluding event clauses.
    for chunk in doc.noun_chunks:
        root = chunk.root
        if root.pos_ not in {"NOUN", "PROPN", "PRON"}:
            continue
        if root.i in ent_by_head:
            continue
        entity = {
            "id": f"e{len(entities)}", "label": root.lemma_.lower(),
            "category": _category(root.lemma_),
            "source": _span(text, chunk.start_char, chunk.end_char),
            "head_source": _token_span(text, root),
            "is_pronoun": root.pos_ == "PRON",
        }
        entities.append(entity)
        ent_by_head[root.i] = entity
    for token in doc:
        if token.pos_ not in {"NOUN", "PROPN", "PRON"} or token.i in ent_by_head:
            continue
        # A compound noun belongs to its head's phrase rather than a new entity.
        if token.dep_ in {"compound", "poss"} and token.head.i in ent_by_head:
            continue
        if token.dep_ in {"expl"}:
            continue
        entity = {
            "id": f"e{len(entities)}", "label": token.lemma_.lower(),
            "category": _category(token.lemma_), "source": _token_span(text, token),
            "head_source": _token_span(text, token), "is_pronoun": token.pos_ == "PRON",
        }
        entities.append(entity)
        ent_by_head[token.i] = entity
    if any(e["is_pronoun"] for e in entities):
        limitations.append("Pronouns are preserved as separate entities; cross-clause coreference is not guessed.")

    def entity_for(token):
        if token is None:
            return None
        if token.i in ent_by_head:
            return ent_by_head[token.i]
        if token.dep_ in {"compound", "amod", "poss", "conj"} and token.head.i in ent_by_head:
            return ent_by_head[token.head.i]
        return None

    def subject_for(token, seen=None):
        seen = set() if seen is None else seen
        if token.i in seen:
            return None
        seen.add(token.i)
        subjects = [child for child in token.children if child.dep_ in SUBJECT_DEPS]
        if subjects:
            return subjects[0]
        if token.dep_ in {"acl", "relcl"} and token.head.pos_ in {"NOUN", "PROPN", "PRON"}:
            return token.head
        if token.dep_ == "pcomp" and token.head.lower_ in {"before", "after", "while", "until"}:
            return subject_for(token.head.head, seen)
        if token.dep_ in {"conj", "advcl", "xcomp"} and token.head != token:
            # Object-control constructions are not resolved by subject inheritance.
            if token.dep_ == "xcomp" and any(c.dep_ in {"dobj", "obj"} for c in token.head.children):
                return None
            return subject_for(token.head, seen)
        return None

    verb_tokens = [t for t in doc if t.pos_ == "VERB" and t.lemma_.lower() != "be"]
    # Root action is the target; subordinate before/after actions supply context.
    verb_tokens.sort(key=lambda t: (t.dep_ != "ROOT", t.i))
    temporal_markers = [t for t in doc if t.lower_ in {"before", "after", "then", "while", "during", "until", "once", "subsequently", "simultaneously"}]
    sequential = any(t.lower_ in {"before", "after", "then", "subsequently"} for t in temporal_markers)
    overlap = any(t.lower_ in {"while", "simultaneously"} for t in temporal_markers)
    if sequential:
        temporal_mode = "sequential"
    elif overlap or len(verb_tokens) <= 1:
        temporal_mode = "simultaneous"
    else:
        temporal_mode = "unknown"
        limitations.append("Multiple unmarked actions do not establish simultaneity or event order.")

    actions: list[dict[str, Any]] = []
    action_by_token: dict[int, dict[str, Any]] = {}
    intervals: list[dict[str, Any]] = []
    for index, token in enumerate(verb_tokens):
        interval = f"I{index}"
        subject = subject_for(token)
        subj_entity = entity_for(subject)
        passive = bool(subject and subject.dep_ in {"nsubjpass", "csubjpass"}) or any(c.dep_ == "auxpass" for c in token.children)
        roles: dict[str, str] = {}
        if subj_entity:
            roles["patient" if passive else "actor"] = subj_entity["id"]
        for child in token.children:
            entity = entity_for(child)
            if child.dep_ in OBJECT_DEPS and entity:
                roles["indirect_object" if child.dep_ in {"dative", "iobj"} else "patient"] = entity["id"]
            if child.dep_ == "agent":
                for obj in child.children:
                    agent_entity = entity_for(obj)
                    if agent_entity and obj.dep_ == "pobj":
                        roles["actor"] = agent_entity["id"]
        particles = [c for c in token.children if c.dep_ == "prt"]
        particle_text = " ".join(c.lower_ for c in particles)
        lemma = token.lemma_.lower()
        if lemma in {"stand", "sit", "lie"} and any(c.lower_ in {"up", "down"} for c in particles):
            aspect = "transition"
        elif lemma in STATE_VERBS:
            aspect = "state"
        else:
            aspect = "event"
        negations = [c for c in token.children if c.dep_ == "neg"]
        role = "target" if index == 0 else "unspecified"
        if any(c.dep_ == "mark" and c.lower_ in {"before", "after", "until", "once", "while"} for c in token.children):
            role = "context"
        if token.dep_ == "pcomp" and token.head.lower_ in {"before", "after", "until", "once", "while"}:
            role = "context"
        intervals.append({"id": interval, "role": role, "seconds": None})
        action = {
            "id": f"a{index}", "name": lemma + ("_" + particle_text.replace(" ", "_") if particle_text else ""),
            "lemma": lemma, "args": list(dict.fromkeys(roles.values())) + [interval],
            "roles": roles, "interval": interval, "aspect": aspect,
            "polarity": "negative" if negations else "positive", "source": _token_span(text, token),
            "particles": [_token_span(text, c) for c in particles],
            "negation_sources": [_token_span(text, c) for c in negations],
        }
        if not roles:
            limitations.append(f"The participant of '{token.text}' is unresolved.")
        actions.append(action)
        action_by_token[token.i] = action
    if not intervals:
        intervals = [{"id": "I0", "role": "target", "seconds": None}]
    target_interval = next((i["id"] for i in intervals if i["role"] == "target"), intervals[0]["id"])

    def interval_for(token):
        current = token
        visited = set()
        while current.i not in visited:
            visited.add(current.i)
            if current.i in action_by_token:
                return action_by_token[current.i]["interval"]
            current = current.head
        return target_interval

    attributes: list[dict[str, Any]] = []
    for token in doc:
        owner = None
        if token.dep_ in {"amod", "nummod"}:
            owner = entity_for(token.head)
        elif token.pos_ == "ADJ" and token.dep_ in {"acomp", "attr", "ROOT"}:
            owner = entity_for(subject_for(token.head if token.dep_ != "ROOT" else token))
        elif token.pos_ == "ADJ" and token.dep_ == "conj" and token.head.dep_ == "amod":
            owner = entity_for(token.head.head)
        if owner is None:
            continue
        negations = [c for c in token.children if c.dep_ == "neg"]
        attributes.append({"id": f"p{len(attributes)}", "name": token.lemma_.lower(), "args": [owner["id"]],
                           "owner": owner["id"], "source": _token_span(text, token),
                           "polarity": "negative" if negations else "positive",
                           "negation_sources": [_token_span(text, c) for c in negations]})

    relations: list[dict[str, Any]] = []
    for prep in doc:
        if prep.dep_ not in {"prep", "agent"} or prep.lower_ in {"before", "after", "during", "until"}:
            continue
        for obj in prep.children:
            obj_entity = entity_for(obj)
            if obj.dep_ not in {"pobj", "obj"} or obj_entity is None:
                continue
            owner = entity_for(prep.head)
            if owner is None:
                owner = entity_for(subject_for(prep.head))
            if owner is None:
                continue
            interpretation = "explicit_preposition"
            name = "with" if prep.lower_ == "without" else prep.lower_
            if name == "in" and owner["category"] == "person" and obj_entity["category"] == "garment":
                name = "wear"
                interpretation = "person_in_garment_rule"
            interval = interval_for(prep.head)
            end = max(obj_entity["source"]["end"], prep.idx + len(prep.text))
            relations.append({"id": f"r{len(relations)}", "name": name,
                              "args": [owner["id"], obj_entity["id"], interval],
                              "source": _span(text, prep.idx, end),
                              "polarity": "negative" if prep.lower_ == "without" else "positive",
                              "interpretation": interpretation})
    if any(r["interpretation"] == "person_in_garment_rule" for r in relations):
        limitations.append("'Person in garment' is interpreted as wearing by an explicit lexical rule.")

    temporal_relations: list[dict[str, Any]] = []
    for marker in temporal_markers:
        if marker.lower_ not in {"before", "after", "while", "then", "subsequently", "simultaneously"}:
            limitations.append(f"Temporal marker '{marker.text}' is preserved but its boundary semantics are unresolved.")
            continue
        if marker.dep_ == "prep":
            complements = [c for c in marker.children if c.i in action_by_token]
            child_action = action_by_token.get(complements[0].i) if complements else None
            parent_action = action_by_token.get(marker.head.i)
        else:
            child_action = action_by_token.get(marker.head.i)
            parent_action = action_by_token.get(marker.head.head.i)
        if child_action is None:
            continue
        if parent_action is None or parent_action == child_action:
            if marker.lower_ in {"then", "subsequently", "simultaneously"}:
                earlier = [t for t in verb_tokens if t.i < marker.head.i]
                parent_action = action_by_token[max(earlier, key=lambda t: t.i).i] if earlier else None
        if parent_action is None or parent_action == child_action:
            continue
        first, second = parent_action["interval"], child_action["interval"]
        if marker.lower_ == "after":
            first, second = second, first
        relation = "overlap" if marker.lower_ in {"while", "simultaneously"} else "before"
        temporal_relations.append({"name": relation, "args": [first, second], "source": _token_span(text, marker)})
    if sequential and not temporal_relations:
        limitations.append("Sequential wording is detected, but event-to-event ordering was not resolved.")

    negation_tokens = [t for t in doc if t.dep_ == "neg" or t.lower_ in {"without", "neither", "nor", "no"}]
    if negation_tokens:
        limitations.append("Negation scope is local and approximate; absence is not evidence of a visual negative.")
    if any(t.lower_ == "or" for t in doc):
        limitations.append("Disjunction is preserved in the whole query but is not expanded into alternative programs.")

    phrases: list[dict[str, Any]] = []
    phrase_keys = set()

    def add_phrase(start, end, kind, weight):
        if len(phrases) >= 6 or end <= start:
            return
        fragment = text[start:end]
        key = fragment.strip().casefold()
        if key in phrase_keys or not key:
            return
        phrase_keys.add(key)
        phrases.append({"text": fragment, "kind": kind, "source_start": start,
                        "source_end": end, "weight": float(weight)})

    add_phrase(0, len(text), "whole_query", 1.0)
    # Contiguous subject/action/object envelopes preserve original grounding.
    # If an inherited subject crosses another event, use the local clause only.
    for token in sorted(verb_tokens, key=lambda t: t.i):
        local = [token] + [c for c in token.children if c.dep_ in {"aux", "auxpass", "neg", "prt"}]
        subject = subject_for(token)
        subject_entity = entity_for(subject)
        if subject_entity:
            between = [v for v in verb_tokens if min(subject.i, token.i) < v.i < max(subject.i, token.i)]
            if not between:
                subject_start = subject_entity["source"]["start"]
                subject_end = subject_entity["source"]["end"]
            else:
                subject_start, subject_end = token.idx, token.idx + len(token.text)
        else:
            subject_start, subject_end = token.idx, token.idx + len(token.text)
        bounds = [(t.idx, t.idx + len(t.text)) for t in local] + [(subject_start, subject_end)]
        for child in token.children:
            entity = entity_for(child)
            if child.dep_ in OBJECT_DEPS and entity:
                bounds.append((entity["source"]["start"], entity["source"]["end"]))
        start, end = min(b[0] for b in bounds), max(b[1] for b in bounds)
        add_phrase(start, end, "action", 0.8)
    for entity in entities:
        if any(a["owner"] == entity["id"] for a in attributes):
            add_phrase(entity["source"]["start"], entity["source"]["end"], "attributed_entity", 0.6)
    for relation in relations:
        add_phrase(relation["source"]["start"], relation["source"]["end"], "relation", 0.5)
    for entity in entities:
        add_phrase(entity["source"]["start"], entity["source"]["end"], "entity", 0.35)

    entity_predicates = [{"name": e["label"], "args": [e["id"]], "source": e["head_source"], "polarity": "positive"} for e in entities]
    return {
        "parser_version": PARSER_VERSION, "parser_model": MODEL_NAME,
        "parser_model_version": nlp.meta.get("version", "unknown"),
        "text": text, "query": text, "target_interval": target_interval,
        "entities": entities, "actions": actions, "attributes": attributes,
        "relations": relations, "intervals": intervals,
        "temporal_mode": temporal_mode, "temporal_relations": temporal_relations,
        "temporal_markers": [_token_span(text, t) for t in temporal_markers],
        "polarity": "contains_negation" if negation_tokens else "positive",
        "negation_sources": [_token_span(text, t) for t in negation_tokens],
        "predicates": entity_predicates + attributes + actions + relations,
        "retrieval_phrases": phrases, "limitations": list(dict.fromkeys(limitations)),
        "visual_evidence_status": "not_observed", "confidence": None,
    }
