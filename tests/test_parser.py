"""Semantic regression tests, including cases where naive token lists fail."""
import json
import unittest
from vcmr.parser import PARSER_VERSION, parse_query


class ParserTests(unittest.TestCase):
    def test_owner_and_shared_actor(self):
        p = parse_query("a man standing in a black suit")
        entities = {e["label"]: e["id"] for e in p["entities"]}
        black = next(a for a in p["attributes"] if a["name"] == "black")
        self.assertEqual(black["owner"], entities["suit"])
        standing = next(a for a in p["actions"] if a["lemma"] == "stand")
        self.assertEqual(standing["roles"]["actor"], entities["man"])
        self.assertEqual(standing["aspect"], "state")
        wear = next(r for r in p["relations"] if r["name"] == "wear")
        self.assertEqual(wear["args"][:2], [entities["man"], entities["suit"]])
        self.assertEqual(wear["args"][-1], standing["interval"])
        self.assertIsNone(p["confidence"])
        self.assertEqual(p["visual_evidence_status"], "not_observed")

    def test_roles_change_meaning(self):
        first = parse_query("A man pushes a woman.")
        second = parse_query("A woman pushes a man.")
        def actor(p):
            action = next(a for a in p["actions"] if a["lemma"] == "push")
            by_id = {e["id"]: e["label"] for e in p["entities"]}
            return by_id[action["roles"]["actor"]], by_id[action["roles"]["patient"]]
        self.assertEqual(actor(first), ("man", "woman"))
        self.assertEqual(actor(second), ("woman", "man"))

    def test_temporal_order_and_transition(self):
        p = parse_query("A man stands up after closing the door.")
        actions = {a["lemma"]: a for a in p["actions"]}
        self.assertEqual(actions["stand"]["aspect"], "transition")
        self.assertEqual(p["temporal_mode"], "sequential")
        self.assertNotEqual(actions["stand"]["interval"], actions["close"]["interval"])
        self.assertTrue(any(r["name"] == "before" and r["args"] == [actions["close"]["interval"], actions["stand"]["interval"]] for r in p["temporal_relations"]))
        self.assertEqual(p["target_interval"], actions["stand"]["interval"])
        context = next(i for i in p["intervals"] if i["id"] == actions["close"]["interval"])
        self.assertEqual(context["role"], "context")

    def test_before_and_overlap(self):
        p = parse_query("A man closes the door before sitting down.")
        actions = {a["lemma"]: a for a in p["actions"]}
        self.assertTrue(any(r["args"] == [actions["close"]["interval"], actions["sit"]["interval"]] for r in p["temporal_relations"]))
        self.assertEqual(actions["close"]["roles"]["actor"], actions["sit"]["roles"]["actor"])
        p = parse_query("A man stands while holding a cup.")
        self.assertEqual(p["temporal_mode"], "simultaneous")
        self.assertTrue(any(r["name"] == "overlap" for r in p["temporal_relations"]))

    def test_negation_scope(self):
        p = parse_query("The man is not holding a cup.")
        hold = next(a for a in p["actions"] if a["lemma"] == "hold")
        self.assertEqual(hold["polarity"], "negative")
        self.assertEqual(p["polarity"], "contains_negation")
        self.assertEqual(hold["negation_sources"][0]["text"], "not")
        self.assertTrue(any("absence" in limitation for limitation in p["limitations"]))

    def test_unmarked_conjunction_is_not_simultaneous(self):
        p = parse_query("A man opens a door and sits down.")
        self.assertEqual(p["temporal_mode"], "unknown")
        self.assertFalse(p["temporal_relations"])

    def test_passive_preserves_roles(self):
        p = parse_query("A woman is pushed by a man.")
        push = next(a for a in p["actions"] if a["lemma"] == "push")
        by_id = {e["id"]: e["label"] for e in p["entities"]}
        self.assertEqual(by_id[push["roles"]["actor"]], "man")
        self.assertEqual(by_id[push["roles"]["patient"]], "woman")

    def test_spans_and_whole_query_are_preserved(self):
        examples = ["a man standing in a black suit", "  A woman puts a red cup down, then leaves.  ", "A person walks without a bag.", "A man watches a woman run."]
        for query in examples:
            p = parse_query(query)
            self.assertEqual(p["query"], query)
            self.assertEqual(p["parser_version"], PARSER_VERSION)
            self.assertEqual(p["retrieval_phrases"][0]["text"], query)
            self.assertLessEqual(len(p["retrieval_phrases"]), 6)
            for field in ["entities", "actions", "attributes", "relations", "temporal_relations"]:
                for item in p[field]:
                    source = item["source"]
                    self.assertEqual(source["text"], query[source["start"]:source["end"]])
            for phrase in p["retrieval_phrases"]:
                self.assertEqual(phrase["text"], query[phrase["source_start"]:phrase["source_end"]])
            json.dumps(p)

    def test_invalid_input(self):
        for query in ["", "   ", None, 12]:
            with self.assertRaises(ValueError):
                parse_query(query)


if __name__ == "__main__":
    unittest.main()
