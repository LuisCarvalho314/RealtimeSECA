"""Hand-derived examples; expected results come from the rules in each docstring."""

import copy
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cluster import (contextual_proxies, direct, evaluate, indirect, merge_proxies,
                     proximity)


def tree(hkts, nodes):
    """Build verbose-export-shaped fixtures from explicit hand-written relations."""
    vocabulary = sorted({word for _, _, words, _ in nodes for word in words})
    word_ids = {word: i + 1 for i, word in enumerate(vocabulary)}
    population = sorted({sid for _, _, _, sources in nodes for sid in sources})
    source_ids = {sid: i + 1 for i, sid in enumerate(population)}
    return {
        "hkts": [{"hkt_id": hid, "parent_node_id": parent,
                  "node_ids": [nid for nid, owner, _, _ in nodes if owner == hid]}
                 for hid, parent in hkts],
        "nodes": [{"node_id": nid, "hkt_id": owner,
                   "is_refuge_node": not words,
                   "words": [{"word_id": word_ids[word], "token": word}
                             for word in words] if words else
                            [{"word_id": -1, "token": "<REFUGE>"}],
                   "sources": [{"internal_source_id": source_ids[sid],
                                "external_source_id": sid} for sid in sources],
                   "top_words": []}
                  for nid, owner, words, sources in nodes],
        "word_legend": [{"word_id": wid, "token": word}
                        for word, wid in word_ids.items()],
        "source_legend": [{"internal_source_id": internal, "external_source_id": sid}
                          for sid, internal in source_ids.items()],
    }


def nested():
    # 1 is a Seed event, 5 is an Orphan through 2->4 (only refuges).
    # 6 follows a refuge under Seed 1, so is a subtopic, never an orphan event.
    return tree([(1, 0), (2, 2), (3, 4), (4, 1), (5, 7)], [
        (1, 1, ["a"], ["d1", "d2", "d3", "d4"]),
        (2, 1, [], ["d5", "d6"]),
        (4, 2, [], ["d5", "d6"]),
        (5, 3, ["b"], ["d5"]),
        (8, 3, [], ["d6"]),
        (3, 4, ["x"], ["d1", "d2"]),
        (7, 4, [], ["d3", "d4"]),
        (6, 5, ["y"], ["d3"]),
    ])


class DirectTests(unittest.TestCase):
    def test_one_event(self):
        """A Seed node's sources are its event members, irrespective of labels."""
        fixture = tree([(1, 0)], [(9, 1, ["quake"], ["d2", "d1"])])
        self.assertEqual(direct(fixture, 0, ["d2", "d1"]), {
            "batch_index": 0, "evaluation_source_ids": ["d1", "d2"],
            "clusters": [{"cluster_id": "event-9", "source_ids": ["d1", "d2"]}],
            "unassigned_source_ids": []})

    def test_independent_and_shared_sources(self):
        """d2 belongs to both Seed events; direct has no hard-assignment rule."""
        fixture = tree([(1, 0)], [(1, 1, ["a"], ["d1", "d2"]),
                                      (2, 1, ["b"], ["d2", "d3"])])
        self.assertEqual(direct(fixture, 4, ["d1", "d2", "d3"])["clusters"], [
            {"cluster_id": "event-1", "source_ids": ["d1", "d2"]},
            {"cluster_id": "event-2", "source_ids": ["d2", "d3"]}])

    def test_orphan_ancestry_and_nested_refuge(self):
        """Only 1 and 5 are event roots; 3 and 6 retain Seed 1 as ancestor."""
        result = direct(nested(), 0, [f"d{i}" for i in range(1, 8)])
        self.assertEqual(result["clusters"], [
            {"cluster_id": "event-1", "source_ids": ["d1", "d2", "d3", "d4"]},
            {"cluster_id": "event-5", "source_ids": ["d5"]}])
        self.assertEqual(result["unassigned_source_ids"], ["d6", "d7"])

    def test_current_population_does_not_expand_to_retained(self):
        """The caller selects d2, even though the model retains d1 through d6."""
        result = direct(nested(), 100, ["d2"])
        self.assertEqual(result["evaluation_source_ids"], ["d2"])
        self.assertEqual(result["clusters"], [
            {"cluster_id": "event-1", "source_ids": ["d2"]}])

    def test_refuge_only_and_empty_population(self):
        """A refuge is no event; an empty requested population yields empty sets."""
        fixture = tree([(1, 0)], [(1, 1, [], ["d1"])])
        self.assertEqual(direct(fixture, 0, ["d1"])["unassigned_source_ids"], ["d1"])
        result = direct(nested(), 0, [])
        self.assertEqual(result["clusters"], [])
        self.assertEqual(result["evaluation_source_ids"], [])

    def test_duplicate_membership_is_set_membership(self):
        """Repeated d1 inside one node contributes one event member."""
        fixture = tree([(1, 0)], [(1, 1, ["a"], ["d1", "d1"])])
        self.assertEqual(direct(fixture, 0, ["d1"])["clusters"][0]["source_ids"], ["d1"])


class IndirectTests(unittest.TestCase):
    def test_context_is_node_support_not_word_frequency_or_evaluation_support(self):
        """Seed a=4/4, child x=2/4, internal-refuge child y=1/4; Orphan b=1/1."""
        self.assertEqual(contextual_proxies(nested()), {
            (1,): {"a": F(1), "x": F(1, 2), "y": F(1, 4)}, (5,): {"b": F(1)}})
        result = indirect(nested(), 0, ["new"], {"new": ["x", "x"]}, "3/4")
        self.assertEqual(result["clusters"], [
            {"cluster_id": "event-1", "source_ids": ["new"]}])

    def test_repeated_descriptor_uses_strongest_context(self):
        """Two x occurrences, 2/4 and 1/4, reduce to 1/2 rather than 3/4."""
        fixture = tree([(1, 0), (2, 1)], [
            (1, 1, ["a"], ["d1", "d2", "d3", "d4"]),
            (2, 2, ["x"], ["d1", "d2"]), (3, 2, ["x"], ["d3"])])
        self.assertEqual(contextual_proxies(fixture), {(1,): {"a": F(1), "x": F(1, 2)}})

    def test_proximity_is_distinct_intersection_sum(self):
        """Repeated x and unknown words do not increase 1/2 + 1/4 = 3/4."""
        self.assertEqual(proximity({"x": F(1, 2), "y": F(1, 4)},
                                   ["x", "x", "y", "unknown"]), F(3, 4))

    def test_threshold_boundary_and_transitive_merge(self):
        """A--x--B--y--C is one component at 1/2, three components above 1/2."""
        proxies = {(1,): {"a": F(1), "x": F(1, 2)},
                   (2,): {"b": F(1), "x": F(1, 2), "y": F(1, 2)},
                   (3,): {"c": F(1), "y": F(1, 2)}}
        self.assertEqual(merge_proxies(proxies, "1/2"), {(1, 2, 3): {
            "a": F(1), "b": F(1), "c": F(1), "x": F(1, 2), "y": F(1, 2)}})
        self.assertEqual(merge_proxies(proxies, "3/4"), proxies)

    def test_merged_weight_is_maximum(self):
        """Shared root word q merges; x keeps max(1/2,3/4), without renormalizing."""
        self.assertEqual(merge_proxies({(1,): {"q": F(1), "x": F(1, 2)},
                                       (2,): {"q": F(1), "x": F(3, 4)}}, 1),
                         {(1, 2): {"q": F(1), "x": F(3, 4)}})

    def test_hard_assignment_sum_tie_and_zero(self):
        """a+b ties 1:1, choosing node 2; b alone chooses 10; z remains unassigned."""
        fixture = tree([(1, 0)], [(10, 1, ["b"], ["d1"]), (2, 1, ["a"], ["d2"])])
        result = indirect(fixture, 0, ["tie", "b", "zero", "empty"],
                          {"tie": ["a", "b"], "b": ["b"], "zero": ["z"], "empty": []}, 1)
        self.assertEqual(result["clusters"], [
            {"cluster_id": "event-10", "source_ids": ["b"]},
            {"cluster_id": "event-2", "source_ids": ["tie"]}])
        self.assertEqual(result["unassigned_source_ids"], ["empty", "zero"])

    def test_complete_merge_and_assignment(self):
        """Both roots describe q, so one merged event receives sources matching a or b."""
        fixture = tree([(1, 0)], [(1, 1, ["q", "a"], ["d1"]),
                                      (2, 1, ["q", "b"], ["d2"])])
        result = indirect(fixture, 1, ["d1", "d2"], {"d1": ["a"], "d2": ["b"]}, 1)
        self.assertEqual(result["clusters"], [
            {"cluster_id": "event-1-2", "source_ids": ["d1", "d2"]}])

    def test_empty_support_and_no_proxies(self):
        """Forgotten empty roots are omitted; no proxy leaves a source unassigned."""
        fixture = tree([(1, 0)], [(1, 1, ["a"], [])])
        self.assertEqual(contextual_proxies(fixture), {})
        self.assertEqual(indirect(fixture, 0, ["d"], {"d": ["a"]}, 1)
                         ["unassigned_source_ids"], ["d"])

    def test_exported_top_words_do_not_create_false_merges(self):
        """Shared presentation top-word q is not a descriptor in either proxy."""
        fixture = tree([(1, 0)], [(1, 1, ["a"], ["d1"]), (2, 1, ["b"], ["d2"])])
        for node in fixture["nodes"]:
            node["top_words"] = [{"word_id": 99, "token": "q"}]
        result = indirect(fixture, 0, ["d1", "d2"], {"d1": ["a"], "d2": ["b"]}, 1)
        self.assertEqual(len(result["clusters"]), 2)


class ContractTests(unittest.TestCase):
    def test_permutations_repeats_labels_and_input_immutability(self):
        """Array order and labels cannot alter sets or exact scores; no input mutates."""
        original = nested()
        permuted = copy.deepcopy(original)
        for key in ("nodes", "hkts", "source_legend", "word_legend"):
            permuted[key].reverse()
        for node in permuted["nodes"]:
            node["sources"].reverse()
            node["words"].reverse()
        for hkt in permuted["hkts"]:
            hkt["node_ids"].reverse()
        for method in ("direct", "indirect"):
            request = {"method": method, "tree": original, "batch_index": 2,
                       "evaluation_source_ids": ["d5", "d2", "zero"],
                       "source_tokens": {"d5": ["b"], "d2": ["x", "a"], "zero": []},
                       "contextual_threshold": "1/2"}
            before = copy.deepcopy(request)
            expected = json.dumps(evaluate(request), sort_keys=True)
            for _ in range(3):
                self.assertEqual(json.dumps(evaluate(request), sort_keys=True), expected)
            self.assertEqual(request, before)
            request.update(tree=permuted, evaluation_source_ids=["zero", "d2", "d5"],
                           labels={"d2": "arbitrary", "d5": "arbitrary"})
            self.assertEqual(json.dumps(evaluate(request), sort_keys=True), expected)

    def test_reject_invalid_population_and_threshold(self):
        """Ambiguous duplicate IDs and missing/nonnumeric thresholds cannot score."""
        for index, ids in [(True, []), (-1, []), (0, ["d", "d"]), (0, [""])]:
            with self.subTest(index=index, ids=ids), self.assertRaises(ValueError):
                direct(nested(), index, ids)
        for cutoff in (None, True, "nan", "inf", "1/0", 0, -1, "1.01"):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                indirect(nested(), 0, [], {}, cutoff)

    def test_missing_tokens_are_not_an_empty_document(self):
        """Missing evidence must fail; an explicitly empty token set is valid."""
        with self.assertRaises(ValueError):
            indirect(nested(), 0, ["new"], {}, 1)

    def test_invalid_tree_fails_without_mutation(self):
        """Reject missing parents, cycles, invalid identities and scope expansion."""
        invalid = []
        fixture = nested()
        fixture["hkts"][1]["parent_node_id"] = 999
        invalid.append(fixture)
        fixture = nested()
        fixture["hkts"][1]["parent_node_id"] = 5
        invalid.append(fixture)  # HKT 2->3->2 is disconnected from the Seed-HKT.
        fixture = nested()
        fixture["nodes"][0]["sources"][0]["external_source_id"] = "wrong"
        invalid.append(fixture)
        fixture = nested()
        fixture["nodes"][5]["sources"].append(fixture["nodes"][4]["sources"][0])
        invalid.append(fixture)  # d6 is not in Seed node 1.
        fixture = nested()
        fixture["nodes"].append(copy.deepcopy(fixture["nodes"][0]))
        invalid.append(fixture)
        for fixture in invalid:
            before = copy.deepcopy(fixture)
            with self.subTest(fixture=fixture), self.assertRaises(ValueError):
                direct(fixture, 0, [])
            self.assertEqual(fixture, before)

    def test_cli_json_and_atomic_failure(self):
        """Valid request emits one complete object; missing tokens emit no assignment."""
        request = {"method": "direct", "tree": nested(), "batch_index": 0,
                   "evaluation_source_ids": ["d2", "missing"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(request), encoding="utf-8")
            command = [sys.executable, str(Path(__file__).with_name("cluster.py")), str(path)]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout), evaluate(request))
            self.assertEqual(result.stderr, "")
            request.update(method="indirect", contextual_threshold=1)
            path.write_text(json.dumps(request), encoding="utf-8")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertIn("source_tokens", result.stderr)


if __name__ == "__main__":
    unittest.main()
