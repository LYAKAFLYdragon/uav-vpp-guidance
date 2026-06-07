"""Tests for McNemar pairing validator."""

import unittest

from uav_vpp_guidance.evaluation.mcnemar_pairing_validator import (
    build_pairing_key,
    validate_mcnemar_pairing,
    mcnemar_paired_exact_by_key,
    mcnemar_from_dataframe,
)


class TestBuildPairingKey(unittest.TestCase):
    def test_basic_key(self):
        ep = {
            "scenario": "favorable",
            "method": "no_prediction",
            "effective_guidance_mode": "los_rate",
            "training_seed": 0,
            "evaluation_seed": 1,
            "episode_index": 5,
        }
        key = build_pairing_key(ep)
        self.assertEqual(key, ("favorable", "no_prediction", "los_rate", 0, 1, 5))

    def test_fallback_to_guidance_mode(self):
        ep = {
            "scenario": "disadvantage",
            "method": "gru_frozen",
            "guidance_mode": "hybrid",
            "training_seed": 0,
            "eval_seed": 2,
            "episode_index": 3,
        }
        key = build_pairing_key(ep)
        self.assertEqual(key, ("disadvantage", "gru_frozen", "hybrid", 0, 2, 3))

    def test_exclude_method(self):
        ep = {
            "scenario": "favorable",
            "method": "no_prediction",
            "effective_guidance_mode": "los_rate",
            "training_seed": 0,
            "evaluation_seed": 1,
            "episode_index": 5,
        }
        key = build_pairing_key(ep, exclude=["method"])
        self.assertEqual(key, ("favorable", "los_rate", 0, 1, 5))


class TestValidateMcnemarPairing(unittest.TestCase):
    def test_unique_keys_pass(self):
        episodes = [
            {"scenario": "s1", "method": "m1", "effective_guidance_mode": "g1", "training_seed": 0, "evaluation_seed": 0, "episode_index": i}
            for i in range(10)
        ]
        ok, issues = validate_mcnemar_pairing(episodes)
        self.assertTrue(ok)
        self.assertEqual(len(issues), 0)

    def test_duplicate_keys_fail(self):
        episodes = [
            {"scenario": "s1", "method": "m1", "effective_guidance_mode": "g1", "training_seed": 0, "evaluation_seed": 0, "episode_index": 0},
            {"scenario": "s1", "method": "m1", "effective_guidance_mode": "g1", "training_seed": 0, "evaluation_seed": 0, "episode_index": 0},
        ]
        ok, issues = validate_mcnemar_pairing(episodes)
        self.assertFalse(ok)
        self.assertTrue(any("Duplicate" in i for i in issues))

    def test_missing_fields_fail(self):
        episodes = [
            {"scenario": "s1", "method": "m1", "effective_guidance_mode": "g1", "training_seed": 0, "evaluation_seed": 0},
        ]
        ok, issues = validate_mcnemar_pairing(episodes)
        self.assertFalse(ok)
        self.assertTrue(any("missing" in i.lower() for i in issues))


class TestMcnemarPairedExactByKey(unittest.TestCase):
    def test_perfect_agreement(self):
        episodes_a = [
            {"scenario": "s1", "method": "m1", "effective_guidance_mode": "g1", "training_seed": 0, "evaluation_seed": 0, "episode_index": i, "is_success": True}
            for i in range(10)
        ]
        episodes_b = [
            {"scenario": "s1", "method": "m2", "effective_guidance_mode": "g1", "training_seed": 0, "evaluation_seed": 0, "episode_index": i, "is_success": True}
            for i in range(10)
        ]
        # Compare methods: exclude method from pairing key
        results = mcnemar_paired_exact_by_key(episodes_a, episodes_b, exclude_from_key=["method"])
        self.assertEqual(len(results), 1)
        r = list(results.values())[0]
        self.assertEqual(r["n_pairs"], 10)
        self.assertEqual(r["a_success_b_failure"], 0)
        self.assertEqual(r["a_failure_b_success"], 0)
        self.assertEqual(r["mcnemar_exact_p"], 1.0)

    def test_strict_pairing_excludes_unmatched(self):
        episodes_a = [
            {"scenario": "s1", "method": "m1", "effective_guidance_mode": "g1", "training_seed": 0, "evaluation_seed": 0, "episode_index": i, "is_success": i % 2 == 0}
            for i in range(10)
        ]
        # b is missing episode_index 5
        episodes_b = [
            {"scenario": "s1", "method": "m2", "effective_guidance_mode": "g1", "training_seed": 0, "evaluation_seed": 0, "episode_index": i, "is_success": i % 2 == 1}
            for i in range(10) if i != 5
        ]
        results = mcnemar_paired_exact_by_key(episodes_a, episodes_b, exclude_from_key=["method"])
        r = list(results.values())[0]
        self.assertEqual(r["n_pairs"], 9)  # episode 5 excluded
        self.assertEqual(r["missing_in_b"], 1)
        # a success when i even: 0,2,4,6,8 = 5 successes
        # b success when i odd: 1,3,7,9 = 4 successes (excluding 5)
        # For each pair:
        # i=0: a=T, b=F -> b_disc +=1
        # i=1: a=F, b=T -> c_disc +=1
        # i=2: a=T, b=F -> b_disc +=1
        # i=3: a=F, b=T -> c_disc +=1
        # i=4: a=T, b=F -> b_disc +=1
        # i=5: missing
        # i=6: a=T, b=F -> b_disc +=1
        # i=7: a=F, b=T -> c_disc +=1
        # i=8: a=T, b=F -> b_disc +=1
        # i=9: a=F, b=T -> c_disc +=1
        self.assertEqual(r["a_success_b_failure"], 5)
        self.assertEqual(r["a_failure_b_success"], 4)

    def test_shuffled_order_still_pairs_correctly(self):
        episodes_a = [
            {"scenario": "s1", "method": "m1", "effective_guidance_mode": "g1", "training_seed": 0, "evaluation_seed": 0, "episode_index": i, "is_success": True}
            for i in range(10)
        ]
        episodes_b = [
            {"scenario": "s1", "method": "m2", "effective_guidance_mode": "g1", "training_seed": 0, "evaluation_seed": 0, "episode_index": i, "is_success": False}
            for i in range(10)
        ]
        import random
        random.seed(42)
        shuffled_b = episodes_b[:]
        random.shuffle(shuffled_b)
        results = mcnemar_paired_exact_by_key(episodes_a, shuffled_b, exclude_from_key=["method"])
        r = list(results.values())[0]
        self.assertEqual(r["n_pairs"], 10)
        self.assertEqual(r["a_success_b_failure"], 10)
        self.assertEqual(r["a_failure_b_success"], 0)

    def test_guidance_mode_comparison_excludes_guidance(self):
        episodes_a = [
            {"scenario": "s1", "method": "m1", "effective_guidance_mode": "los_rate", "training_seed": 0, "evaluation_seed": 0, "episode_index": i, "is_success": True}
            for i in range(5)
        ]
        episodes_b = [
            {"scenario": "s1", "method": "m1", "effective_guidance_mode": "hybrid", "training_seed": 0, "evaluation_seed": 0, "episode_index": i, "is_success": False}
            for i in range(5)
        ]
        results = mcnemar_paired_exact_by_key(episodes_a, episodes_b, exclude_from_key=["guidance_mode"])
        r = list(results.values())[0]
        self.assertEqual(r["n_pairs"], 5)
        self.assertEqual(r["a_success_b_failure"], 5)
        self.assertEqual(r["a_failure_b_success"], 0)


class TestMcnemarFromDataFrame(unittest.TestCase):
    def test_basic_comparison(self):
        import pandas as pd
        rows = []
        for i in range(10):
            rows.append({
                "scenario": "s1", "method": "m1", "effective_guidance_mode": "g1",
                "training_seed": 0, "evaluation_seed": 0, "episode_index": i,
                "is_success": i % 2 == 0,
            })
            rows.append({
                "scenario": "s1", "method": "m2", "effective_guidance_mode": "g1",
                "training_seed": 0, "evaluation_seed": 0, "episode_index": i,
                "is_success": i % 2 == 1,
            })
        df = pd.DataFrame(rows)
        result = mcnemar_from_dataframe(df, group_cols=["scenario", "effective_guidance_mode"], method_col="method")
        self.assertEqual(len(result), 1)
        r = result.iloc[0]
        self.assertEqual(r["n_pairs"], 10)
        self.assertEqual(r["a_success_b_failure"], 5)
        self.assertEqual(r["a_failure_b_success"], 5)

    def test_missing_columns_raises(self):
        import pandas as pd
        df = pd.DataFrame({"scenario": ["s1"], "method": ["m1"]})
        with self.assertRaises(ValueError) as ctx:
            mcnemar_from_dataframe(df, group_cols=["scenario"], method_col="method")
        self.assertIn("missing", str(ctx.exception).lower())

    def test_shuffle_resistant(self):
        import pandas as pd
        rows = []
        for i in range(10):
            rows.append({
                "scenario": "s1", "method": "m1", "effective_guidance_mode": "g1",
                "training_seed": 0, "evaluation_seed": 0, "episode_index": i,
                "is_success": True,
            })
            rows.append({
                "scenario": "s1", "method": "m2", "effective_guidance_mode": "g1",
                "training_seed": 0, "evaluation_seed": 0, "episode_index": i,
                "is_success": False,
            })
        df = pd.DataFrame(rows)
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        result = mcnemar_from_dataframe(df_shuffled, group_cols=["scenario", "effective_guidance_mode"], method_col="method")
        r = result.iloc[0]
        self.assertEqual(r["n_pairs"], 10)
        self.assertEqual(r["a_success_b_failure"], 10)
        self.assertEqual(r["a_failure_b_success"], 0)


if __name__ == "__main__":
    unittest.main()
