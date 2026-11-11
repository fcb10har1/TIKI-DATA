"""Behavioral regression tests for P4 contracts, inference and evaluation protocol."""

import copy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from evaluation import possession_metrics as pm
from tiki_data.artifacts import (read_batch, read_json, read_predictions, write_batch,
                                 write_json, write_predictions)
from tiki_data.benchmark import (PUBLIC_GT, benchmark, load_manifest, metric_report,
                                 priorities, score_match, smoke, summarize)
from tiki_data.contracts import (CLASS_ORDER, FeatureBatch, MatchSpec, PredictionBatch,
                                 ProbabilityBatch)
from tiki_data.demo import FEATURE_NAMES, generate
from tiki_data.experiments import export
from tiki_data.holdout import freeze
from tiki_data.pipeline import config_inputs, load_config, reproduce, run
from tiki_data.provenance import ROOT, sha256, source_sha256


class TemporaryCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()  # macOS: /var -> /private/var

    def json(self, name, value):
        path = self.root / name
        write_json(path, value)
        return path

    def config(self, predictor, match=None, method="rules", **extra):
        value = {"schema_version": 1, "match": (match or MatchSpec("fixture", 6, 25)).as_dict(),
                 "method": method, "data_kind": "synthetic", "seed": 7, "predictor": predictor}
        value.update(extra)
        return self.json("config.json", value)

    def labels(self, path, spec, labels):
        write_predictions(path, PredictionBatch(spec, np.arange(spec.n_frames), np.array(list(labels))))
        return path

    def demo(self):
        return generate(self.root / "inputs")


class ContractTests(TemporaryCase):
    def setUp(self):
        super().setUp()
        self.spec = MatchSpec("fixture", 6, 10, ((0, 3), (4, 6)))
        self.frames = np.arange(6)
        self.values = np.ones((6, 2), dtype=float)
        self.features = FeatureBatch(self.spec, self.frames, self.values,
                                     np.ones((6, 2), dtype=bool), ("ball_speed", "nearest_home"), self.spec.period_ids())
        self.probabilities = ProbabilityBatch(self.spec, self.frames, np.full((6, 3), 1 / 3), self.spec.period_ids())

    def test_match_bounds_and_supplied_contract(self):
        for fields in ({"n_frames": True}, {"n_frames": 0}, {"fps": 0}, {"fps": float("nan")},
                       {"periods": ((0, 4), (3, 6))}, {"periods": ((-1, 6),)},
                       {"periods": ((0, 7),)}):
            value = {"match_id": "fixture", "n_frames": 6, "fps": 25, **fields}
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                MatchSpec(**value)
        with self.assertRaisesRegex(ValueError, "199319"):
            MatchSpec("bar_ath", 6, 25)
        match = MatchSpec("BAR-ATH", 199319, 25)
        self.assertEqual(match.periods, ((11185, 90652), (110808, 190198)))
        self.assertEqual(self.spec.period_ids().tolist(), [1, 1, 1, 0, 2, 2])

    def test_feature_roundtrip_preserves_order_and_observation_flags(self):
        observed = self.features.observed.copy()
        observed[0, 1] = False
        values = self.values.copy()
        values[0, 1] = 0
        batch = FeatureBatch(self.spec, self.frames, values, observed, self.features.feature_names, self.spec.period_ids())
        write_batch(self.root / "features.npz", batch)
        restored = read_batch(self.root / "features.npz", self.spec, "features")
        np.testing.assert_array_equal(restored.values, values)
        np.testing.assert_array_equal(restored.observed, observed)
        self.assertEqual(restored.feature_names, batch.feature_names)

    def test_invalid_features_fail_instead_of_implicit_imputation(self):
        invalid = []
        for values, observed, names in (
                (np.full((6, 2), np.nan), self.features.observed, ("a", "b")),
                (self.values, np.zeros((6, 2), dtype=bool), ("a", "b")),
                (self.values, np.ones((6, 2), dtype=int), ("a", "b")),
                (self.values, self.features.observed, ("a", "a")),
                (self.values[:, :1], self.features.observed, ("a", "b"))):
            invalid.append(FeatureBatch(self.spec, self.frames, values, observed, names, self.spec.period_ids()))
        for index, batch in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                batch.validate(self.spec)

    def test_probabilities_reject_bad_values_class_order_and_alignment(self):
        for values in (np.full((6, 3), np.nan), np.full((6, 3), np.inf),
                       np.full((6, 3), 0.5), np.full((6, 3), -1), np.zeros((6, 2)),
                       np.ones((6, 3), dtype=int)):
            with self.subTest(values=values.shape), self.assertRaises(ValueError):
                ProbabilityBatch(self.spec, self.frames, values, self.spec.period_ids()).validate(self.spec)
        for fields in ({"class_order": ("A", "H", "D")}, {"frames": self.frames[::-1]},
                       {"periods": np.ones(6, dtype=int)}, {"match": MatchSpec("other", 6, 10)}):
            value = dict(vars(self.probabilities), **fields)
            with self.subTest(fields=list(fields)), self.assertRaises(ValueError):
                ProbabilityBatch(**value).validate(self.spec)

    def test_npz_schema_is_strict_and_pickle_is_disabled(self):
        path = self.root / "bad.npz"
        np.savez(path, schema_version=1, match=json.dumps(self.spec.as_dict()), frames=np.arange(6), periods=self.spec.period_ids(),
                 probabilities=np.full((6, 3), 1 / 3), class_order=np.array(["H", "A", "D"], dtype=object))
        with self.assertRaisesRegex(ValueError, "Object arrays"):
            read_batch(path, self.spec, "probabilities")
        np.savez(path, unexpected=np.ones(1))
        with self.assertRaisesRegex(ValueError, "NPZ fields"):
            read_batch(path, self.spec, "features")

    def test_csv_validation_matches_official_contract(self):
        for text in ("frame,label\n0,H\n", "frame,label\n1,H\n2,A\n3,D\n4,H\n5,A\n6,D\n",
                     "frame,label\n0,H\n1,A\n2,\n3,H\n4,A\n5,D\n",
                     "frame,label\n0,H\n1,A\n2,Z\n3,H\n4,A\n5,D\n",
                     "frame,label\n0,H\n1,A\n1,D\n3,H\n4,A\n5,D\n",
                     "frame,label\n0.0,H\n1.0,A\n2.0,D\n3.0,H\n4.0,A\n5.0,D\n"):
            path = self.root / "invalid.csv"
            path.write_text(text, encoding="utf-8")
            with self.subTest(text=text), self.assertRaises(ValueError):
                read_predictions(path, self.spec)
        path = self.root / "valid.csv"
        path.write_text(" FRAME , LABEL \n0,h\n1, A \n2,d\n3,X\n4,H\n5,A\n", encoding="utf-8")
        self.assertEqual(read_predictions(path, self.spec).labels.tolist(), list("HADXHA"))

    def test_environment_files_are_not_read(self):
        path = self.root / ".env"
        path.write_text("must not be read", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "credential"):
            read_json(path)

    def test_source_newlines_are_portable_but_input_bytes_and_code_changes_are_not_hidden(self):
        lf, crlf = self.root / "lf.py", self.root / "crlf.py"
        lf.write_bytes(b"value = 1\nprint(value)\n")
        crlf.write_bytes(b"value = 1\r\nprint(value)\r\n")
        self.assertEqual(source_sha256(lf), source_sha256(crlf))
        self.assertNotEqual(sha256(lf), sha256(crlf))
        crlf.write_bytes(b"value = 2\r\nprint(value)\r\n")
        self.assertNotEqual(source_sha256(lf), source_sha256(crlf))

    def test_duplicate_json_fields_and_nonfinite_json_are_rejected(self):
        for text in ('{"seed": 0, "seed": 1}', '{"probability": NaN}'):
            path = self.root / "invalid.json"
            path.write_text(text, encoding="utf-8")
            with self.subTest(text=text), self.assertRaises(ValueError):
                read_json(path)


class PipelineTests(TemporaryCase):
    def baseline_config(self, raw=None):
        metadata = self.root / "atd.xml"
        metadata.write_text('<Root><Team id="h"/><Team id="a"/><Player teamId="h"/><Player teamId="a"/></Root>', encoding="utf-8")
        raw_path = self.root / "atd.txt"
        raw_path.write_text(raw or "".join("{}:0.1,0.2,0;0.8,0.7,0:{}\n".format(
            frame, "NaN,NaN,NaN" if frame == 3 else ("0.1,0.2,0" if frame < 4 else "0.8,0.7,0"))
            for frame in range(1, 7)), encoding="utf-8")
        return self.config({"kind": "baseline", "window": 0, "pitch": [105.3, 68],
                            "inputs": {"atd_meta": metadata.name, "atd_raw": raw_path.name}})

    def test_raw_baseline_one_command_and_reproduction(self):
        config = self.baseline_config()
        original = run(config, self.root / "first")
        self.assertEqual(read_predictions(self.root / "first/submission.csv", MatchSpec("fixture", 6, 25)).labels.tolist(),
                         list("HHDAAA"))
        self.assertEqual(original["status"], "complete")
        self.assertEqual(len(original["inputs"]), 2)
        result = reproduce(self.root / "first", self.root / "repeated")
        self.assertTrue(result["identical_csv"])
        self.assertEqual((self.root / "first/submission.csv").read_bytes(),
                         (self.root / "repeated/submission.csv").read_bytes())

    def test_raw_alignment_errors_are_recorded(self):
        config = self.baseline_config(raw="2:0.1,0.2,0;0.8,0.7,0:0.1,0.2,0\n")
        with self.assertRaisesRegex(ValueError, "contiguous"):
            run(config, self.root / "failed")
        self.assertEqual(read_json(self.root / "failed/run.json")["status"], "failed")
        self.assertFalse((self.root / "failed/submission.csv").exists())

    def test_probability_file_argmax_and_ties(self):
        spec = MatchSpec("fixture", 6, 25)
        probabilities = np.array([[0.4, 0.4, 0.2], [0, 1, 0], [0, 0, 1]] * 2, dtype=float)
        write_batch(self.root / "probabilities.npz", ProbabilityBatch(spec, np.arange(6), probabilities, spec.period_ids()))
        config = self.config({"kind": "probabilities", "path": "probabilities.npz"}, method="tcn")
        run(config, self.root / "result")
        self.assertEqual(read_predictions(self.root / "result/submission.csv", spec).labels.tolist(), list("HADHAD"))

    def test_model_feature_order_and_missing_adapter_are_hard_failures(self):
        manifest_path = self.demo()
        manifest = read_json(manifest_path)
        config_path = manifest_path.parent / manifest["matches"][1]["predictions"]["tcn"]["config"]
        config = read_json(config_path)
        config["predictor"]["model"]["expected_features"] = list(reversed(FEATURE_NAMES))
        write_json(config_path, config)
        with self.assertRaisesRegex(ValueError, "feature names/order"):
            run(config_path, self.root / "wrong-order")
        config["predictor"]["model"]["expected_features"] = list(FEATURE_NAMES)
        config["predictor"]["model"]["callable"] = "tiki_data.demo:no_such_model"
        write_json(config_path, config)
        with self.assertRaisesRegex(ValueError, "not callable"):
            run(config_path, self.root / "missing-adapter")

    def test_configuration_paths_resolve_relative_to_file_not_cwd(self):
        config = self.baseline_config()
        loaded = load_config(config)
        self.assertEqual(Path(loaded["predictor"]["inputs"]["atd_raw"]), self.root / "atd.txt")
        self.assertEqual(len(config_inputs(loaded)), 2)

    def test_input_change_invalidates_reproduction(self):
        config = self.baseline_config()
        run(config, self.root / "first")
        with (self.root / "atd.txt").open("a", encoding="utf-8") as handle:
            handle.write("7:NaN:NaN\n")
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            reproduce(self.root / "first", self.root / "repeat")
        self.assertFalse((self.root / "repeat").exists())

    def test_reproduction_allows_relocated_identical_inputs(self):
        config_path = self.baseline_config()
        run(config_path, self.root / "first")
        moved = self.root / "another-setup"
        moved.mkdir()
        for filename in ("atd.txt", "atd.xml"):
            shutil.move(str(self.root / filename), str(moved / filename))
        config = read_json(config_path)
        write_json(moved / "config.json", config)
        result = reproduce(self.root / "first", self.root / "relocated", moved / "config.json")
        self.assertTrue(result["identical_csv"])

    def test_importing_existing_labels_cannot_claim_inference_reproduction(self):
        spec = MatchSpec("fixture", 6, 25)
        self.labels(self.root / "labels.csv", spec, "HHDAAA")
        config = self.config({"kind": "labels", "path": "labels.csv"})
        run(config, self.root / "imported")
        with self.assertRaisesRegex(ValueError, "not inference reproduction"):
            reproduce(self.root / "imported", self.root / "repeat")

    def test_reproduction_detects_nondeterministic_output(self):
        config = self.baseline_config()
        run(config, self.root / "first")
        from tiki_data.pipeline import predict

        def different(*args):
            batch, probabilities, timings = predict(*args)
            changed = batch.labels.copy()
            changed[0] = "A"
            return PredictionBatch(batch.match, batch.frames, changed), probabilities, timings

        with mock.patch("tiki_data.pipeline.predict", side_effect=different), self.assertRaisesRegex(ValueError, "different submission bytes"):
            reproduce(self.root / "first", self.root / "nondeterministic")
        self.assertEqual(read_json(self.root / "nondeterministic/reproduction.json")["status"], "failed")

    def test_input_mutation_during_inference_fails(self):
        config = self.baseline_config()
        from tiki_data.pipeline import predict

        def mutating(*args):
            result = predict(*args)
            with (self.root / "atd.txt").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            return result

        with mock.patch("tiki_data.pipeline.predict", side_effect=mutating), self.assertRaisesRegex(ValueError, "input changed"):
            run(config, self.root / "mutated")
        self.assertEqual(read_json(self.root / "mutated/run.json")["status"], "failed")

    def test_no_overwrite_and_failed_run_log(self):
        config = self.baseline_config()
        out = self.root / "result"
        run(config, out)
        original = (out / "run.json").read_bytes()
        with self.assertRaises(FileExistsError):
            run(config, out)
        self.assertEqual(original, (out / "run.json").read_bytes())
        (self.root / "atd.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            run(config, self.root / "missing-input")
        self.assertEqual(read_json(self.root / "missing-input/run.json")["status"], "failed")

    def test_unknown_configuration_and_boolean_seed_rejected(self):
        config = self.baseline_config()
        for field, value in (("seed", True), ("schema_version", True), ("unexpected", 0)):
            data = read_json(config)
            data[field] = value
            write_json(config, data)
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_config(config)
            config = self.baseline_config()

    def test_example_adapters_cannot_be_mislabelled_as_external(self):
        manifest = self.demo()
        data = read_json(manifest)
        path = manifest.parent / data["matches"][1]["predictions"]["tcn"]["config"]
        config = read_json(path)
        config["data_kind"] = "external"
        write_json(path, config)
        with self.assertRaisesRegex(ValueError, "example adapters require synthetic"):
            load_config(path)

    def test_cli_reports_errors_with_nonzero_exit(self):
        result = subprocess.run([sys.executable, "-m", "tiki_data", "validate", str(self.root / "missing.csv")],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ERROR", result.stderr)


class MetricTests(TemporaryCase):
    def test_all_committed_baselines_match_official_validator_and_metrics(self):
        spec = MatchSpec("BAR-ATH", 199319, 25)
        truth = read_predictions(PUBLIC_GT, spec)
        expected = {"baseline_raw": (0.5972, 0.0378), "baseline_smooth25": (0.6200, 0.2868),
                    "baseline_smooth50": (0.6207, 0.3256), "baseline_smooth125": (0.5812, 0.4242)}
        for stem, scores in expected.items():
            with self.subTest(stem=stem):
                prediction = read_predictions(ROOT / "evaluation/results" / (stem + ".csv"), spec)
                direct = pm.evaluate(pm.label_codes(truth.labels), pm.label_codes(prediction.labels))
                adapted = score_match(truth, prediction, spec)
                self.assertEqual(metric_report(adapted), metric_report(direct))
                self.assertAlmostEqual(adapted["macro_f1"], scores[0], places=4)
                self.assertAlmostEqual(adapted["spell_f1_mean"], scores[1], places=4)

    def test_external_fps_controls_duration_and_transition_tolerances(self):
        spec = MatchSpec("ten_hz", 30, 10)
        labels = np.array(list("H" * 30))
        batch = PredictionBatch(spec, np.arange(30), labels)
        result = score_match(batch, batch, spec)
        self.assertEqual(result["mean_spell_duration_s"], 3)
        self.assertEqual(set(result["transitions"]), {5, 10})
        self.assertEqual(result["spell_f1_mean"], 1)

    def test_adjacent_periods_never_merge_possessions(self):
        spec = MatchSpec("two_halves", 40, 25, ((0, 20), (20, 40)))
        batch = PredictionBatch(spec, np.arange(40), np.array(list("H" * 40)))
        result = score_match(batch, batch, spec)
        self.assertEqual(result["n_gt_spells"], 2)
        self.assertEqual(result["n_pred_spells"], 2)
        self.assertEqual(result["mean_spell_duration_s"], 0.8)
        self.assertEqual(result["transitions"][25]["n_gt"], 0)

    def test_original_frame_numbers_survive_ungraded_gaps(self):
        spec = MatchSpec("gaps", 80, 25)
        labels = np.array(list("X" * 10 + "H" * 10 + "A" * 10 + "X" * 20 + "D" * 10 + "H" * 10 + "X" * 10))
        batch = PredictionBatch(spec, np.arange(80), labels)
        result = score_match(batch, batch, spec, bootstrap=20)
        pairs = result["transitions"][25]["pairs"]
        self.assertEqual(sorted(pair[0][0] for pair in pairs), [20, 60])
        self.assertEqual(result["n_frames"], 80)
        self.assertEqual(result["n_graded"], 40)
        self.assertEqual(result["spell_bootstrap_low"], 1)

    def test_empty_truth_or_labels_outside_periods_fail(self):
        spec = MatchSpec("empty", 6, 25, ((1, 5),))
        for labels in ("XXXXXX", "HHHHHH"):
            batch = PredictionBatch(spec, np.arange(6), np.array(list(labels)))
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                score_match(batch, batch, spec)

    def test_example_decoder_resets_at_adjacent_period_boundary(self):
        from tiki_data.demo import example_decoder
        spec = MatchSpec("reset", 8, 25, ((0, 4), (4, 8)))
        values = np.array([[0.001, 0.998, 0.001]] * 4 + [[0.4, 0.35, 0.25]] * 4)
        batch = ProbabilityBatch(spec, np.arange(8), values, spec.period_ids())
        result = example_decoder(batch, {}, {"switch_cost": 1000})
        self.assertEqual(result.labels.tolist(), list("AAAAHHHH"))

    def test_match_average_does_not_weight_long_matches_more(self):
        entries = [{"method": "rules", "metrics": {"spell_f1_mean": score, "macro_f1": score},
                    "prediction_seconds": duration} for score, duration in ((0, 100), (1, 1))]
        self.assertEqual(summarize(entries)[0]["spell_f1_mean"], 0.5)


class BenchmarkTests(TemporaryCase):
    def test_synthetic_ablation_runs_all_methods_and_retains_diagnostics(self):
        manifest = self.demo()
        report = benchmark(manifest, self.root / "validation", bootstrap=20)
        self.assertFalse(report["accuracy_evidence"])
        self.assertEqual(report["evidence_kind"], "synthetic")
        self.assertEqual({row["method"] for row in report["summary"]}, {"rules", "tcn", "tcn_viterbi"})
        self.assertEqual({row["matches"] for row in report["summary"]}, {2})
        self.assertEqual(len(report["entries"]), 6)
        # The leading fixture has perfect spell matching, so only its measured dead-ball
        # errors qualify. Requiring two findings would force the report to invent one.
        self.assertTrue(1 <= len(report["priorities"]) <= 2)
        self.assertTrue(all(row["count"] > 0 for row in report["priorities"]))
        self.assertEqual(len(list((self.root / "validation/matches").glob("*_transitions.csv"))), 6)
        self.assertTrue((self.root / "validation/report.md").exists())

    def test_two_priorities_are_reported_when_two_error_categories_exist(self):
        spec = MatchSpec("errors", 120, 25)
        frames = np.arange(120)
        gt = PredictionBatch(spec, frames, np.array(list("H" * 40 + "A" * 40 + "D" * 40)))
        pred = PredictionBatch(spec, frames, np.array(list("A" * 40 + "H" * 40 + "D" * 40)))
        result = priorities([{"method": "rules", "metrics": score_match(gt, pred, spec)}], "rules")
        self.assertEqual(len(result), 2)
        self.assertEqual({row["priority"] for row in result},
                         {"missed possessions at IoU 0.5", "spurious possessions at IoU 0.5"})

    def test_group_and_path_leakage_and_supplied_match_rejected(self):
        manifest = self.demo()
        original = read_json(manifest)
        for mutation in ("group", "path", "provided"):
            data = copy.deepcopy(original)
            if mutation == "group":
                data["matches"][3]["group_id"] = data["matches"][0]["group_id"]
            elif mutation == "path":
                data["matches"][3]["ground_truth"] = data["matches"][0]["ground_truth"]
            else:
                data["matches"][3]["group_id"] = "BAR_ATH"
            write_json(manifest, data)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                load_manifest(manifest)

    def test_copied_training_labels_are_not_independent_validation(self):
        manifest = self.demo()
        data = read_json(manifest)
        shutil.copyfile(manifest.parent / data["matches"][0]["ground_truth"],
                        manifest.parent / data["matches"][1]["ground_truth"])
        with self.assertRaisesRegex(ValueError, "duplicate ground-truth"):
            benchmark(manifest, self.root / "invalid")

    def test_inference_cannot_receive_a_ground_truth_file(self):
        manifest = self.demo()
        data = read_json(manifest)
        entry = data["matches"][1]
        path = manifest.parent / entry["predictions"]["tcn"]["config"]
        config = read_json(path)
        config["predictor"]["model"]["inputs"] = {"labels": entry["ground_truth"]}
        write_json(path, config)
        with self.assertRaisesRegex(ValueError, "must not include ground truth"):
            benchmark(manifest, self.root / "leaky-model")

    def test_validation_never_opens_test_labels_or_predictions(self):
        manifest = self.demo()
        data = read_json(manifest)
        data["matches"][3]["ground_truth"] = "untouched_missing_test_labels.csv"
        data["matches"][3]["predictions"] = {method: {"config": "unavailable_test_config.json"}
                                               for method in ("rules", "tcn", "tcn_viterbi")}
        write_json(manifest, data)
        self.assertEqual(benchmark(manifest, self.root / "validation")["status"], "complete")

    def test_public_labels_cannot_be_renamed_and_reformatted_as_external(self):
        manifest = self.demo()
        data = read_json(manifest)
        entry = data["matches"][1]
        entry["match"] = MatchSpec("renamed_public", 199319, 25).as_dict()
        gt = manifest.parent / entry["ground_truth"]
        with gt.open("w", encoding="utf-8", newline="") as handle:
            text = PUBLIC_GT.read_text(encoding="utf-8").replace("frame,label", " FRAME , LABEL ", 1)
            handle.write(text.replace(",H", ",h"))
        self.assertNotEqual(sha256(gt), sha256(PUBLIC_GT))
        entry["predictions"] = {method: {"csv": str(ROOT / "evaluation/results/baseline_smooth50.csv")}
                                for method in ("rules", "tcn", "tcn_viterbi")}
        write_json(manifest, data)
        with self.assertRaisesRegex(ValueError, "cannot be renamed"):
            benchmark(manifest, self.root / "public-copy")

    def test_method_coverage_and_malformed_truth_fail(self):
        manifest = self.demo()
        data = read_json(manifest)
        del data["matches"][1]["predictions"]["tcn"]
        write_json(manifest, data)
        with self.assertRaisesRegex(ValueError, "needs rules"):
            benchmark(manifest, self.root / "missing-method")
        manifest = generate(self.root / "other")
        data = read_json(manifest)
        (manifest.parent / data["matches"][1]["ground_truth"]).write_text("frame,label\n0,Z\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            benchmark(manifest, self.root / "invalid-truth")
        self.assertEqual(read_json(self.root / "invalid-truth/run.json")["status"], "failed")

    def test_public_smoke_is_separate_from_accuracy_evidence(self):
        report = smoke(ROOT / "evaluation/results/baseline_smooth50.csv", self.root / "smoke")
        self.assertFalse(report["accuracy_evidence"])
        self.assertEqual(report["metrics"]["n_graded"], 7500)
        self.assertEqual(report["kind"], "public_smoke")

    def test_experiment_index_contains_failures_and_successes(self):
        manifest = self.demo()
        benchmark(manifest, self.root / "validation")
        bad = self.config({"kind": "labels", "path": "absent.csv"})
        with self.assertRaises(FileNotFoundError):
            run(bad, self.root / "failed")
        count = export(self.root, self.root / "index.csv")
        with (self.root / "index.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(count, 8)
        self.assertEqual({row["status"] for row in rows}, {"complete", "failed"})
        comparison = next(row for row in rows if row["kind"] == "benchmark")
        self.assertEqual(comparison["dataset_id"], "synthetic_integration")
        self.assertEqual(len(json.loads(comparison["comparison_results"])), 3)
        self.assertEqual(export(self.root, self.root / "index.csv"), 8)

    def test_index_cannot_overwrite_predictions_or_an_unrelated_file(self):
        path = self.root / "important.csv"
        path.write_text("frame,label\n0,H\n", encoding="utf-8")
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            export(self.root, path)
        self.assertEqual(path.read_bytes(), before)


class HoldoutTests(TemporaryCase):
    def validated(self):
        manifest = self.demo()
        benchmark(manifest, self.root / "validation")
        freeze_path = self.root / "freeze.json"
        freeze(manifest, self.root / "validation", "tcn_viterbi", freeze_path)
        return manifest, freeze_path

    def test_test_requires_freeze_and_evaluates_only_the_selected_algorithm_once(self):
        manifest = self.demo()
        with self.assertRaisesRegex(ValueError, "requires --freeze"):
            benchmark(manifest, self.root / "unauthorized", split="test")
        benchmark(manifest, self.root / "validation")
        freeze_path = self.root / "freeze.json"
        freeze(manifest, self.root / "validation", "tcn_viterbi", freeze_path)
        report = benchmark(manifest, self.root / "test", split="test", freeze_path=freeze_path)
        self.assertEqual([row["method"] for row in report["summary"]], ["tcn_viterbi"])
        with self.assertRaisesRegex(ValueError, "already attempted"):
            benchmark(manifest, self.root / "test-again", split="test", freeze_path=freeze_path)

    def test_changed_weights_or_features_invalidate_freeze(self):
        manifest, freeze_path = self.validated()
        with (manifest.parent / "synthetic_3_features.npz").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            benchmark(manifest, self.root / "changed", split="test", freeze_path=freeze_path)

    def test_failed_test_attempt_is_retained(self):
        manifest, freeze_path = self.validated()
        data = read_json(manifest)
        (manifest.parent / data["matches"][3]["ground_truth"]).write_text("frame,label\n0,Z\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            benchmark(manifest, self.root / "bad-test", split="test", freeze_path=freeze_path)
        attempts = list((self.root / "test_attempts").glob("*.json"))
        self.assertEqual(len(attempts), 1)
        self.assertEqual(read_json(attempts[0])["status"], "failed")
        with self.assertRaisesRegex(ValueError, "already attempted"):
            benchmark(manifest, self.root / "retry", split="test", freeze_path=freeze_path)

    def test_failure_while_fingerprinting_test_labels_also_consumes_attempt(self):
        manifest, freeze_path = self.validated()
        data = read_json(manifest)
        test_gt = manifest.parent / data["matches"][3]["ground_truth"]
        test_gt.unlink()
        with self.assertRaises(FileNotFoundError):
            benchmark(manifest, self.root / "missing-test", split="test", freeze_path=freeze_path)
        self.assertEqual(len(list((self.root / "test_attempts").glob("*.json"))), 1)
        with self.assertRaisesRegex(ValueError, "already attempted"):
            benchmark(manifest, self.root / "retry-missing", split="test", freeze_path=freeze_path)

    def test_tampered_validation_or_freeze_is_rejected(self):
        manifest, freeze_path = self.validated()
        frozen = read_json(freeze_path)
        frozen["method"] = "rules"
        write_json(freeze_path, frozen)
        with self.assertRaisesRegex(ValueError, "modified"):
            benchmark(manifest, self.root / "tampered", split="test", freeze_path=freeze_path)

    def test_input_change_after_validation_prevents_freeze(self):
        manifest = self.demo()
        benchmark(manifest, self.root / "validation")
        with (manifest.parent / "synthetic_1_features.npz").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "validation inputs changed"):
            freeze(manifest, self.root / "validation", "tcn_viterbi", self.root / "freeze.json")


if __name__ == "__main__":
    unittest.main()
