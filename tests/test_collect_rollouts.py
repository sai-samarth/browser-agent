import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "collect_rollouts.py"
spec = importlib.util.spec_from_file_location("collect_rollouts", SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load {SCRIPT_PATH}")
collect_rollouts = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = collect_rollouts
spec.loader.exec_module(collect_rollouts)


class ObservationDiagnosticsTests(unittest.TestCase):
    def test_root_only_observation_detected(self):
        obs = {"text": "RootWebArea 'Click Color Task' focused"}
        diag = collect_rollouts._observation_diagnostics(obs)
        self.assertEqual(diag["actionable_node_count"], 0)
        self.assertTrue(diag["is_root_only"])
        self.assertTrue(diag["is_sparse"])

    def test_actionable_observation_not_root_only(self):
        obs = {
            "text": "RootWebArea 'Task'\n[12] textbox ''\n[14] button 'Submit'"
        }
        diag = collect_rollouts._observation_diagnostics(obs)
        self.assertEqual(diag["actionable_node_count"], 2)
        self.assertFalse(diag["is_root_only"])
        self.assertFalse(diag["is_sparse"])


class StepDiagnosticsTests(unittest.TestCase):
    def test_repeated_action_without_progress_counts_as_loop(self):
        diag = collect_rollouts._step_diagnostics(
            action_str="click('13')",
            previous_action_str="click('13')",
            previous_consecutive_same_action_count=1,
            previous_no_progress_streak=1,
            step_reward=0.0,
            done=False,
        )
        self.assertTrue(diag["same_action_as_previous"])
        self.assertEqual(diag["consecutive_same_action_count"], 2)
        self.assertEqual(diag["no_progress_streak"], 2)
        self.assertTrue(diag["repeated_action_loop"])

    def test_progress_resets_no_progress_streak(self):
        diag = collect_rollouts._step_diagnostics(
            action_str="click('19')",
            previous_action_str="click('13')",
            previous_consecutive_same_action_count=3,
            previous_no_progress_streak=4,
            step_reward=1.0,
            done=True,
        )
        self.assertFalse(diag["same_action_as_previous"])
        self.assertEqual(diag["consecutive_same_action_count"], 1)
        self.assertEqual(diag["no_progress_streak"], 0)
        self.assertFalse(diag["repeated_action_loop"])


if __name__ == "__main__":
    unittest.main()
