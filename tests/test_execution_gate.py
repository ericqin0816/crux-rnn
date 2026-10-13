import sys
import unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from execution_provenance import git_execution_identity, require_phase_authorization


class ExecutionGateTests(unittest.TestCase):
    def test_final_blocked_without_approval(self):
        with self.assertRaisesRegex(PermissionError, "pending explicit user approval"):
            require_phase_authorization({"phase": "final"}, {"final_training_allowed": False})
        with self.assertRaises(PermissionError):
            require_phase_authorization({"phase": "final"}, {})

    def test_pilot_allowed_and_explicit_final_gate(self):
        require_phase_authorization({"phase": "pilot"}, {"final_training_allowed": False})
        require_phase_authorization({"phase": "final"}, {"final_training_allowed": True})

    def test_execution_identity_rejects_dirty_tree(self):
        with mock.patch("execution_provenance.subprocess.check_output", side_effect=["a" * 40, " M model.py"]):
            with self.assertRaisesRegex(RuntimeError, "Commit tracked"):
                git_execution_identity()
        with mock.patch("execution_provenance.subprocess.check_output", side_effect=["b" * 40, ""]):
            self.assertEqual(git_execution_identity(), "b" * 40)
