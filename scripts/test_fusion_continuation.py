"""Synthetic tests for completed-run preservation, not model performance."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cloud.kaggle.resume_edu_fusion import CORE_SHA256, digest, validate_completed


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-continuation-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.run, self.code = root / "run", root / "code"
        (self.run / "source").mkdir(parents=True)
        (self.code / "experiments").mkdir(parents=True)
        hashes = {}
        for name in CORE_SHA256:
            path = self.code / "experiments" / name
            path.write_text("# Synthetic source fixture\n")
            (self.run / "source" / name).write_bytes(path.read_bytes())
            hashes[name] = digest(path)
        override = patch.dict(CORE_SHA256, hashes, clear=True)
        override.start()
        self.addCleanup(override.stop)
        (self.run / "best.pt").write_bytes(b"Synthetic checkpoint fixture")
        self.result = {
            "status": "validation_completed", "condition": "fusion_cls", "seed": 42,
            "test_evaluated": False, "test": None,
            "best_checkpoint_sha256": digest(self.run / "best.pt"),
        }
        self.write("result.json", self.result)
        self.write("config.json", {
            "config": {"condition": "fusion_cls", "seed": 42, "max_epochs": 20,
                       "batch_size": 16},
            "training_only_smoke": False, "validation_only": True,
            "cache_manifest_sha256": "cache-fixture",
        })
        self.write("source_hashes.json", hashes)

    def write(self, name, value):
        (self.run / name).write_text(json.dumps(value))

    def validate(self, cache_hash="cache-fixture"):
        return validate_completed(self.run, "fusion_cls", 42, self.code, cache_hash)

    def test_complete_frozen_run_is_reusable(self):
        self.assertTrue(self.validate())

    def test_partial_checkpoint_is_not_a_completed_run(self):
        (self.run / "result.json").unlink()
        self.assertFalse(self.validate())

    def test_modified_checkpoint_and_cache_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Cache hash"):
            self.validate("different-cache")
        (self.run / "best.pt").write_bytes(b"Modified synthetic checkpoint")
        with self.assertRaisesRegex(ValueError, "Checkpoint hash"):
            self.validate()

    def test_test_access_or_changed_source_is_rejected(self):
        self.result["test_evaluated"] = True
        self.write("result.json", self.result)
        with self.assertRaisesRegex(ValueError, "Invalid completed"):
            self.validate()
        self.result["test_evaluated"] = False
        self.write("result.json", self.result)
        next((self.run / "source").iterdir()).write_text("# Modified fixture\n")
        with self.assertRaisesRegex(ValueError, "source hash"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
