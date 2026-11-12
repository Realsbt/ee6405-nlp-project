from pathlib import Path
import json
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from scripts.prepare_fusion_followup import prepare
from scripts.supervise_fusion_stages import quota_error, safe_target
from cloud.kaggle.run_fusion_dfs import materialize_code


class FusionSupervisionTests(unittest.TestCase):
    def test_quota_requires_explicit_gpu_resource_signal(self):
        for message in ("GPU quota exceeded", "Your weekly GPU quota has been exhausted",
                        "Accelerator usage limit reached", "Insufficient GPU hours",
                        "You do not have enough quota available for GPU",
                        "GPU hours have been exhausted"):
            self.assertTrue(quota_error(message))
        for message in ("CUDA unavailable", "Network timeout", "Out of disk space",
                        "Forward failed", "HTTP 403", "CPU quota exceeded",
                        "GPU memory exhausted", "GPU allocation failed: CUDA out of memory",
                        "GPU: Tesla P100\nDisk quota exceeded"):
            self.assertFalse(quota_error(message))

    def test_collector_limits_scope_and_rejects_traversal(self):
        root = Path("/tmp/evidence")
        self.assertEqual(safe_target("stage/runs/m/best.pt", root, "stage"),
                         root / "stage/runs/m/best.pt")
        self.assertIsNone(safe_target("unrelated/runs/m/best.pt", root, "stage"))
        self.assertIsNone(safe_target("stage/smoke/m/best.pt", root, "stage"))
        self.assertIsNone(safe_target("stage/logs/private.txt", root, "stage"))
        for name in ("/stage/test.json", "stage/../../test.json"):
            with self.assertRaises(ValueError):
                safe_target(name, root, "stage")

    def test_code_package_excludes_outputs_data_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "experiments/configs").mkdir(parents=True)
            (source / "cloud/kaggle").mkdir(parents=True)
            (source / "experiments/configs/neural_v1.json").write_text("{}")
            (source / "experiments/model.py").write_text("pass\n")
            (source / "experiments/test_data.jsonl").write_text('{"private":"text"}')
            (source / "experiments/weights.pt").write_bytes(b"weights")
            (source / "cloud/kaggle/run_fusion_scale.py").write_text("pass\n")
            with patch("scripts.prepare_fusion_followup.ROOT", source):
                prepare(root / "package")
            with zipfile.ZipFile(root / "package/dataset/fusion_followup_code_v1.zip") as bundle:
                self.assertEqual(set(bundle.namelist()), {
                    "experiments/model.py", "experiments/configs/neural_v1.json",
                    "cloud/kaggle/run_fusion_scale.py",
                })

    def make_dfs_package(self, root):
        source = root / "source"
        (source / "experiments/configs").mkdir(parents=True)
        (source / "cloud/kaggle").mkdir(parents=True)
        (source / "experiments/configs/neural_v1.json").write_text("{}")
        (source / "experiments/fusion_dfs_policy.py").write_text("pass\n")
        (source / "cloud/kaggle/run_fusion_dfs.py").write_text("pass\n")
        with patch("scripts.prepare_fusion_followup.ROOT", source):
            prepare(root / "package", "dfs")
        return root / "package/dataset"

    def test_dfs_bootstrap_verifies_zipped_and_kaggle_unpacked_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.make_dfs_package(root)
            zipped = materialize_code(inputs, root / "zipped")
            self.assertTrue(zipped["code_archive_verified"])
            with zipfile.ZipFile(inputs / "fusion_followup_code_v3.zip") as bundle:
                bundle.extractall(inputs)
            (inputs / "fusion_followup_code_v3.zip").unlink()
            unpacked = materialize_code(inputs, root / "unpacked")
            self.assertFalse(unpacked["code_archive_verified"])
            self.assertTrue(unpacked["code_content_verified"])
            self.assertEqual(zipped["code_content_sha256"], unpacked["code_content_sha256"])
            (inputs / "experiments/fusion_dfs_policy.py").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "source content hash"):
                materialize_code(inputs, root / "corrupted")

    def test_dfs_bootstrap_rejects_manifest_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.make_dfs_package(root)
            path = inputs / "fusion_dfs_code_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["files"]["../escape.py"] = "invalid"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                materialize_code(inputs, root / "unsafe")


if __name__ == "__main__":
    unittest.main()
