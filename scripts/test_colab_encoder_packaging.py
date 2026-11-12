import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.prepare_colab_encoder_pilot import prepare


class ColabPackagingTests(unittest.TestCase):
    def test_allowlist_excludes_test_and_weights_and_preserves_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "code/experiments/train_encoder_pilot.py"
            source.parent.mkdir(parents=True)
            source.write_text("# synthetic\n")
            runner = root / "code/cloud/colab/run_encoder_pilot.py"
            runner.parent.mkdir(parents=True)
            runner.write_text("# synthetic runner\n")
            dataset = root / "package/dataset"
            dataset.mkdir(parents=True)
            for name in ("encoder_pilot_code_v1.zip", "pilot_train.jsonl",
                         "pilot_val.jsonl", "pilot_protocol.json"):
                (dataset / name).write_text("{}")
            (dataset / "encoder_pilot_manifest.json").write_text(json.dumps({
                "code_files": {"experiments/train_encoder_pilot.py":
                               hashlib.sha256(source.read_bytes()).hexdigest()},
            }))
            (dataset / "test.jsonl").write_text("private test")
            backbone = root / "backbone"
            backbone.mkdir()
            for name in ("config.json", "merges.txt", "tokenizer.json",
                         "tokenizer_config.json", "vocab.json", "model.safetensors"):
                (backbone / name).write_text("{}")
            output = root / "upload.zip"
            with patch("scripts.prepare_colab_encoder_pilot.ROOT", root / "code"):
                manifest = prepare(root / "package", backbone, output)
                with self.assertRaises(FileExistsError):
                    prepare(root / "package", backbone, output)
            with zipfile.ZipFile(output) as archive:
                self.assertFalse(any("test.jsonl" in n or "model.safetensors" in n
                                     for n in archive.namelist()))
                for name, expected in manifest["files"].items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected)
            self.assertFalse(manifest["test_partition_included"])


if __name__ == "__main__":
    unittest.main()
