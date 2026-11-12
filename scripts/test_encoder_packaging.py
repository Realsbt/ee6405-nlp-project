import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from cloud.kaggle.run_encoder_pilot import materialize_sources, safe_relative
from scripts.prepare_encoder_pilot import SOURCES


class EncoderPackagingTests(unittest.TestCase):
    def test_no_source_data_or_model_weights_in_code_archive(self):
        self.assertTrue(all(name.endswith(".py") for name in SOURCES))
        self.assertNotIn("test", " ".join(SOURCES))
        self.assertIn("experiments/train_encoder_pilot.py", SOURCES)

    def test_bootstrap_rejects_path_traversal(self):
        for name in ("../key", "/tmp/key", "experiments/../../key"):
            with self.assertRaises(ValueError):
                safe_relative(name)
        self.assertEqual(str(safe_relative("experiments/train_encoder_pilot.py")),
                         "experiments/train_encoder_pilot.py")

    def test_bootstrap_accepts_zip_or_nested_auto_unpacked_code(self):
        name = "experiments/train_encoder_pilot.py"
        content = b"# synthetic source\n"
        manifest = {"code_files": {name: hashlib.sha256(content).hexdigest()}}
        for packed in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                mount = root / "mount"
                mount.mkdir()
                if packed:
                    with zipfile.ZipFile(mount / "encoder_pilot_code_v1.zip", "w") as archive:
                        archive.writestr(name, content)
                else:
                    source = mount / "encoder_pilot_code_v1" / name
                    source.parent.mkdir(parents=True)
                    source.write_bytes(content)
                result = materialize_sources(mount, root / "code", manifest)
                self.assertEqual(result, manifest["code_files"])


if __name__ == "__main__":
    unittest.main()
