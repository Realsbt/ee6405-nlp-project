import unittest
from pathlib import Path

from scripts.fetch_resources import digest, materialize_resource


def row(path):
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


class ResourceLinkTests(unittest.TestCase):
    def test_auto_materializes_and_is_idempotent(self):
        source = self.test_dir / "source.bin"
        source.write_bytes(b"portable-resource")
        target = self.test_dir / "nested" / "target.bin"
        self.assertIn(materialize_resource(source, target, row(source)), {"symlink", "hardlink", "copy"})
        self.assertEqual(target.read_bytes(), source.read_bytes())
        self.assertEqual(materialize_resource(source, target, row(source)), "existing")

    def test_copy_mode_repairs_dangling_link(self):
        source = self.test_dir / "source.bin"
        source.write_bytes(b"portable-resource")
        target = self.test_dir / "target.bin"
        target.symlink_to(self.test_dir / "missing.bin")
        self.assertEqual(materialize_resource(source, target, row(source), "copy"), "copy")
        self.assertEqual(target.read_bytes(), source.read_bytes())

    def test_conflicting_file_is_preserved(self):
        source = self.test_dir / "source.bin"
        source.write_bytes(b"source")
        target = self.test_dir / "target.bin"
        target.write_bytes(b"unknown")
        with self.assertRaises(RuntimeError):
            materialize_resource(source, target, row(source), "copy")
        self.assertEqual(target.read_bytes(), b"unknown")

    def setUp(self):
        import tempfile
        self._temporary = tempfile.TemporaryDirectory()
        self.test_dir = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
