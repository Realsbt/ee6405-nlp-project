"""Verify pinned parser/tokenizer bytes using preserved HF download metadata."""

import hashlib
import json
from pathlib import Path

from experiments.common import ASSETS_ROOT, file_hash, write_json


ASSETS = ASSETS_ROOT


def verify_download(root, filename, revision):
    path = root / filename
    metadata = root / ".cache/huggingface/download" / f"{filename}.metadata"
    recorded_revision, etag, *_ = metadata.read_text().splitlines()
    assert recorded_revision == revision, "Download metadata revision mismatch"
    sha256 = file_hash(path)
    if len(etag) == 64:
        actual = sha256
        algorithm = "LFS SHA-256"
    elif len(etag) == 40:
        digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                digest.update(chunk)
        actual, algorithm = digest.hexdigest(), "Git blob SHA-1"
    else:
        raise ValueError(f"Unrecognized HF etag format for {filename}")
    assert actual == etag, f"Pinned source bytes differ: {path}"
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256,
            "revision": revision, "download_etag": etag, "etag_algorithm": algorithm,
            "matches_preserved_download_metadata": True}


def main():
    parser_revision = "5141450ac3eb7decad0b8735fdbf5ec6928ec97a"
    encoder_revision = "c23d21b0620b635a76227c604d44e43a9f0ee389"
    source, encoder, local = ASSETS / "isanlp_rst_rstdt", ASSETS / "xlm-roberta-large-config", ASSETS / "isanlp_rst_local"
    files = [verify_download(source, filename, parser_revision) for filename in ("best_weights.pt", "config.json", "relation_table.txt")]
    files += [verify_download(encoder, filename, encoder_revision) for filename in ("config.json", "tokenizer_config.json", "tokenizer.json", "sentencepiece.bpe.model")]
    original = json.loads((source / "config.json").read_text())
    original["model"]["transformer"]["model_name"] = str(encoder)
    assert json.loads((local / "config.json").read_text()) == original
    assert (local / "best_weights.pt").resolve() == (source / "best_weights.pt").resolve()
    assert (local / "relation_table.txt").resolve() == (source / "relation_table.txt").resolve()
    result = {"status": "passed", "files": files,
              "source": "Offline verification against preserved HF download revision + etag metadata; no new remote verification or download",
              "parser_repo": "tchewik/isanlp_rst_v3", "encoder_config_tokenizer_repo": "FacebookAI/xlm-roberta-large",
              "local_config_only_changes_model_name_to_local_tokenizer": True,
              "local_config_sha256": file_hash(local / "config.json"),
              "checkpoint_and_relations_symlink_to_verified_source": True,
              "audit_script_sha256": file_hash(__file__)}
    path = Path(__file__).parent / "reports/parser_assets.json"
    write_json(path, result)
    print(json.dumps({"status": "passed", "verified_files": len(files), "parser_checkpoint_sha256": files[0]["sha256"],
                      "parser_checkpoint_bytes": files[0]["bytes"]}, indent=2))


if __name__ == "__main__":
    main()
