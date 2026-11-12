import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import AutoModel, DebertaV2Config, RobertaConfig

from experiments.common import LABELS, file_hash
from experiments.train_encoder_pilot import EncoderClassifier, PadCollator, load_partitions, train


def fixture(root):
    partitions = {}
    for split in ("train", "val"):
        rows = [{
            "item_id": f"{split}-{i}", "group_id": f"{split}-{i}",
            "split_block_id": f"{split}-{i}", "source_split": split,
            "label": label, "label_id": i, "domain": "essay", "article": "synthetic example",
        } for i, label in enumerate(LABELS)]
        path = root / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows))
        partitions[split] = {"sha256": file_hash(path), "count": len(rows)}
    (root / "protocol.json").write_text(json.dumps({
        "partitions": partitions, "source_inputs_sha256": "synthetic",
    }))
    (root / "test.jsonl").write_text("not valid JSON and must never be opened")


class TinyTokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": [[2, 4, 5, 3] for _ in texts],
                "attention_mask": [[1, 1, 1, 1] for _ in texts]}

    def pad(self, features, return_tensors):
        length = max(len(row["input_ids"]) for row in features)
        return {
            "input_ids": torch.tensor([row["input_ids"] + [0] * (length - len(row["input_ids"]))
                                       for row in features]),
            "attention_mask": torch.tensor([row["attention_mask"] + [0] * (
                length - len(row["attention_mask"])) for row in features]),
        }

    def save_pretrained(self, path):
        Path(path).mkdir()


class EncoderPilotTests(unittest.TestCase):
    def test_both_architectures_have_finite_gradients_and_same_head(self):
        configs = [
            RobertaConfig(vocab_size=32, hidden_size=24, num_hidden_layers=1,
                          num_attention_heads=4, intermediate_size=48, max_position_embeddings=64),
            DebertaV2Config(vocab_size=32, hidden_size=24, num_hidden_layers=1,
                            num_attention_heads=4, intermediate_size=48, max_position_embeddings=64),
        ]
        heads = []
        for config in configs:
            model = EncoderClassifier(AutoModel.from_config(config), 3407)
            heads.append(model.classifier.weight.detach().clone())
            logits = model({"input_ids": torch.tensor([[2, 4, 5, 3]] * 4),
                            "attention_mask": torch.ones(4, 4, dtype=torch.long)})
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(4))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())
        torch.testing.assert_close(heads[0], heads[1], atol=0, rtol=0)

    def test_dynamic_padding_uses_tokenizer_not_roberta_pad_id(self):
        features, target, ids = PadCollator(TinyTokenizer())([
            ({"input_ids": [2, 3], "attention_mask": [1, 1]}, 0, "a"),
            ({"input_ids": [2, 4, 3], "attention_mask": [1, 1, 1]}, 1, "b"),
        ])
        self.assertEqual(features["input_ids"][0, -1], 0)
        self.assertEqual(features["attention_mask"][0, -1], 0)
        self.assertEqual(ids, ["a", "b"])
        self.assertEqual(target.tolist(), [0, 1])

    def test_loader_never_opens_test_and_rejects_partition_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            partitions, _ = load_partitions(root)
            self.assertEqual(set(partitions), {"train", "val"})
            (root / "val.jsonl").write_text("{}")
            with self.assertRaises(ValueError):
                load_partitions(root)

    def test_full_smoke_entry_point_reloads_without_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            config = RobertaConfig(
                vocab_size=32, hidden_size=24, num_hidden_layers=1,
                num_attention_heads=4, intermediate_size=48, max_position_embeddings=64,
            )
            args = argparse.Namespace(
                data_root=root, condition="roberta_full", roberta=root,
                output=root / "smoke", epochs=4, micro_batch=4, batch_size=16,
                max_seconds=60, device="cpu", seed=3407, smoke=True,
            )
            with patch("experiments.train_encoder_pilot.AutoTokenizer.from_pretrained",
                       return_value=TinyTokenizer()), patch(
                    "experiments.train_encoder_pilot.AutoModel.from_pretrained",
                    return_value=AutoModel.from_config(config)):
                result = train(args)
            self.assertEqual(result["status"], "smoke_completed")
            self.assertFalse(result["test_evaluated"])
            self.assertEqual(result["validation"]["samples"], 4)
            self.assertTrue((args.output / "best.pt").exists())


if __name__ == "__main__":
    unittest.main()
