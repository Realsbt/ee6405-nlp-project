"""Persistent CPU parser process; stdin/stdout carry one JSON message per line."""

from contextlib import redirect_stdout
import json
import queue
import threading
import sys
import time

from experiments.rst_worker import ASSETS, ROOT, leaves, peak_process_rss_mib, tree_dict
from experiments.paths import UPSTREAM
from experiments.rst_alignment import exact_tree_alignment


def main():
    started = time.perf_counter()
    with redirect_stdout(sys.stderr):
        import torch
        from transformers import PreTrainedModel
        from isanlp_rst.parser import Parser
        torch.set_num_threads(2)
        load, resize = torch.load, PreTrainedModel.resize_token_embeddings
        def mapped_load(path, *values, **kwargs):
            if str(path).endswith("best_weights.pt"):
                kwargs["mmap"] = True
            return load(path, *values, **kwargs)
        def resize_without_mean(self, *values, **kwargs):
            kwargs["mean_resizing"] = False
            return resize(self, *values, **kwargs)
        try:
            torch.load = mapped_load
            PreTrainedModel.resize_token_embeddings = resize_without_mean
            parser = Parser(model_dir=str(ASSETS / "isanlp_rst_local"), hf_model_version="rstdt", cuda_device=-1)
        finally:
            torch.load, PreTrainedModel.resize_token_embeddings = load, resize
    names = {line.split("_")[0] for line in (UPSTREAM / "utils/relation_table.txt").read_text().splitlines() if line.strip()}
    names = {name.casefold(): name for name in names}
    print(json.dumps({"status": "ready", "model_load_seconds": time.perf_counter() - started,
                      "device": "cpu", "parser_revision": "5141450ac3eb7decad0b8735fdbf5ec6928ec97a"}), flush=True)
    for line in sys.stdin:
        try:
            message = json.loads(line)
            text = message["text"]
            if not isinstance(text, str) or not text.strip() or len(text) > 12000:
                raise ValueError("Provide nonempty English text of at most 12000 characters")
            started = time.perf_counter()
            with redirect_stdout(sys.stderr):
                with exact_tree_alignment():
                    output = parser(text)
            if not output.get("rst") or len(output["rst"]) != 1:
                raise ValueError("Parser did not return a document tree")
            tree = tree_dict(output["rst"][0], text, names)
            units = leaves(tree)
            assert all(a["end"] <= b["start"] for a, b in zip(units, units[1:]))
            response = {"status": "ok", "rst_structure": tree, "edus": len(units),
                        "parse_seconds": time.perf_counter() - started,
                        "peak_process_rss_mib": peak_process_rss_mib()}
        except Exception as error:
            response = {"status": "error", "error": str(error), "error_type": type(error).__name__}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
