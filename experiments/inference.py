"""Local inference for the course Demo, using only verified project checkpoints."""

import atexit
import json
import os
from pathlib import Path

from experiments.paths import STORAGE, BACKBONE, UPSTREAM, resolve_recorded_path, parser_python
import queue
import subprocess
import threading
import time

import joblib
import numpy as np
import torch

from experiments.common import DATA_ROOT, LABELS, file_hash
from experiments.neural import PROJECT_ROOT, TextDataset, build_model, collate_batch, tokenizer_local
from experiments.raw_graph import RawGraphDataset


RUN_NAMES = {"tfidf": "tfidf_lr_v1", "roberta": "roberta_s3407", "race": "race_s3407",
             "race_no_supcon": "race_no_supcon_s3407"}
DISPLAY = {"tfidf": "TF-IDF + Logistic Regression", "roberta": "RoBERTa", "race": "RACE",
           "race_no_supcon": "RACE · no contrastive loss"}


class DetectorService:
    def __init__(self, device="cpu"):
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.set_num_threads(2)
        self.device = device
        self.models = {}
        self.tokenizer = tokenizer_local()
        self.parser = None
        self.parser_queue = None
        self.parser_reader = None
        self.lock = threading.Lock()
        self.runtime = STORAGE / "experiments/demo_runtime"
        self.runtime.mkdir(parents=True, exist_ok=True)
        atexit.register(self.close)

    def registry(self):
        output = []
        for name, run in RUN_NAMES.items():
            report = PROJECT_ROOT / "experiments/reports" / run
            ready = (report / "result.json").exists() and (report / "verification.json").exists()
            reason = "Training or verification pending"
            if ready:
                result = json.loads((report / "result.json").read_text())
                ready = result["status"] == "completed"
                ready = ready and json.loads((report / "verification.json").read_text())["status"] == "passed"
                if ready:
                    required = [resolve_recorded_path(result.get("checkpoint_path", result.get("model_path")))]
                    if name != "tfidf":
                        required += [BACKBONE / "model.safetensors", UPSTREAM / "models/flexible_model.py"]
                    if name.startswith("race"):
                        required += [STORAGE / "assets/isanlp_rst_local/best_weights.pt",
                                     STORAGE / "assets/isanlp_rst_local/config.json"]
                    ready = all(path.is_file() for path in required)
                    reason = "Model resources not installed" if not ready else ""
            output.append({"id": name, "name": DISPLAY[name], "available": ready,
                           "availability_reason": reason,
                           "requires_rst": name.startswith("race"), "run_id": run})
        return {"models": output, "device": self.device, "max_characters": 12000,
                "labels": LABELS, "parser_loaded": self.parser is not None and self.parser.poll() is None}

    def _load_model(self, name):
        if name in self.models:
            return self.models[name], 0.0
        if not any(row["id"] == name and row["available"] for row in self.registry()["models"]):
            raise ValueError("Model unavailable: install its resources and check the verification record")
        started = time.perf_counter()
        run = RUN_NAMES[name]
        report = PROJECT_ROOT / "experiments/reports" / run
        result = json.loads((report / "result.json").read_text())
        if name == "tfidf":
            assert file_hash(result["model_path"]) == result["model_sha256"]
            model = joblib.load(resolve_recorded_path(result["model_path"]))
        else:
            assert file_hash(result["checkpoint_path"]) == result["checkpoint_sha256"]
            saved = torch.load(resolve_recorded_path(result["checkpoint_path"]), map_location="cpu", weights_only=True)
            model = build_model(result["config"], saved.get("metadata")).to(self.device).eval()
            model.load_state_dict(saved["model_state_dict"], strict=True)
        self.models[name] = model
        return model, time.perf_counter() - started

    def _parser_message(self, timeout):
        if self.parser is None or self.parser_queue is None:
            raise RuntimeError("RST parser is not running")
        try:
            line = self.parser_queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("RST parser response timed out; inspect the local parser log")
        if not line:
            raise RuntimeError("RST parser stopped; inspect the local parser log")
        message = json.loads(line)
        if message["status"] == "error":
            raise ValueError(message["error"])
        return message

    def _parse(self, text):
        startup = None
        if self.parser is None or self.parser.poll() is not None:
            log = (self.runtime / "rst_service.stderr.log").open("a")
            environment = {**os.environ, "HF_HOME": str(STORAGE / "cache/huggingface"),
                           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
            self.parser = subprocess.Popen([parser_python(), "-u", "-m", "experiments.rst_service"],
                                           cwd=PROJECT_ROOT, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=log, text=True, encoding="utf-8", bufsize=1)
            log.close()
            self.parser_queue = queue.Queue()
            parser_queue = self.parser_queue
            def read_parser_output():
                try:
                    for line in self.parser.stdout:
                        parser_queue.put(line)
                finally:
                    parser_queue.put(None)
            self.parser_reader = threading.Thread(target=read_parser_output, daemon=True)
            self.parser_reader.start()
            try:
                startup = self._parser_message(180)
                assert startup["status"] == "ready"
            except Exception:
                self.close()
                raise
        self.parser.stdin.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        self.parser.stdin.flush()
        try:
            message = self._parser_message(180)
        except Exception:
            self.close()
            raise
        return message, startup

    def predict(self, text, name):
        if not isinstance(text, str) or not text.strip() or len(text) > 12000:
            raise ValueError("Enter non-empty English text, up to 12,000 characters.")
        if name not in RUN_NAMES:
            raise ValueError("Unknown model")
        with self.lock:
            started = time.perf_counter()
            model, loading_seconds = self._load_model(name)
            encoded = self.tokenizer(text, max_length=512, truncation=True, return_offsets_mapping=True)
            end = max((end for start, end in encoded["offset_mapping"] if end > start), default=0)
            if end == 0:
                raise ValueError("The input has no usable text tokens")
            record = {"item_id": "interactive", "article": text, "label_id": -1}
            parsed, startup = None, None
            if name.startswith("race"):
                parsed, startup = self._parse(text)
                record["rst_structure"] = parsed["rst_structure"]
                dataset = RawGraphDataset([record], self.tokenizer)
            elif name == "roberta":
                dataset = TextDataset([record], self.tokenizer)
            if self.device == "cuda" and name != "tfidf":
                torch.cuda.synchronize()
            forward_started = time.perf_counter()
            if name == "tfidf":
                scores = model.predict_proba([text[:end]])[0]
            else:
                with torch.inference_mode():
                    logits = model(collate_batch([dataset[0]]))["logits"].float()
                    scores = logits.softmax(-1)[0].cpu().numpy()
                if self.device == "cuda":
                    torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - forward_started
            assert np.isfinite(scores).all() and abs(float(sum(scores)) - 1) < 1e-5
            total_seconds = time.perf_counter() - started
            result = {"model": name, "run_id": RUN_NAMES[name], "prediction": LABELS[int(np.argmax(scores))],
                      "scores": dict(zip(LABELS, map(float, scores))), "ai_creator_score": float(scores[2] + scores[3]),
                      "input": {"characters": len(text), "lexical_tokens_with_specials": len(encoded["input_ids"]),
                                "lexical_covered_characters": end, "lexical_truncated": end < len(text.rstrip()),
                                "rst_scope": "full document" if parsed else None, "edu_count": parsed["edus"] if parsed else None},
                      "timing": {"end_to_end_seconds": total_seconds, "detector_model_load_seconds": loading_seconds,
                                 "detector_forward_seconds": forward_seconds,
                                 "rst_parser_initialization": startup,
                                 "rst_parse_seconds": parsed["parse_seconds"] if parsed else None},
                      "device": self.device,
                      "scope": "English HART-trained four-class model; class scores are not calibrated authorship probabilities. HC3 only supports auxiliary binary evaluation."}
            if parsed:
                result["rst_tree"] = parsed["rst_structure"]
            return result

    def close(self):
        if self.parser is not None and self.parser.poll() is None:
            self.parser.terminate()
            try:
                self.parser.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.parser.kill()
                self.parser.wait()
        self.parser = None
        self.parser_queue = None
        self.parser_reader = None
