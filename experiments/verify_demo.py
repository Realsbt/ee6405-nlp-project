"""Check actual HTTP-to-model inference, saved references and raw RST outputs."""

import argparse
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from experiments.common import DATA_ROOT, LABELS, read_jsonl, write_json
from experiments.neural import PROJECT_ROOT


SAMPLE = "An author first develops an argument. An editor may then improve its wording, while the main logical structure remains. However, extensive rewriting can also change the argument itself."


def call(base, text, model):
    request = Request(base + "/api/predict", data=json.dumps({"text": text, "model": model}).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=240) as response:
        return json.load(response)


def validate(response):
    scores = np.asarray([response["scores"][name] for name in LABELS])
    assert np.isfinite(scores).all() and np.allclose(scores.sum(), 1)
    assert response["prediction"] == LABELS[int(scores.argmax())]
    assert response["timing"]["end_to_end_seconds"] > 0
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--models", nargs="+", default=["tfidf", "roberta", "race", "race_no_supcon"])
    args = parser.parse_args()
    with urlopen(args.base_url + "/api/models", timeout=10) as response:
        registry = json.load(response)
    available = {row["id"] for row in registry["models"] if row["available"]}
    assert set(args.models) <= available, f"Models still pending: {set(args.models) - available}"
    try:
        call(args.base_url, "   ", args.models[0])
    except HTTPError as error:
        assert error.code == 400
    else:
        raise AssertionError("Empty text was accepted")
    rows = read_jsonl(DATA_ROOT / "test.jsonl")
    row = next(row for row in rows if row["input_truncated"] and len(row["article"]) <= 12000)
    checks = {}
    for model in args.models:
        first = call(args.base_url, SAMPLE, model)
        second = call(args.base_url, SAMPLE, model)
        a, b = validate(first), validate(second)
        assert np.allclose(a, b, rtol=0, atol=1e-6), "Repeated eval-mode prediction differs"
        check = {"first_request": {key: value for key, value in first.items() if key != "rst_tree"},
                 "repeat_request": {key: value for key, value in second.items() if key != "rst_tree"},
                 "repeated_predictions_max_difference": float(np.abs(a-b).max())}
        if model.startswith("race"):
            assert first["rst_tree"] and first["input"]["edu_count"] > 1
            assert first["timing"]["rst_parse_seconds"] > 0
            check["raw_text_parser_to_graph_to_checkpoint_verified"] = True
        if model in ("tfidf", "roberta"):
            reference = call(args.base_url, row["article"], model)
            scores = validate(reference)
            directory = "tfidf_lr_v1" if model == "tfidf" else "roberta_s3407"
            saved = next(record for record in read_jsonl(PROJECT_ROOT / "experiments/reports" / directory / "test_predictions.jsonl") if record["item_id"] == row["item_id"])
            expected = np.asarray([saved["probabilities"][name] for name in LABELS])
            tolerance = 1e-8 if model == "tfidf" else 5e-4
            assert np.allclose(scores, expected, rtol=0, atol=tolerance)
            assert reference["prediction"] == saved["prediction"]
            assert reference["input"]["lexical_truncated"]
            check["long_text_reference"] = {"item_id": row["item_id"], "saved_prediction_matches": True,
                                             "max_probability_difference_cpu_vs_saved": float(np.abs(scores - expected).max()),
                                             "tolerance": tolerance, "truncation_reported": True}
        checks[model] = check
    report = {"status": "passed", "scope": "Actual local HTTP inference; synthetic examples have no authorship labels",
              "models": args.models, "device": registry["device"], "empty_request_rejected": True,
              "timing_context": "CPU Demo while local GPU training / CPU auxiliary parsing may be active; these are observed request times, not isolated benchmark latency",
              "cache_note": "First request may reuse earlier server requests. A cold detector load has detector_model_load_seconds > 0; a cold parser load has non-null rst_parser_initialization. Legacy reports used cold_request/warm_request keys; inspect their actual load fields.",
              "checks": checks}
    suffix = "all" if len(args.models) == 4 else "_".join(args.models)
    path = PROJECT_ROOT / f"experiments/reports/demo_verification_{suffix}.json"
    write_json(path, report)
    print(json.dumps({"status": "passed", "models": args.models, "report": str(path)}, indent=2))


if __name__ == "__main__":
    main()
