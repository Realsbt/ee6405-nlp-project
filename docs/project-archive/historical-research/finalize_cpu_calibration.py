"""Consolidate completed replay/calibration evidence without changing source reports."""

import json
import platform
import sys
from race_cpu_replay import ROOT, file_hash, write


def main():
    race = ROOT / "race_cpu_v1"
    hart = json.loads((race / "hart_report.json").read_text())
    hc3 = json.loads((race / "hc3_report.json").read_text())
    coverage = json.loads((race / "hc3_parse_coverage.json").read_text())
    assert hart["n_validation"] == 1600 and hart["n_test"] == 3194 and hc3["n_test"] == 1000
    assert coverage["status"] == "passed" and coverage["records"] == 1000
    rows, temperatures = [], {}
    for model in ("tfidf", "roberta", "race"):
        if model == "race":
            report = {"hart": hart["metrics"], "hc3": hc3["metrics"]}
            config = race / "calibration.json"
        else:
            report = json.loads((ROOT / f"saved_temperature_v1/{model}_report.json").read_text())
            config = ROOT / f"saved_temperature_v1/{model}_calibration.json"
        temperatures[model] = json.loads(config.read_text())["temperature"]
        for dataset, conditions in report.items():
            for condition in ("uncalibrated", "temperature_scaled"):
                c = conditions[condition]["creator"]
                operating = c["at_threshold"]["hart_validation_locked"]
                rows.append({"model": model, "dataset": dataset, "condition": condition,
                             "brier": c["binary_brier"], "ece_10bins": c["positive_class_ece_10_bins"],
                             "auroc": c["auroc"], "locked_fpr": operating["fpr"],
                             "locked_tpr": operating["tpr"],
                             "macro_f1_4class": conditions[condition].get("four_class", {}).get("macro_f1")})
    paths = list(race.glob("*.json")) + list(race.glob("*.jsonl"))
    paths += list((ROOT / "saved_temperature_v1").glob("*.json"))
    paths += [ROOT / name for name in (
        "race_cpu_replay.py", "calibrate_saved_models.py", "audit_cpu_parses.py",
        "evaluation_metrics.py", "fetch_cpu_assets.py", "test_race_cpu.py", "finalize_cpu_calibration.py")]
    result = {"execution_status": "completed", "platform": platform.platform(), "python": sys.version,
              "replay": {"hart": hart["replay"], "hc3": hc3["replay"]},
              "temperatures": temperatures, "metrics": rows,
              "hc3_parsing": coverage,
              "evidence_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in paths},
              "limitations": [
                  "RACE seed3407 CE+SupCon only; no new neural training.",
                  "HART val was also used for checkpoint selection; post-hoc calibration.",
                  "HC3 CPU reparsing does not establish tree identity with the old unavailable parse file.",
                  "Temperature fitting used four-class NLL, not external-test Brier.",
                  "No new confidence intervals or multiplicity-adjusted significance tests.",
                  "Original results and model code preserved.",
              ]}
    write(ROOT / "cpu_calibration_completed_v1.json", result)
    print(json.dumps({k: result[k] for k in ("execution_status", "replay", "temperatures", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
