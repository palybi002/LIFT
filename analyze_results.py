import glob
import os
import re
from typing import Dict, List, Optional

import pandas as pd


FLOAT_PATTERN = r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|nan|NaN|inf|-inf)"


def _to_float(value: Optional[str]):
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _extract_namespace(content: str) -> Dict:
    ns_match = re.search(r"Namespace\((.*?)\)", content, flags=re.DOTALL)
    if not ns_match:
        return {}
    ns_text = ns_match.group(1)

    def pick_str(key: str):
        m = re.search(rf"{key}='([^']*)'", ns_text)
        return m.group(1) if m else None

    def pick_int(key: str):
        m = re.search(rf"{key}=([0-9]+)", ns_text)
        return int(m.group(1)) if m else None

    def pick_bool(key: str):
        m = re.search(rf"{key}=(True|False)", ns_text)
        return (m.group(1) == "True") if m else None

    return {
        "model": pick_str("model"),
        "dataset": pick_str("dataset"),
        "features": pick_str("features"),
        "lift": pick_bool("lift"),
        "leader_num": pick_int("leader_num"),
        "state_num": pick_int("state_num"),
        "top_k": pick_int("top_k"),
        "train_epochs": pick_int("train_epochs"),
        "only_test": pick_bool("only_test"),
    }


def _extract_metrics(content: str) -> Dict:
    metrics = {
        "MSE": None,
        "MAE": None,
        "Params": None,
        "TrainTime": None,
        "InferenceTime": None,
    }

    metric_match = re.search(rf"mse:{FLOAT_PATTERN},\s*mae:{FLOAT_PATTERN}", content)
    if metric_match:
        metrics["MSE"] = _to_float(metric_match.group(1))
        metrics["MAE"] = _to_float(metric_match.group(2))

    params_match = re.search(r"(?:Model Params|Number of Params):\s*([0-9]+)", content)
    if params_match:
        metrics["Params"] = int(params_match.group(1))

    epoch_times = [float(x) for x in re.findall(r"cost time:\s*([0-9.]+)", content)]
    if epoch_times:
        metrics["TrainTime"] = sum(epoch_times) / len(epoch_times)

    infer_match = re.search(r"Inference Time:\s*([0-9.]+)\s*s/batch", content)
    if infer_match:
        metrics["InferenceTime"] = float(infer_match.group(1))

    return metrics


def _parse_log_file(filepath: str) -> Dict:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    filename = os.path.basename(filepath)
    namespace = _extract_namespace(content)
    metrics = _extract_metrics(content)

    row = {
        "LogFile": filename,
        "RunType": "unknown",
        "Model": namespace.get("model"),
        "Dataset": namespace.get("dataset"),
        "Features": namespace.get("features"),
        "lift": namespace.get("lift"),
        "leader_num": namespace.get("leader_num"),
        "state_num": namespace.get("state_num"),
        "top_k": namespace.get("top_k"),
        "train_epochs": namespace.get("train_epochs"),
        "only_test": namespace.get("only_test"),
    }
    row.update(metrics)

    # Fallback from filename for old logs: Model_Dataset_Features.log
    if not row["Model"] or not row["Dataset"] or not row["Features"]:
        parts = filename.replace(".log", "").split("_")
        if len(parts) >= 3:
            row["Model"] = row["Model"] or parts[0]
            row["Dataset"] = row["Dataset"] or parts[1]
            row["Features"] = row["Features"] or parts[2]

    if filename.startswith("cmp_"):
        row["RunType"] = "comparison"
    elif filename.startswith("abl_"):
        row["RunType"] = "ablation"
    elif row.get("top_k") not in [None, 3] or row.get("lift"):
        row["RunType"] = "ablation"
    else:
        row["RunType"] = "comparison"

    # Human-readable ablation label.
    if row["RunType"] == "ablation":
        if row.get("Model") == "LACFNet":
            row["AblationType"] = "LACFNet_top_k"
            row["AblationValue"] = row.get("top_k")
        elif row.get("lift"):
            if "leader" in filename:
                row["AblationType"] = "LIFT_leader_num"
                row["AblationValue"] = row.get("leader_num")
            elif "state" in filename:
                row["AblationType"] = "LIFT_state_num"
                row["AblationValue"] = row.get("state_num")
            else:
                row["AblationType"] = "LIFT_misc"
                row["AblationValue"] = None
        else:
            row["AblationType"] = "other"
            row["AblationValue"] = None
    else:
        row["AblationType"] = None
        row["AblationValue"] = None

    return row


def _keep_valid(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["Dataset"].notna() & df["Model"].notna()].copy()


def _print_summary(title: str, df: pd.DataFrame, columns: List[str]):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    if df.empty:
        print("No valid rows.")
        return
    print(df[columns].to_string(index=False))


def main():
    log_files = sorted(glob.glob("logs/*.log"))
    if not log_files:
        print("No log files found in logs/ directory.")
        return

    rows = []
    for log_file in log_files:
        try:
            rows.append(_parse_log_file(log_file))
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")

    all_df = _keep_valid(pd.DataFrame(rows))
    if all_df.empty:
        print("No valid parsed rows found.")
        return

    comp_df = all_df[all_df["RunType"] == "comparison"].copy()
    abl_df = all_df[all_df["RunType"] == "ablation"].copy()

    # Keep latest run per log file and stable order for readability.
    if not comp_df.empty:
        comp_df = comp_df.sort_values(by=["Dataset", "Model", "Features", "MSE"], na_position="last")
        comp_df.to_csv("comparison_results.csv", index=False)

    if not abl_df.empty:
        abl_df = abl_df.sort_values(by=["Dataset", "AblationType", "AblationValue", "MSE"], na_position="last")
        abl_df.to_csv("ablation_results.csv", index=False)

    _print_summary(
        "COMPARISON RESULTS SUMMARY",
        comp_df,
        ["Dataset", "Model", "Features", "MSE", "MAE", "Params", "TrainTime", "LogFile"],
    )
    _print_summary(
        "ABLATION RESULTS SUMMARY",
        abl_df,
        ["Dataset", "AblationType", "AblationValue", "Model", "MSE", "MAE", "Params", "TrainTime", "LogFile"],
    )

    print("\nSaved: comparison_results.csv")
    if not abl_df.empty:
        print("Saved: ablation_results.csv")


if __name__ == "__main__":
    main()
