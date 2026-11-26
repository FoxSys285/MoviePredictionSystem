"""Publication figures from existing result JSON; never trains a model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "experiment-lab"
IDS = {
    "rating": "20261126T015401-4c1f33e3",
    "classification": "20261126T060857-59d1c9fa",
    "clustering": "20261126T063703-2538ed62",
    "reinforcement": "20261126T064844-25de2fc0",
}
plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 12,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grouped(directory, name, labels, data, ylabel, title, errors=None, ylim=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.0), layout="constrained")
    positions = np.arange(len(labels))
    width = .7 / len(data)
    shades = ["0.72", "white", "0.35"]
    hatches = ["", "///", "..."]
    for j, (label, values) in enumerate(data):
        ax.bar(positions + (j - (len(data)-1)/2)*width, values, width,
               color=shades[j], hatch=hatches[j], edgecolor="black", linewidth=.8,
               label=label, yerr=errors if j == 0 else None, capsize=4)
    ax.set_xticks(positions, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="0.88", linewidth=.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="best")
    save(fig, directory, name)


def main(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    runs = {key: json.loads((RUNS / value / "result.json").read_text(encoding="utf-8")) for key, value in IDS.items()}
    rating = runs["rating"]
    models = ["linear", "xgboost", "svd", "ncf"]
    grouped(directory, "regression", ["Linear", "XGBoost", "SVD-CF", "NCF"], [
        ("CV mean ± std (3 fold)", [rating["cv"][key + "_raw"]["metrics"]["rmse"]["mean"] for key in models]),
        ("Test", [next(row["rmse"] for row in rating["rows"] if row["algorithm"] == key and row["split"] == "test") for key in models]),
    ], "RMSE (sao)", "Sai số dự đoán rating", errors=[rating["cv"][key + "_raw"]["metrics"]["rmse"]["std"] for key in models], ylim=(.75,.96))
    curve = next(t for t in rating["tuning"] if t["algorithm"] == "ncf")["learning_curve"]
    fig, ax = plt.subplots(figsize=(7.2,4.0), layout="constrained")
    ax.plot([r["epoch"] for r in curve], [r["train_rating_mse"] for r in curve], "o-", color="black", label="Train MSE (mẫu 50.000)")
    ax.plot([r["epoch"] for r in curve], [r["validation_rmse"]**2 for r in curve], "s--", color="0.35", label="Validation MSE = RMSE²")
    ax.axvline(5, color="0.5", linestyle=":", label="Epoch được chọn: 5")
    ax.set(xlabel="Epoch", ylabel="MSE (sao²)", title="Learning curve NCF")
    ax.legend(frameon=False)
    ax.grid(color="0.88", linewidth=.6)
    save(fig,directory,"ncf")
    classifier = runs["classification"]
    grouped(directory, "classification", ["Logistic Regression", "Random Forest"], [
        ("F1", [next(row["f1"] for row in classifier["rows"] if row["algorithm"] == key and row["split"] == "test") for key in ["logistic","random_forest"]]),
        ("AUC-ROC", [next(row["auc_roc"] for row in classifier["rows"] if row["algorithm"] == key and row["split"] == "test") for key in ["logistic","random_forest"]]),
    ], "Giá trị chỉ số", "Phân loại trên Test", ylim=(0,1))
    ranking = runs["clustering"]["recommendation"]["rows"]
    grouped(directory,"clustering",["Popularity","K-Means","DBSCAN"], [
        (split.title(), [next(row["ndcg_at_10"] for row in ranking if row["algorithm"] == key and row["split"] == split) for key in ["popularity","kmeans","dbscan"]]) for split in ["validation","test"]
    ],"NDCG@10", "Chất lượng gợi ý từ nhóm người dùng", ylim=(0,.08))
    rl = runs["reinforcement"]
    grouped(directory,"rl",["DQN","Random","Popularity"], [
        ("Test", [next(row["average_reward"] for row in rl["rows"] if row["algorithm"] == key and row["split"] == "test") for key in ["dqn","random","popular"]])
    ],"Reward/action", "Reward trên cùng tập episode Test", ylim=(0,.04))
    checkpoint = rl["charts"][3]["series"][0]["points"]
    fig, ax = plt.subplots(figsize=(7.2,4.0), layout="constrained")
    ax.plot([p["x"] for p in checkpoint], [p["y"] for p in checkpoint], "o-",color="black",label="Validation reward/action")
    ax.axvline(15003, color="0.5", linestyle=":",label="Checkpoint được chọn: 15.003")
    ax.set(xlabel="Training step",ylabel="Reward/action",title="DQN: reward đánh giá tại các checkpoint")
    ax.legend(frameon=False)
    ax.grid(color="0.88", linewidth=.6)
    save(fig,directory,"rl_curve")
    report = json.loads((ROOT.parent/"dataset_cleaned"/"cleaning_report.json").read_text(encoding="utf-8"))
    distribution = report["files"]["ratings.csv"]["rating_distribution"]
    grouped(directory,"ratings",[str(k).replace(".",",") for k in distribution],[("Số lượt",[v/1e6 for v in distribution.values()])],"Số lượt đánh giá (triệu)","Phân bố rating MovieLens 25M",ylim=(0,7.5))
    print("Created 7 PNG/PDF figures from saved evidence.")


if __name__ == "__main__":
    main(sys.argv[1])
