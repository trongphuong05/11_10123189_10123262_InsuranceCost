"""
Đánh giá và vẽ biểu đồ so sánh 4 model (+ baseline Dummy).
Đọc results.csv do train.py sinh ra.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

AI_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = AI_ROOT.parent
RESULTS = AI_ROOT / "models" / "results.csv"
FIG_DIR = REPO_ROOT / "docs" / "figures"


def main() -> None:
    df = pd.read_csv(RESULTS)
    print("=== Bảng so sánh ===")
    print(df.to_string(index=False))

    chosen = df.loc[df["chosen"] == True]  # noqa: E712
    print("\nModel được chọn:")
    print(chosen.to_string(index=False))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    colors = [
        "#94a3b8" if m == "dummy_mean" else ("#0f6e62" if bool(c) else "#64748b")
        for m, c in zip(df.model, df.chosen)
    ]
    labels = df.model.str.replace("_", "\n")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].bar(range(len(df)), df.R2, color=colors)
    axes[0].set_xticks(range(len(df)))
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_title("R² test — càng cao càng tốt")
    axes[0].set_ylim(0, 1)

    axes[1].bar(range(len(df)), df.RMSE, color=colors)
    axes[1].set_xticks(range(len(df)))
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_title("RMSE (USD) — càng thấp càng tốt")

    axes[2].bar(range(len(df)), df.MAE, color=colors)
    axes[2].set_xticks(range(len(df)))
    axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_title("MAE (USD) — càng thấp càng tốt")

    plt.tight_layout()
    out = FIG_DIR / "09_model_comparison.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print("Đã lưu", out)


if __name__ == "__main__":
    main()
