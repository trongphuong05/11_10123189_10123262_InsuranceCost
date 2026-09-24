"""
EDA — tối thiểu 5 hình, mỗi hình có giải thích 3 ý.
Lưu hình vào docs/figures/ và docs/figures/explanations.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocess import FEATURE_ORDER, TARGET, inspect_and_clean, load_raw_dataframe

AI_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = AI_ROOT.parent
FIG_DIR = REPO_ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

EXPLANATIONS = [
    {
        "file": "01_charges_distribution.png",
        "title": "Phân phối biến mục tiêu charges",
        "shows": "Histogram lệch phải: median ~9.4k USD, mean ~13.3k, max ~63.8k. Đuôi dài là nhóm chi phí rất cao.",
        "meaning": "Bài toán hồi quy, target không chuẩn-Gauss. MAE dễ đọc (USD), RMSE sẽ phạt mạnh các ca đắt.",
        "next": "Không biến đổi log bắt buộc (giữ đơn vị USD cho app). Dùng MAE + RMSE + R², không chỉ R².",
    },
    {
        "file": "02_smoker_vs_charges.png",
        "title": "Boxplot smoker vs charges",
        "shows": "smoker=yes có median charges cao gấp ~3–4 lần smoker=no; hai phân phối gần như tách nhóm.",
        "meaning": "smoker là đặc trưng phân biệt mạnh nhất với chi phí. Model tuyến tính thuần có thể underfit nhóm này.",
        "next": "Giữ nguyên cột smoker, one-hot. Không loại outlier của nhóm hút thuốc.",
    },
    {
        "file": "03_age_vs_charges.png",
        "title": "Scatter age vs charges (tô smoker)",
        "shows": "charges tăng theo age gần như 3 dải song song; dải trên hầu hết là smoker=yes.",
        "meaning": "Quan hệ age–charges gần tuyến tính trong từng nhóm, nhưng có tương tác age × smoker.",
        "next": "Cây/ensemble bắt được tương tác; Linear Regression làm baseline để đo phần phi tuyến.",
    },
    {
        "file": "04_bmi_vs_charges.png",
        "title": "Scatter bmi vs charges (tô smoker)",
        "shows": "Với smoker=yes, bmi > ~30 đi kèm charges nhảy vọt. Với smoker=no, bmi gần như phẳng.",
        "meaning": "Có tương tác bmi × smoker, không phải quan hệ tuyến tính đơn.",
        "next": "Không cắt bmi cao. Random Forest / cây sẽ tách nhánh smoker rồi mới tách bmi.",
    },
    {
        "file": "05_region_vs_charges.png",
        "title": "Boxplot region vs charges",
        "shows": "Bốn vùng median gần nhau; southeast hơi rộng hơn một chút, không có vùng ngoại lai hệ thống.",
        "meaning": "region đóng góp yếu so với smoker/age/bmi, nhưng vẫn là đặc trưng hợp lệ.",
        "next": "One-hot region, handle_unknown=ignore. Không gộp vùng.",
    },
    {
        "file": "06_missing_heatmap.png",
        "title": "Bản đồ giá trị thiếu",
        "shows": "Mọi ô đều không thiếu (0%). Có 1 dòng trùng trên 1338 bản ghi.",
        "meaning": "Dữ liệu đáng tin, không cần quy tắc điền phức tạp.",
        "next": "Vẫn gắn SimpleImputer trong pipeline để API không gãy nếu request thiếu số. Giữ duplicate.",
    },
    {
        "file": "07_correlation_heatmap.png",
        "title": "Heatmap tương quan",
        "shows": "smoker_yes tương quan mạnh nhất với charges (~0.79). age ~0.30, bmi ~0.20, children/sex yếu.",
        "meaning": "Không có cặp numeric trùng thông tin (multicollinear nặng). smoker áp đảo.",
        "next": "Giữ đủ 6 cột. Không PCA. Chuẩn hóa numeric vì SVR dùng khoảng cách.",
    },
    {
        "file": "08_smoker_sex_violin.png",
        "title": "Violin charges theo smoker và sex",
        "shows": "Phân phối smoker=yes lệch lên ở cả hai giới; sex gần như không tách nhóm trong từng smoker.",
        "meaning": "sex là đặc trưng yếu; model không nên phụ thuộc sex để 'giải thích' chi phí cao.",
        "next": "Vẫn one-hot sex theo schema gốc. Đánh giá lỗi sẽ tập trung vào nhóm smoker.",
    },
]


def main() -> None:
    df = inspect_and_clean(load_raw_dataframe())
    print(f"EDA rows={len(df)} cols={list(df.columns)}")
    print(df.describe(include="all").transpose())

    # 1
    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET], kde=True, bins=40, color="steelblue")
    plt.title("Hình 1 — Phân phối charges")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_charges_distribution.png", dpi=130)
    plt.close()

    # 2
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="smoker", y=TARGET, hue="smoker", palette="Set2", legend=False)
    plt.title("Hình 2 — charges theo smoker")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_smoker_vs_charges.png", dpi=130)
    plt.close()

    # 3
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="age", y=TARGET, hue="smoker", palette="Set1", alpha=0.7)
    plt.title("Hình 3 — age vs charges")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_age_vs_charges.png", dpi=130)
    plt.close()

    # 4
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="bmi", y=TARGET, hue="smoker", palette="Set1", alpha=0.7)
    plt.title("Hình 4 — bmi vs charges")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_bmi_vs_charges.png", dpi=130)
    plt.close()

    # 5
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="region", y=TARGET, hue="region", palette="Set3", legend=False)
    plt.title("Hình 5 — region vs charges")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_region_vs_charges.png", dpi=130)
    plt.close()

    # 6
    plt.figure(figsize=(8, 4.5))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="YlOrRd")
    plt.title("Hình 6 — Bản đồ giá trị thiếu")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_missing_heatmap.png", dpi=130)
    plt.close()

    # 7
    tmp = df.copy()
    tmp["smoker_yes"] = (tmp["smoker"] == "yes").astype(int)
    tmp["sex_male"] = (tmp["sex"] == "male").astype(int)
    corr = tmp[["age", "bmi", "children", "smoker_yes", "sex_male", TARGET]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1)
    plt.title("Hình 7 — Heatmap tương quan")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_correlation_heatmap.png", dpi=130)
    plt.close()

    # 8
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x="smoker", y=TARGET, hue="sex", split=True, palette="Set2")
    plt.title("Hình 8 — Violin charges theo smoker và sex")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_smoker_sex_violin.png", dpi=130)
    plt.close()

    with open(FIG_DIR / "explanations.json", "w", encoding="utf-8") as f:
        json.dump(EXPLANATIONS, f, ensure_ascii=False, indent=2)

    md_lines = ["# Giải thích hình EDA\n"]
    for i, item in enumerate(EXPLANATIONS, 1):
        md_lines += [
            f"## Hình {i}. {item['title']}\n",
            f"![{item['title']}]({item['file']})\n",
            f"1. **Hình cho thấy gì?** {item['shows']}\n",
            f"2. **Ý nghĩa với bài toán?** {item['meaning']}\n",
            f"3. **Quyết định xử lý tiếp theo?** {item['next']}\n",
        ]
    (FIG_DIR / "README.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Đã lưu {len(EXPLANATIONS)} hình + giải thích vào {FIG_DIR}")


if __name__ == "__main__":
    main()
