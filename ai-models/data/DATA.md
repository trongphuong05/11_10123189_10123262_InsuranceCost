# Dữ liệu

## Nguồn

- **Tên:** Medical Cost Personal Datasets
- **Kaggle:** https://www.kaggle.com/datasets/mirichoi0218/insurance
- **Tác giả trên Kaggle:** Miri Choi (`mirichoi0218`)
- **Giấy phép:** Open Database License (ODbL) / dataset công khai trên Kaggle (xem trang dataset để xác nhận bản mới nhất)
- **Cỡ:** 1 338 dòng, 7 cột
- **File trong repo:** `ai-models/data/dataset.zip` (chứa `insurance.csv`)

## Bài toán

Hồi quy: dự đoán `charges` (chi phí bảo hiểm y tế, USD) từ 6 đặc trưng cá nhân.

## Cột

| Cột | Kiểu | Vai trò | Miền giá trị (quan sát) |
|---|---|---|---|
| age | number | feature | 18–64 |
| sex | categorical | feature | female, male |
| bmi | number | feature | ≈15.96–53.13 |
| children | number | feature | 0–5 |
| smoker | categorical | feature | yes, no |
| region | categorical | feature | northeast, northwest, southeast, southwest |
| charges | number | **target** | ≈1 122–63 770 USD |

## Cách giải nén

```bash
unzip ai-models/data/dataset.zip -d ai-models/data/
# ra file insurance.csv
```

Notebook Colab đọc trực tiếp từ zip, không bắt buộc giải nén.

## Ghi chú chất lượng

- Không có giá trị thiếu.
- Có 1 dòng trùng; nhóm giữ nguyên để khớp số dòng công bố 1 338.
- Không có cột nào là hàm trực tiếp của `charges` (không rò rỉ nhãn).
