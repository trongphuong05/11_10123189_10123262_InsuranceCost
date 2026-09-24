# Insurance Cost Prediction

Dự đoán **chi phí bảo hiểm y tế (`charges`)** từ `age, sex, bmi, children, smoker, region`.

Repo tuân thủ cấu trúc boilerplate của học phần _Học máy cơ bản (221180)_: `app/` · `ai-models/` · `docs/` · `docker-compose.yml` · `.env.example`.

---

## 1. Thành viên

| Họ tên                  | MSSV     | Phần việc                                                                                                                                                  |
| ----------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Đào Văn Khiêm**       | 10123189 | **ML nền tảng + AI Service**: EDA, Preprocessing, Linear Regression, Decision Tree, AI Service, 1/2 báo cáo (Dữ liệu + EDA + Model)                        |
| **Nguyễn Trọng Phương** | 10123262 | **Model nâng cao + Web App**: Random Forest, SVR, Backend, Frontend, Docker + triển khai, 1/2 báo cáo (Kiến trúc hệ thống + Triển khai + Kết quả kiểm thử) |

### Làm chung

- **Slide thuyết trình (`slide.pptx`)**: mỗi người phụ trách 50% slide theo đúng phần mình thực hiện.
- **Chạy thử toàn hệ thống**: `docker compose up`, smoke test và sửa lỗi tích hợp.
- **README**: cùng hoàn thiện mục "Thành viên" và demo online.

Mỗi người commit bằng tài khoản của mình. Nếu ≥2 người sửa cùng một phần: **nhánh riêng + merge** (không commit đè lên `main`).

---

## 2. Bài toán

- **Loại:** hồi quy (regression)
- **Cột mục tiêu:** `charges` (USD / năm)
- **Đặc trưng:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- **Ý nghĩa thực tế:** ước lượng phí bảo hiểm cá nhân để minh họa quy trình ML → API → app chạy thật

---

## 3. Dữ liệu

- Kaggle: [mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Giấy phép / mô tả cột: [`ai-models/data/DATA.md`](ai-models/data/DATA.md)
- File trong repo: `ai-models/data/dataset.zip` (chứa `insurance.csv`, 1338 dòng)
- Giải nén: `unzip ai-models/data/dataset.zip -d ai-models/data/`

---

## 4. Kết quả model

Cùng split 80/20, `random_state=42`. Dummy là mốc so sánh, **không** tính vào 4 model bắt buộc.

| Model               |  MAE (test) |        RMSE |    R² test | R² train | Predict |   Size | Nhận xét                                        |
| ------------------- | ----------: | ----------: | ---------: | -------: | ------: | -----: | ----------------------------------------------- |
| Dummy (mean)        |     9593.34 |    12465.61 |    -0.0009 |     0.00 |    3 ms |   2 KB | baseline                                        |
| Linear Regression   |     4181.19 |     5796.28 |     0.7836 |   0.7417 |    4 ms |   2 KB | baseline tuyến tính, underfit phần smoker×bmi   |
| Decision Tree       |     2697.77 |     4592.76 |     0.8641 |   0.8653 |    3 ms |   3 KB | GridSearch `max_depth=4`; gần như không overfit |
| **Random Forest ★** | **2447.56** | **4325.34** | **0.8795** |   0.8944 |   37 ms | 501 KB | R²/RMSE tốt nhất, khe train–test nhỏ            |
| SVR                 | **1846.60** |     4828.88 |     0.8498 |   0.8407 |    3 ms |  13 KB | MAE thấp nhất, RMSE kém RF (đuôi charges cao)   |

**Chọn Random Forest** vì R² và RMSE tốt nhất, ổn định hơn một cây đơn, file ~0.5 MB, latency ~37 ms — phù hợp deploy. SVR đáng cân nhắc nếu ưu tiên MAE.

Siêu tham số (GridSearchCV trên tập train):

- DT: `max_depth=4`, `min_samples_leaf=4`
- RF: `n_estimators=200`, `max_depth=6`, `min_samples_leaf=4`
- SVR: `C=10000`, `epsilon=100`, `kernel=rbf`

---

## 5. Đóng gói model

| Artifact                                        | Đường dẫn                                                          |
| ----------------------------------------------- | ------------------------------------------------------------------ |
| Pipeline + model                                | [`ai-models/models/model.joblib`](ai-models/models/model.joblib)   |
| Schema (tên cột, kiểu, min/max, enum)           | [`ai-models/models/schema.json`](ai-models/models/schema.json)     |
| Metadata (metric, version sklearn/numpy/pandas) | [`ai-models/models/metadata.json`](ai-models/models/metadata.json) |

`model.joblib` **chứa cả preprocessor và estimator** (`sklearn.pipeline.Pipeline`). AI Service **không** viết lại bước scale/encode.

### Export từ Colab

1. Mở lần lượt `ai-models/colab/01_eda.ipynb` → `02_preprocess.ipynb` → `03_train.ipynb` → `04_evaluate.ipynb` (Restart & Run All).
2. Notebook 03 ghi `model.joblib`, `schema.json`, `metadata.json`.
3. Tải file về (icon Download hoặc `files.download`) rồi copy vào `ai-models/models/`.
4. Rebuild AI Service: `docker compose up --build ai-service`.

Phiên bản đã ghim (khớp Colab ↔ Docker): `scikit-learn==1.5.1`, `numpy==1.26.4`, `pandas==2.2.2`, `joblib==1.4.2`.

---

## 6. Kiến trúc hệ thống

```
Người dùng  --HTTP-->  Frontend :3000
                         |
                         | POST /api/predict  (X-Request-ID)
                         v
                       Backend :8000  --validate schema.json-->  AI Service :8001
                         |                                         |
                         | lưu lịch sử                             | pipeline.predict
                         v                                         v
                      MongoDB                               models/model.joblib
```

Ba khối FE / BE / AI **mỗi khối một container**, nối bằng `docker-compose.yml` qua **tên service** (không dùng `localhost` giữa các container).

---

## 7. Chạy trên máy

Yêu cầu: Docker + Docker Compose.

```bash
cp .env.example .env
docker compose up --build
```

- App (FE): http://localhost:3000
- Backend docs: http://localhost:8000/docs
- AI Service docs: http://localhost:8001/docs
- Health: `/health` trên từng service (`port`, `uptime_sec`, trạng thái model/DB)

Kiểm tra container:

```bash
docker compose ps
docker compose logs -f
docker compose logs -f frontend backend ai-service
```

---

## 8. Huấn luyện lại model

```bash
pip install -r ai-models/requirements-train.txt
cd ai-models/src
python eda.py
python train.py
python evaluate.py
```

Hoặc chạy 4 notebook trong `ai-models/colab/` theo thứ tự 01→04. Đường dẫn zip: `../data/dataset.zip`.

---

## 9. Biến môi trường

Xem [`.env.example`](.env.example). Tóm tắt:

| Biến                                                  | Ý nghĩa                                                 | Local (compose)                     |
| ----------------------------------------------------- | ------------------------------------------------------- | ----------------------------------- |
| `API_URL`                                             | URL Backend mà **trình duyệt** gọi (khi FE không proxy) | `http://localhost:8000`             |
| `BACKEND_INTERNAL_URL`                                | URL Backend trong mạng Docker (FE proxy `/api`)         | `http://backend:8000`               |
| `AI_SERVICE_URL`                                      | URL AI Service mà Backend gọi                           | `http://ai-service:8001`            |
| `MONGODB_URI`                                         | Chuỗi kết nối Mongo                                     | `mongodb://mongodb:27017/insurance` |
| `FRONTEND_PORT` / `BACKEND_PORT` / `AI_SERVICE_PORT`  | Cổng lắng nghe trong container                          | 80 / 8000 / 8001                    |
| `PUBLIC_APP_URL` / `PUBLIC_API_URL` / `PUBLIC_AI_URL` | URL public/tunnel                                       | để trống khi local                  |
| `CORS_ORIGINS`                                        | Origin được phép gọi BE                                 | `*` (siết lại khi có domain thật)   |
| `MODEL_PATH` / `SCHEMA_PATH` / `METADATA_PATH`        | Artifact trong AI container                             | `/app/models/...`                   |

**Không** hardcode `localhost` hay IP trong source. Đổi tunnel → sửa `.env` + README mục 11 + [`docs/tunnel-changelog.md`](docs/tunnel-changelog.md).

---

## 10. Triển khai

Hai cách (đều chấp nhận):

1. **Cloud:** FE (Vercel/static), BE + AI (Render Docker), MongoDB Atlas. Điền URL vào `.env`.
2. **Máy cá nhân / Colab + tunnel (ngrok):**

```bash
docker compose up --build
ngrok http 3000          # App (FE đã proxy /api)
# nếu muốn public riêng BE/AI: ngrok http 8000 / 8001
```

Cập nhật `.env` và README mỗi lần link đổi. Không có IP tĩnh: **làm lại mỗi sáng thứ Hai**.

---

## 11. Demo online

| Thành phần         | Địa chỉ hiện tại                                             |
| ------------------ | ------------------------------------------------------------ |
| App (FE)           | _(điền sau khi public, ví dụ `https://xxxx.ngrok-free.app`)_ |
| Backend `/docs`    |                                                              |
| AI Service `/docs` |                                                              |
| Cập nhật lúc       | xem `docs/tunnel-changelog.md`                               |

Smoke test sau khi public:

```bash
curl -s $PUBLIC_APP_URL/health
curl -s -X POST $PUBLIC_API_URL/api/predict \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: smoke001' \
  -d '{"features":{"age":30,"sex":"male","bmi":25.5,"children":2,"smoker":"no","region":"southeast"}}'
```

---

## 12. Nhật ký đổi cổng/tunnel

Bảng đầy đủ: [`docs/tunnel-changelog.md`](docs/tunnel-changelog.md).

---

## 13. Kết quả kiểm thử hiệu năng

Chạy test chức năng:

```bash
# AI Service (cần file model.joblib)
pip install -r ai-models/requirements.txt
pytest ai-models/service/tests -q

# Backend (validate, không cần Mongo)
pip install -r app/backend/requirements.txt
pytest app/backend/tests -q
```

Load test gợi ý (k6) sau khi hệ thống đã “nóng”:

```bash
# 15 VU trong 60s, kỳ vọng p95 < 2s, error < 1%
k6 run docs/k6-predict.js
```

Ghi kết quả thật vào đây sau khi public (số VU, p50/p95, error rate, ngưỡng chịu tải).

| Ngày | VU × thời lượng | req/s | p50 | p95 | Error | Ghi chú                   |
| ---- | --------------- | ----- | --- | --- | ----- | ------------------------- |
|      |                 |       |     |     |       | chưa chạy trên URL public |

---

## 14. Hạn chế và hướng phát triển

- Dataset Mỹ, 1338 dòng — không đại diện phí bảo hiểm Việt Nam.
- Không giải thích từng dự đoán (SHAP) trên UI.
- MongoDB local không bật auth; production nên dùng Atlas + user/password trong `.env`.
- Hướng phát triển: log-transform target, XGBoost, CI `docker compose build` trên GitHub Actions, IP tĩnh.

---

## Log theo request_id (dùng khi bảo vệ — Máy 2)

Mỗi service log cùng format:

```
10:15:02 INFO  frontend    req=7f3a1c2b0d11  click Dự đoán -> POST /api/predict
10:15:02 INFO  backend     req=7f3a1c2b0d11  validate OK -> gọi ai-service
10:15:02 INFO  ai-service  req=7f3a1c2b0d11  predict 5834.23 USD model=1.0.0 p=- in 12ms
10:15:02 INFO  backend     req=7f3a1c2b0d11  200 OK total 84ms, saved history
```

```bash
docker compose logs -f frontend backend ai-service
```

Header `X-Request-ID` được FE sinh và chuyển xuyên BE → AI.

## Cấu trúc thư mục

```
├── app/
│   ├── frontend/          # Dockerfile, form theo schema.json
│   └── backend/           # Dockerfile, tests/, validate + MongoDB
├── ai-models/
│   ├── colab/             # 01_eda … 04_evaluate
│   ├── src/               # eda.py, preprocess.py, train.py, evaluate.py
│   ├── data/dataset.zip + DATA.md
│   ├── models/            # model.joblib, schema.json, metadata.json
│   ├── service/           # AI API + tests + nạp model lúc start
│   └── requirements.txt
├── docs/                  # baocao.docx, slide.pptx, figures/
├── docker-compose.yml
├── .env.example
└── README.md
```
