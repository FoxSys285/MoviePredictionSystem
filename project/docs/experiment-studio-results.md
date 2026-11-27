# Kết quả hồi quy đã lưu (lần chạy cũ)

Ngày chạy: **11/11/2026**. Bảng dưới được trích tự động từ `result.json` của các job thật; không điền số liệu minh họa.

**Phạm vi cũ:** 150 users, 24.319 ratings từ mẫu MovieLens; seed 42, chia IID 70/15/15, tuning bật. Đây là kết quả hồi quy của lần chạy cũ, không đại diện cho giao thức hiện hành.

| Tập | Ratings |
|---|---:|
| train | 17023 |
| validation | 3648 |
| test | 3648 |

## Regression — so sánh đủ bốn mô hình

| Mô hình | RMSE CV Mean ± Std | RMSE Validation | RMSE Test | MAE Test | Fit cuối (s) | Inference (µs/rating) | Model (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear Regression | 0.9586 ± 0.0065 | 0.9225 | 0.9687 | 0.7402 | 0.0569 | 0.467 | 1032.0 |
| XGBoost Regressor | 1.0455 ± 0.0048 | 1.0100 | 1.0626 | 0.8430 | 0.1755 | 0.982 | 1132.6 |
| SVD Collaborative Filtering | 1.0295 ± 0.0141 | 0.9934 | 1.0107 | 0.7694 | 0.0115 | 0.265 | 246.0 |
| Neural Collaborative Filtering | 1.0376 ± 0.0195 | 0.9934 | 1.0204 | 0.7786 | 0.0261 | 0.289 | 434.3 |

Chọn bằng Validation: Linear Regression (rmse=0.9225). Khả năng tổng quát hóa: CV rmse=0.9586 ± 0.0065; Test sau refit=0.9687. Mô hình được chọn cần 0.057s fit, 0.47 µs/rating và 1032.0 KiB lưu trữ. Inference nhanh nhất: SVD Collaborative Filtering; model nhỏ nhất: SVD Collaborative Filtering. Ưu tiên mô hình Validation tốt nhất khi ngân sách đáp ứng các chi phí đo được; nếu cần tối ưu tài nguyên, xác nhận lại ứng viên nhỏ/nhanh trên một thí nghiệm mới. Không chọn lại bằng Test.

Cấu hình được chọn (Linear dùng bình phương tối thiểu mặc định):

```json
{
  "linear": {},
  "xgboost": {
    "n_estimators": 80,
    "max_depth": 3,
    "learning_rate": 0.005
  },
  "svd": {
    "factors": 8,
    "learning_rate": 0.005,
    "regularization": 0.02,
    "epochs": 8
  },
  "ncf": {
    "factors": 16,
    "hidden_size": 32,
    "learning_rate": 0.005,
    "regularization": 0.02,
    "dropout": 0.1,
    "batch_size": 2048,
    "epochs": 1
  }
}
```

NCF có Train/Validation MSE theo epoch và Early Stopping. CV dùng epoch cố định; refit dùng epoch đã chọn. Learning curve và mọi cấu hình đã thử nằm trong JSON lịch sử.
