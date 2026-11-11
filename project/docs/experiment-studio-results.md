# Kết quả xác nhận Experiment Studio v3

Ngày chạy: **11/11/2026**. Bảng dưới được trích tự động từ `result.json` của các job thật; không điền số liệu minh họa.

**Phạm vi:** 150 users, 24.319 ratings từ mẫu MovieLens đã chuẩn bị; không phải benchmark toàn bộ 25 triệu ratings. Seed 42, chia IID 70/15/15, tuning bật, SVD/NCF tối đa 8 epochs, Q-Learning 500 episodes.

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

## Classification

| Mô hình | F1 CV Mean ± Std | Accuracy Test | F1 Test | AUC-ROC Test | Precision | Recall | Confusion matrix [[TN,FP],[FN,TP]] |
|---|---:|---:|---:|---:|---:|---:|---|
| Logistic Regression | 0.7934 ± 0.0070 | 0.7056 | 0.7928 | 0.7732 | 0.7026 | 0.9097 | [[519, 870], [204, 2055]] |
| Random Forest Classifier | 0.7919 ± 0.0075 | 0.7182 | 0.7932 | 0.7695 | 0.7270 | 0.8725 | [[649, 740], [288, 1971]] |

Chọn bằng Validation: Logistic Regression (f1=0.8026). Khả năng tổng quát hóa: CV f1=0.7934 ± 0.0070; Test sau refit=0.7928. Mô hình được chọn cần 0.228s fit, 0.36 µs/rating và 1034.1 KiB lưu trữ. Inference nhanh nhất: Logistic Regression; model nhỏ nhất: Logistic Regression. Ưu tiên mô hình Validation tốt nhất khi ngân sách đáp ứng các chi phí đo được; nếu cần tối ưu tài nguyên, xác nhận lại ứng viên nhỏ/nhanh trên một thí nghiệm mới. Không chọn lại bằng Test.

Nhãn Thích = rating ≥3,5; F1/Precision/Recall tính cho lớp Thích. Ngưỡng xác suất chọn bằng Validation và khóa trước Test.

## Clustering

| Thuật toán | Tập | Silhouette | DBI | Cụm | Noise |
|---|---|---:|---:|---:|---:|
| K-Means | validation | 0.1931 | 1.4855 | 3 | 0.00% |
| K-Means | test | 0.2088 | 1.4420 | 3 | 0.00% |
| DBSCAN | validation | 0.4306 | 0.8045 | 2 | 81.33% |
| DBSCAN | test | 0.3795 | 0.8860 | 2 | 75.84% |

Silhouette Validation cao nhất: dbscan.

Đọc silhouette cùng noise: DBSCAN có thể giữ ít điểm dễ phân cụm. N/A khi không đủ cụm hợp lệ; không thay bằng 0.

## Apriori

Số luật đạt ngưỡng: **128**; transactions Train: 150. Rating thích ≥4, min_support=0,08, min_confidence=0,30, tối đa 80 phim / itemset dài 2.

| Vế trái | Vế phải | Support | Confidence | Lift |
|---|---|---:|---:|---:|
| Godfather: Part II, The (1974) | Godfather, The (1972) | 0.0933 | 0.7000 | 4.3750 |
| Godfather, The (1972) | Godfather: Part II, The (1974) | 0.0933 | 0.5833 | 4.3750 |
| Lord of the Rings: The Fellowship of the Ring, The (2001) | Lord of the Rings: The Two Towers, The (2002) | 0.0933 | 0.5833 | 3.9773 |
| Lord of the Rings: The Two Towers, The (2002) | Lord of the Rings: The Fellowship of the Ring, The (2001) | 0.0933 | 0.6364 | 3.9773 |
| Goodfellas (1990) | Godfather, The (1972) | 0.0800 | 0.6316 | 3.9474 |

Luật đầu: Godfather: Part II, The (1974) → Godfather, The (1972) (lift=4.375). Có liên kết đồng xuất hiện dương; có thể tạo ứng viên gợi ý và xác nhận trên dữ liệu mới.

## Dimensionality Reduction

| Cấu hình | Components | Explained Variance Train | Reconstruction MSE Test |
|---|---:|---:|---:|
| PCA · 2 | 2 | 0.6071 | 0.4727 |
| PCA · 4 | 4 | 0.7223 | 0.3789 |
| PCA · 8 | 8 | 0.8490 | 0.2578 |
| TruncatedSVD · 2 | 2 | 0.6071 | 0.4727 |
| TruncatedSVD · 4 | 4 | 0.7223 | 0.3789 |
| TruncatedSVD · 8 | 8 | 0.8490 | 0.2578 |

Chọn trên Validation: TruncatedSVD · 8. Ưu tiên ít chiều giữ ≥90% phương sai; nếu không đạt, chọn reconstruction error thấp nhất.

Đây là giảm chiều vector user–genre chuẩn hóa; không phải rating prediction SVD Collaborative Filtering.

## Reinforcement Learning — đánh giá offline

| Policy | Cumulative Reward Test | Average Reward/action Test | Phản hồi đã quan sát | Training (s) |
|---|---:|---:|---:|---:|
| Q-Learning | 10.7778 | 0.0216 | 4.20% | 0.1655 |
| Random Recommendation | 10.8889 | 0.0218 | 5.60% | 0.0001 |
| Popularity Recommendation | 11.5556 | 0.0231 | 4.40% | 0.0000 |

Reward Validation cao nhất: Popularity Recommendation. Kết luận chỉ áp dụng cho môi trường offline được mô tả; kiểm tra observed-feedback coverage trước khi sử dụng.

Cùng 100 episode × 5 hành động cho mọi policy. Reward thiếu phản hồi =0 theo quy ước môi trường; không phải rating giả. Độ phủ phản hồi thấp và thiếu propensity giới hạn kết luận, không chứng minh policy tốt với user thật.

## Dấu vết và kiểm chứng

| Nhóm | Run ID |
|---|---|
| classification | `20261111T115205-e423ebcc` |
| clustering | `20261111T115218-93b2ef30` |
| association | `20261111T115219-7f9c516d` |
| pca | `20261111T115219-4f2657e4` |
| reinforcement | `20261111T115220-f62ab619` |
| rating | `20261111T115220-95c26edd` |

Mỗi thư mục `artifacts/experiment-lab/<run_id>/` chứa `job.json` và `result.json`; mở lại trực tiếp trong lịch sử UI hoặc tải qua API CSV/XLSX/JSON.

Môi trường: Windows-11-10.0.26200-SP0; Intel64 Family 6 Model 154 Stepping 3, GenuineIntel; 16 logical CPUs; RAM 15.7 GiB; Python 3.12.10; CPU inference.

Thời gian là phép đo cục bộ, phụ thuộc tải máy và không phải benchmark triển khai. RSS là toàn tiến trình, model size là pickle trong RAM. Tham số và fingerprint nằm trong artifact gốc.

Đã kiểm tra bộ tests Python, TypeScript/Vite build, sáu nhóm qua Playwright trên dữ liệu thật, xuất dữ liệu và responsive dark/light. Xem [phương pháp và hướng dẫn](experiment-studio.md).
