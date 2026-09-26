# MovieLens 25M — Đồ án Học máy

Dự án gợi ý phim của nhóm 5 thành viên. Phân công và hướng dẫn chi tiết nằm tại [kế hoạch 4 tuần](docs/KE_HOACH_NHOM_4_TUAN.md) và [hướng dẫn công việc](docs/HUONG_DAN_CONG_VIEC_TUNG_THANH_VIEN.md).

## Cấu trúc

```text
project/
├── configs/default.json       # Đường dẫn dữ liệu và quy tắc thực nghiệm chung
├── docs/                      # Kế hoạch và hướng dẫn công việc của nhóm
├── tasks/                     # Checklist 4 tuần cho từng thành viên
├── data/
│   ├── raw/                   # Chỉ có hướng dẫn; dữ liệu gốc đang ở ../Dataset/ml-25m
│   └── processed/             # Train/validation/test sau tiền xử lý (TV1)
├── src/
│   ├── common/                # Tiện ích dùng chung, giao diện model, đo lường
│   ├── data/                  # TV1: EDA, chia tập, baseline, cold-start
│   ├── classical/             # TV2: Linear, XGBoost, Logistic, Random Forest
│   ├── factorization/         # TV3: SVD, KNN, giảm chiều, phân cụm
│   ├── ncf/                   # TV4: Neural Collaborative Filtering
│   ├── association/           # TV5: Apriori, FP-Growth nếu còn thời gian
│   └── app/                   # TV5: web demo
├── artifacts/                  # Model, encoder, ánh xạ ID đã huấn luyện
├── results/                    # CSV dự đoán, metric, bảng so sánh
├── figures/                    # Biểu đồ đưa vào báo cáo
└── report/                     # Phần viết của từng thành viên và báo cáo cuối
```

TV1 đã có pipeline dữ liệu và baseline trong `src/data/`; các thư mục mô hình còn lại là khung để thành viên tương ứng bổ sung mã.

## Theo dõi công việc 4 tuần

Mỗi thành viên đánh dấu tiến độ trong file riêng. Các bảng trong từng file nêu rõ sản phẩm phải tạo, vị trí lưu và điều kiện để xác nhận hoàn thành:

- [TV1 — dữ liệu, chia tập, baseline](tasks/TV1.md)
- [TV2 — regression và classification](tasks/TV2.md)
- [TV3 — SVD, KNN, giảm chiều, phân cụm](tasks/TV3.md)
- [TV4 — NCF](tasks/TV4.md)
- [TV5 — Apriori, web và báo cáo](tasks/TV5.md)

## Dữ liệu

Dữ liệu đã có tại `../Dataset/ml-25m` (tính từ thư mục `project`). File [configs/default.json](configs/default.json) trỏ tới đó; không cần sao chép bộ dữ liệu vào `data/raw`. Xem [nguồn MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) và README đi kèm dữ liệu trước khi sử dụng.

TV1 đã tạo dữ liệu đã chia trong `data/processed`. Cả nhóm dùng chung `train.parquet`, `validation.parquet`, `test.parquet`, `splits.json` và `cv_folds.json`; không tự chia lại theo cách khác. `train.parquet` là tập huấn luyện chính. `train_core.parquet` chỉ dành cho phân tích tương tác đủ dày theo ngưỡng cấu hình. Để so sánh mô hình trên cùng nhóm warm-start, dùng `validation_warm.parquet` và `cv_warm_ids.parquet` (cột `fold,row_id`). Báo cáo số liệu và giới hạn nằm tại [phần TV1](report/tv1_data.md).

## Chuẩn bị môi trường

Chạy các lệnh sau trong thư mục `project`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu thực hiện Q-Learning mở rộng, cài thêm `python -m pip install -r requirements-rl.txt`. Sau khi nhóm xác nhận môi trường chạy được, ghi phiên bản thư viện đã dùng vào báo cáo hoặc file khóa phiên bản để tái lập. Pipeline TV1 đã được chạy bằng Python 3.14.3 và DuckDB 1.5.5 trên máy hiện tại; các phần mô hình khác vẫn là danh sách phụ thuộc dự kiến.

## Chạy pipeline TV1

Từ thư mục `project`, sau khi cài `requirements.txt`:

```powershell
python -m src.data.prepare
python -m src.data.split
python -m src.data.eda
python -m src.data.baseline --split validation
python -m src.data.baseline --cv
python -m src.data.audit
python -m src.data.cold_start --ratings 1:5 296:4.5 --top-n 10
```

`prepare` và `split` yêu cầu `--force` nếu muốn tạo lại file đã có. Baseline test vẫn khóa cho tới khi nhóm ghi quyết định chọn mô hình ở `report/model_selection.md`; sau đó mới chạy `python -m src.data.baseline --split test --allow-test`.

Nếu nhóm đổi ngưỡng `min_user_ratings`/`min_movie_ratings` trong cấu hình, chạy lại `split --force`, baseline validation/CV và audit, rồi cập nhật các số liệu trong báo cáo TV1. `row_id` là khóa chung của mọi file dự đoán; năm fold ở `cv_folds.json` ghi khoảng `row_id` cho train và validation của từng fold.

**Lưu ý về giao thức:** cách chia theo thời gian toàn cục tạo tỷ lệ user mới rất cao trong validation. Các mô hình embedding ID cần báo cáo riêng kết quả warm-start và kết quả toàn tập khi dùng fallback; xem `results/coverage_validation.csv`.

## Quy ước bàn giao

- **Dự đoán rating:** CSV có `row_id,model,split,userId,movieId,rating_true,rating_pred`.
- **Log thực nghiệm:** CSV có `model,task,fold,seed,metric,value,train_time_s,inference_ms,hardware,config`.
- **Model dùng cho web:** lưu model cùng encoder/ánh xạ ID và mô tả cách nạp trong `artifacts/`.
- **Báo cáo:** mỗi người viết phần mình phụ trách trong `report/`; TV5 ghép bản cuối sau khi mọi người xác nhận số liệu.

Không dùng tập test để chọn tham số hoặc mô hình. Điểm chưa có trong ma trận user–item là tương tác chưa quan sát, không phải rating bằng 0.

File dự đoán đầy đủ có thể rất lớn nên `results/predictions_*.csv` chỉ lưu cục bộ và được Git bỏ qua; mã chạy lại và file metric vẫn có thể đưa lên Git.
