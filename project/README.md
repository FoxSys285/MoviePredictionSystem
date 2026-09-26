# MovieLens 25M — Đồ án Học máy

Dự án gợi ý phim của nhóm 5 thành viên. Phân công và hướng dẫn chi tiết nằm tại [kế hoạch 4 tuần](../KE_HOACH_NHOM_4_TUAN.md) và [hướng dẫn công việc](../HUONG_DAN_CONG_VIEC_TUNG_THANH_VIEN.md).

## Cấu trúc

```text
project/
├── configs/default.json       # Đường dẫn dữ liệu và quy tắc thực nghiệm chung
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

Các thư mục `src/` hiện là khung để từng thành viên bổ sung mã. Chưa có mô hình được huấn luyện hay kết quả thực nghiệm.

## Dữ liệu

Dữ liệu đã có tại `../Dataset/ml-25m` (tính từ thư mục `project`). File [configs/default.json](configs/default.json) trỏ tới đó; không cần sao chép bộ dữ liệu vào `data/raw`. Xem [nguồn MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) và README đi kèm dữ liệu trước khi sử dụng.

TV1 sẽ tạo dữ liệu đã chia trong `data/processed`. Cả nhóm dùng chung danh sách ID train/validation/test; không tự chia lại theo cách khác.

## Chuẩn bị môi trường

Chạy các lệnh sau trong thư mục `project`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu thực hiện Q-Learning mở rộng, cài thêm `python -m pip install -r requirements-rl.txt`. Sau khi nhóm xác nhận môi trường chạy được, ghi phiên bản thư viện đã dùng vào báo cáo hoặc file khóa phiên bản để tái lập. Các phụ thuộc trong `requirements.txt` là danh sách khởi đầu, chưa được cài hoặc kiểm thử trong thư mục này.

## Quy ước bàn giao

- **Dự đoán rating:** CSV có `row_id,model,split,userId,movieId,rating_true,rating_pred`.
- **Log thực nghiệm:** CSV có `model,task,fold,seed,metric,value,train_time_s,inference_ms,hardware,config`.
- **Model dùng cho web:** lưu model cùng encoder/ánh xạ ID và mô tả cách nạp trong `artifacts/`.
- **Báo cáo:** mỗi người viết phần mình phụ trách trong `report/`; TV5 ghép bản cuối sau khi mọi người xác nhận số liệu.

Không dùng tập test để chọn tham số hoặc mô hình. Điểm chưa có trong ma trận user–item là tương tác chưa quan sát, không phải rating bằng 0.
