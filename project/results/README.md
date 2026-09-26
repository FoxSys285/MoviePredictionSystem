# Kết quả thực nghiệm

Mỗi lần chạy lưu kết quả có thể truy vết tới model, dữ liệu chia tập, seed và cấu hình. Không điền số liệu ví dụ vào bảng kết quả cuối.

- `predictions_<model>_<split>.csv`: `row_id,model,split,userId,movieId,rating_true,rating_pred`. File đầy đủ có thể lớn và được Git bỏ qua.
- `metrics_<model>_<split>.csv` và `metrics_<model>_cv.csv`: `model,task,fold,seed,metric,value,train_time_s,inference_ms,hardware,config`. CV gồm metric từng fold và `mean ± std`.
- `coverage_validation.csv`, `coverage_cv.csv`: số rating toàn bộ, warm-start và các nhóm chưa thấy trong train.
- `data_profile.json`, `data_audit.md`: hồ sơ dữ liệu, dấu vết nguồn và kết quả kiểm tra tập chia.
- `summary.csv`: bảng tổng hợp do TV5 tạo từ các log đã được TV1–TV4 xác nhận.

Các tên file trên là quy ước; thành viên có thể thêm file riêng nhưng cần giữ các cột chung để ghép kết quả.
