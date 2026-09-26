# Kết quả thực nghiệm

Mỗi lần chạy lưu kết quả có thể truy vết tới model, dữ liệu chia tập, seed và cấu hình. Không điền số liệu ví dụ vào bảng kết quả cuối.

- `predictions_<model>_<split>.csv`: `row_id,model,split,userId,movieId,rating_true,rating_pred`.
- `metrics.csv`: `model,task,fold,seed,metric,value,train_time_s,inference_ms,hardware,config`.
- `summary.csv`: bảng tổng hợp do TV5 tạo từ các log đã được TV1–TV4 xác nhận.

Các tên file trên là quy ước; thành viên có thể thêm file riêng nhưng cần giữ các cột chung để ghép kết quả.
