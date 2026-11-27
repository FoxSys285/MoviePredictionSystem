# Kết quả hồi quy rating

Trang Experiment Studio là nguồn kết quả hiện hành. Mỗi lần chạy lưu cấu hình, chỉ số, thời gian fit, biểu đồ và nhật ký theo mã run; có thể xuất kết quả thành CSV, XLSX hoặc JSON.

## Thuật toán

Lab hiện chạy bốn mô hình dự đoán rating:

- Linear Regression
- XGBoost
- SVD
- Neural Collaborative Filtering (NCF)

PCA là lựa chọn biểu diễn dành cho Linear Regression, không phải một thuật toán dự đoán riêng.

## Chỉ số và chọn mô hình

RMSE và MAE đo độ lệch giữa rating dự đoán và rating thật trên thang 0,5–5 sao. Cả hai càng thấp càng tốt. RMSE phạt sai số lớn mạnh hơn; MAE diễn giải trực tiếp sai lệch tuyệt đối trung bình theo đơn vị sao.

Global Mean là baseline: dự đoán mỗi rating bằng trung bình của tập Train. So các mô hình với cùng baseline và cùng cách chia dữ liệu.

Tham số được đánh giá bằng năm fold trên Train. Validation dùng để so sánh các mô hình; chỉ dùng Test cho báo cáo cuối, không chọn lại mô hình sau khi xem điểm Test.

## Nhật ký chạy

Log ghi tiến độ, thông báo từ quá trình fit, cảnh báo, lỗi và trạng thái tác vụ. Chọn một lần chạy trong lịch sử để xem lại log; tải toàn bộ log khi cần lưu hoặc phân tích bên ngoài.

Kết quả của các lần chạy cũ có thể dùng giao thức khác. Hãy đọc cấu hình và protocol đi cùng từng run trước khi so điểm với một lần chạy mới.
