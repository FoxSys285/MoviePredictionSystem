# Biểu đồ và độ ổn định hồi quy

Experiment Studio hiện chỉ hiển thị biểu đồ và kết quả của các mô hình hồi quy rating. Đọc các đường hoặc cột cùng split, cấu hình, baseline và log của cùng một run.

GridSearchCV báo RMSE và MAE qua năm fold trên Train. Độ lệch giữa các fold cho biết kết quả thay đổi thế nào theo các phần dữ liệu; nó không thay thế đánh giá cuối trên Test.

Khi so sánh hai lần chạy, giữ nguyên cách chia dữ liệu và xem cả Validation, Test, thời gian fit cùng log. Một run có RMSE tốt hơn nhưng cấu hình hoặc dữ liệu khác không phải là so sánh trực tiếp.
