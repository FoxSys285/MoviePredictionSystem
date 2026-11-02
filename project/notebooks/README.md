# Kế hoạch notebook

Chưa cần cài Jupyter để chạy pipeline/API. Các notebook sẽ bổ sung khi có dữ liệu:

1. `01_eda.ipynb`: số user/movie/rating, phân bố điểm, độ thưa, thời gian và long tail.
2. `02_validation_experiments.ipynb`: đọc kết quả validation để so sánh và chỉnh tham số.
3. `03_final_report.ipynb`: biểu đồ kết quả cuối, đọc artifacts mà không huấn luyện lại.

Logic dùng lại đặt trong `src/movie_rating`; notebook chỉ gọi các module và minh họa.
