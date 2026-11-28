### Thông tin bộ dữ liệu MovieLens 
- Lượt đánh giá: 25.000.095
- Người dùng: 162.541
- Phim trong danh mục: 62.423
- Thể loại: 19
- Thời gian: 9/1/1995 – 21/11/2019

### Các thuật toán
- Linear Regression
- XGBoost
- SVD
- NCF
Dự đoán rating (0.5 - 5.0) dữ trên tập dữ liệu MovieLen

### So sánh 2 tham số RMSE và MAE

### Chọn Baseline Global Mean giai đoạn đầu

### Chia dữ liệu
- Chia dữ liệu theo thời gian trên từng User
Ví dụ: User 1 có 100 rating, train: 70, valid: 15, test: 15 và dữ liệu được chia theo thời gian

1. **Trên train 70%:** chạy GridSearchCV 5-fold (MAE + RMSE, refit theo RMSE) để chọn siêu tham số tốt nhất cho từng thuật toán
2. **Refit** mỗi mô hình với siêu tham số tốt nhất trên toàn bộ train 70%.
3. **Trên val 15%:** tính RMSE/MAE của 4 mô hình (và baseline). Dùng val để early stopping, so sánh sơ bộ, và học trọng số ensemble nếu có.
4. **(Tuỳ chọn)** Gộp train + val (85%) để refit lần cuối với siêu tham số đã chốt. Nếu làm vậy thì số epoch/số cây phải lấy từ early stopping ở bước 3.
5. **Trên test 15%:** đánh giá một lần cho cả 4 mô hình, báo cáo MAE và RMSE.