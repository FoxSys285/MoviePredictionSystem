# TV2 — Regression và classification

**Thành viên:** ................................  
**Thời gian:** 4 tuần  
**Quy trình:** nhận dữ liệu sau bước 1–2; thực hiện 3. Huấn luyện → 4. Dự đoán → 5. Đánh giá.

Đánh dấu `[x]` khi **file đã có, chạy/đọc kiểm tra được và số liệu đã đối chiếu**. Mọi đường dẫn tính từ `project/`. Dùng đúng tập chia do TV1 bàn giao, không tự chia lại.

## Tuần 1 — Chuẩn bị đặc trưng và pipeline chạy thử

### Việc cần làm

- [ ] Đọc schema của TV1, kiểm tra `row_id` và số dòng; xác định nhãn rating cho hồi quy và nhãn `rating >= 4` cho phân loại.
- [ ] Thiết kế đặc trưng user/movie/genre/year có thể tính lúc suy luận. Trung bình rating hoặc số lượt tương tác của user/item trong hàng train phải loại chính rating của hàng đó hoặc chỉ dùng lịch sử trước hàng đó.
- [ ] Tạo pipeline Linear và XGBoost chạy thử trên tập con nhỏ; ghi seed và đặc trưng dùng. Không xem kết quả test.
- [ ] Thống nhất với TV5 hàm suy luận nhận user, danh sách phim ứng viên và trả điểm theo đúng thứ tự.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `src/classical/features.py` | Biến đổi train/validation nhất quán; không dùng nhãn tương lai để tạo đặc trưng |
| `src/classical/linear.py`, `xgboost_model.py` | Có mô hình thử chạy trên tập nhỏ và lưu được cấu hình |
| `report/tv2_models.md` | Bản nháp mô tả nhãn, đặc trưng và lý do chọn hai mô hình |

- [ ] **Nghiệm thu tuần 1:** pipeline chạy thử không lỗi; TV1 xác nhận không rò rỉ rating trong đặc trưng lịch sử.

## Tuần 2 — Huấn luyện lần đầu, dự đoán validation

### Việc cần làm

- [ ] Huấn luyện Linear và XGBoost trên train chung; xuất rating dự đoán validation, RMSE/MAE và thời gian.
- [ ] Huấn luyện Logistic Regression và Random Forest với quy ước thích/không thích đã chốt; lưu nhãn và xác suất lớp thích.
- [ ] Tính Accuracy, F1, AUC-ROC; ghi tỷ lệ hai lớp để giải thích số liệu.
- [ ] So sánh Linear/XGBoost với baseline của TV1; ghi các trường hợp dự đoán sai nhiều.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `src/classical/classification.py`, `train.py` | Chạy được bốn mô hình và lưu kết quả |
| `results/predictions_linear_validation.csv`, `predictions_xgboost_validation.csv` | Đủ `row_id` và các cột dự đoán rating chuẩn |
| `results/predictions_logistic_validation.csv`, `predictions_random_forest_validation.csv` | Có `row_id`, nhãn thật, nhãn dự đoán và xác suất lớp thích để tính lại AUC-ROC |
| `results/metrics_linear_validation.csv`, `metrics_xgboost_validation.csv`, `metrics_classification_validation.csv` | Có RMSE/MAE hoặc Accuracy/F1/AUC-ROC cùng số mẫu đo |
| `report/tv2_models.md` | Bổ sung bảng kết quả validation lần đầu |

- [ ] **Nghiệm thu tuần 2:** tính lại RMSE/MAE từ các file dự đoán cho ra đúng số đã báo cáo; xác suất phân loại nằm trong `[0,1]`.

## Tuần 3 — Tìm tham số, năm fold và đóng gói suy luận

### Việc cần làm

- [ ] Dùng GridSearchCV/RandomizedSearchCV với chia tiến theo thời gian trong train; lưu số cấu hình thử và metric từng lần.
- [ ] Chạy Linear/XGBoost trên cùng năm fold do TV1 cấp; tính `mean ± std` cho RMSE/MAE.
- [ ] Huấn luyện lại cấu hình cuối trên train; chỉ dùng validation để quyết định mô hình triển khai.
- [ ] Đóng gói scaler/encoder cùng model, tạo hàm dự đoán cho danh sách phim ứng viên; bàn giao TV5.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `src/classical/tune.py`, `predict.py` | Tuning có seed và suy luận trả đúng thứ tự ứng viên |
| `results/tuning_classical.csv`, `metrics_linear_cv.csv`, `metrics_xgboost_cv.csv` | Có cấu hình, fold, metric, thời gian và thống kê tổng hợp |
| `artifacts/linear.joblib`, `xgboost_model.json`, `logistic.joblib`, `random_forest.joblib`, `classical_features.joblib` | Nạp lại được model và đúng bộ biến đổi đặc trưng đi kèm |
| `report/tv2_models.md` | Nhận xét mô hình nào vượt baseline và chi phí của từng mô hình |

- [ ] **Nghiệm thu tuần 3:** TV5 nạp được một model và gọi hàm dự đoán; nhóm đã khóa tham số trước khi xem test.

## Tuần 4 — Dự đoán, đánh giá test và báo cáo

### Việc cần làm

- [ ] Dự đoán trên test một lần bằng cấu hình đã khóa; tính RMSE/MAE cho hồi quy và Accuracy/F1/AUC-ROC cho phân loại.
- [ ] Đo train time và inference time trên cùng quy mô mẫu/hardware đã thống nhất với nhóm.
- [ ] Xác nhận mọi số trong bảng chung của TV5 khớp file metric; viết nhận xét sai số và giới hạn của mô hình.
- [ ] Chuẩn bị phần trình bày và cách trả lời vì sao XGBoost/Linear phù hợp hoặc không phù hợp để triển khai.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `results/predictions_linear_test.csv`, `predictions_xgboost_test.csv` | Có cùng tập test và đúng schema chuẩn |
| `results/predictions_logistic_test.csv`, `predictions_random_forest_test.csv` | Có xác suất lớp thích và nhãn để kiểm tra chỉ số phân loại |
| `results/metrics_linear_test.csv`, `metrics_xgboost_test.csv`, `metrics_classification_test.csv` | Kết quả test một lần, ghi số mẫu đo |
| `results/latency_classical.csv`, `report/tv2_models.md` | Có đơn vị thời gian, phần cứng, phương pháp và kết luận |

- [ ] **Nghiệm thu tuần 4/hoàn thành TV2:** cả bốn mô hình có mã và số liệu kiểm chứng được; Linear/XGBoost hiện diện trong bảng so sánh rating; TV5 dùng được hàm suy luận nếu một trong hai được chọn.

### Mở rộng nếu phần bắt buộc đã xong

- [ ] Nếu nhóm làm Q-Learning: viết mô tả trạng thái, hành động, reward và cập nhật Q trong `report/rl_design.md`; không dùng phản hồi test để thiết kế reward.
