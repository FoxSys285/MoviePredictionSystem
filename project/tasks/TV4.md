# TV4 — Neural Collaborative Filtering (NCF)

**Thành viên:** ................................  
**Thời gian:** 4 tuần  
**Quy trình:** nhận tập chia sau bước 1–2; thực hiện 3. Huấn luyện → 4. Dự đoán → 5. Đánh giá.

Đánh dấu `[x]` sau khi **file đã có, checkpoint nạp lại được và metric có thể tính lại**. Mọi đường dẫn tính từ `project/`. NCF trong đồ án dự đoán rating tường minh; không tự tạo nhãn âm từ các phim chưa được đánh giá.

## Tuần 1 — Ánh xạ ID và khung mô hình

### Việc cần làm

- [ ] Nhận train/validation từ TV1; tạo ánh xạ user/item **chỉ từ train** và quy tắc ID chưa thấy.
- [ ] Tạo DataLoader đưa ra `(user_index, movie_index, rating)`; xác nhận rating và chỉ số embedding hợp lệ.
- [ ] Cài embedding user/item + MLP + đầu ra rating; chọn loss MSE/MAE và cách giới hạn/hoàn nguyên thang điểm.
- [ ] Chạy một batch thử, ghi seed, thiết bị CPU/GPU và cấu hình đầu tiên.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `src/ncf/dataset.py`, `model.py` | Ánh xạ ID và forward pass chạy được; không truy cập embedding ngoài phạm vi |
| `src/ncf/train.py` | Vòng lặp train/validation cơ bản và lưu checkpoint |
| `data/processed/ncf_id_maps.json` | Ánh xạ chỉ từ train, dùng lại được khi nạp model |
| `report/tv4_ncf.md` | Bản nháp kiến trúc, đầu vào/đầu ra và loss |

- [ ] **Nghiệm thu tuần 1:** một batch qua mô hình không lỗi, loss hữu hạn; TV1 xác nhận ánh xạ không được học từ validation/test.

## Tuần 2 — Huấn luyện đầu tiên và learning curve

### Việc cần làm

- [ ] Chạy thử trên tập nhỏ, kiểm tra loss train giảm và kết quả checkpoint nạp lại khớp.
- [ ] Huấn luyện trên dữ liệu chung trong giới hạn tài nguyên; ghi train/validation loss mỗi epoch và dùng early stopping.
- [ ] Dự đoán validation, tính RMSE/MAE, thời gian train/suy luận; vẽ learning curve và mô tả dấu hiệu overfitting.
- [ ] Bàn giao định dạng đầu vào/đầu ra của NCF cho TV5, đặc biệt cách xử lý user mới.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `artifacts/ncf_best.pt`, `ncf_id_maps.json` | Checkpoint và ID map nạp lại cho cùng dự đoán |
| `results/ncf_history.csv`, `metrics_ncf_validation.csv`, `predictions_ncf_validation.csv` | Có loss từng epoch, metric và cột dự đoán chuẩn |
| `figures/ncf_learning_curve.png`, `report/tv4_ncf.md` | Biểu đồ train/validation loss và nhận xét |

- [ ] **Nghiệm thu tuần 2:** RMSE/MAE từ file dự đoán khớp bảng metric; điểm dự đoán được đưa về thang rating 0,5–5 khi đánh giá.

## Tuần 3 — Optuna, năm fold và hàm suy luận

### Việc cần làm

- [ ] Chốt ngân sách Optuna trước khi chạy; thử embedding dimension, số tầng/kích thước MLP, learning rate, regularization và batch size.
- [ ] Lưu tất cả trial, seed, số epoch, thời gian, score; chọn cấu hình từ train/validation.
- [ ] Huấn luyện NCF từ đầu trên cùng năm fold tiến theo thời gian; tính RMSE/MAE `mean ± std`.
- [ ] Đóng gói hàm dự đoán cho user/movie đã biết và trả thông báo/fallback rõ ràng với ID chưa biết; bàn giao TV5.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `src/ncf/tune.py`, `predict.py` | Tuning có seed; suy luận theo đúng thứ tự ứng viên |
| `results/optuna_ncf_trials.csv`, `metrics_ncf_cv.csv` | Có đủ trial và năm fold, thời gian, `mean ± std` |
| `configs/ncf_best.json`, `artifacts/ncf_best.pt` | Cấu hình và checkpoint cuối dùng cho validation/web |
| `report/tv4_ncf.md` | Nêu lựa chọn tham số và phân tích learning curve |

- [ ] **Nghiệm thu tuần 3:** TV5 nạp được model hoặc nhận rõ giới hạn user mới; nhóm khóa cấu hình trước khi xem test.

## Tuần 4 — Test cuối và so sánh chi phí

### Việc cần làm

- [ ] Dự đoán test một lần bằng cấu hình đã khóa; tính RMSE/MAE và so sánh với Linear/XGBoost/SVD.
- [ ] Đo thời gian train, suy luận và dung lượng checkpoint; ghi phần cứng, batch size, số phim ứng viên.
- [ ] Kiểm tra learning curve đã dùng đúng tập validation, không dùng test để chọn epoch.
- [ ] Duyệt bảng chung của TV5; hoàn thiện báo cáo NCF và phần trình bày.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `results/predictions_ncf_test.csv`, `metrics_ncf_test.csv` | Đủ test, đúng cột và RMSE/MAE tính lại được |
| `results/latency_ncf.csv`, `report/tv4_ncf.md` | Có thời gian, phần cứng, mô hình, nhận xét độ chính xác/chi phí |

- [ ] **Nghiệm thu tuần 4/hoàn thành TV4:** NCF có năm fold, test một lần, learning curve, log Optuna và model nạp lại được; số liệu nằm trong bảng so sánh rating.

### Mở rộng nếu phần bắt buộc đã xong

- [ ] Nếu nhóm làm Q-Learning: cài huấn luyện trong `src/ncf/q_learning.py`, lưu `results/rl_training.csv`; ghi rõ reward chỉ đến từ môi trường mô phỏng.
