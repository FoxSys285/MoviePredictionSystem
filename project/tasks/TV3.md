# TV3 — SVD, KNN, giảm chiều và phân cụm

**Thành viên:** ................................  
**Thời gian:** 4 tuần  
**Quy trình:** nhận tập chia sau bước 1–2; thực hiện 3. Huấn luyện → 4. Dự đoán → 5. Đánh giá.

Đánh dấu `[x]` sau khi **file đã có, phương pháp được mô tả đúng và metric kiểm tra lại được**. Mọi đường dẫn tính từ `project/`. Chỉ dùng tương tác quan sát trong train để xây ma trận user–item.

## Tuần 1 — Ma trận thưa và KNN baseline

### Việc cần làm

- [ ] Nạp train từ TV1, tạo ánh xạ `userId/movieId` sang chỉ số nội bộ; lưu số user, số item và số rating.
- [ ] Tạo ma trận tương tác dạng thưa; kiểm tra rating chưa quan sát **không bị hiểu là 0 sao**.
- [ ] Cài KNN item–item hoặc user–user với số láng giềng giới hạn; xác định fallback khi user/phim chưa có trong train.
- [ ] Thống nhất với TV5 giao diện trả điểm rating và danh sách phim ứng viên.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `src/factorization/matrix.py`, `knn.py` | Tạo ma trận thưa, ánh xạ ID và KNN chạy thử trên mẫu nhỏ |
| `results/matrix_profile.json` | Số user/item/rating, độ thưa, ước lượng bộ nhớ và ngưỡng lọc |
| `report/tv3_factorization.md` | Bản nháp cách tạo ma trận và xử lý rating thiếu |

- [ ] **Nghiệm thu tuần 1:** không gọi chuyển toàn ma trận 25M rating sang mảng đặc; kết quả ánh xạ ID có thể tái tạo.

## Tuần 2 — SVD, giảm chiều và dự đoán validation

### Việc cần làm

- [ ] Huấn luyện SVD/matrix factorization trên rating đã quan sát; thử ít nhất một số chiều latent và regularization.
- [ ] Xuất dự đoán rating của SVD và KNN trên cùng tập validation; tính RMSE/MAE, thời gian train/suy luận.
- [ ] Với ma trận thưa, dùng TruncatedSVD để khảo sát giảm chiều; chỉ dùng PCA nếu dữ liệu biểu diễn dạng đặc phù hợp. Phân biệt phép giảm chiều với SVD dự đoán rating.
- [ ] Đo reconstruction error và tỷ lệ phương sai giải thích khi phương pháp hỗ trợ; vẽ biểu đồ theo số chiều.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `src/factorization/svd.py`, `reduction.py` | Huấn luyện và dự đoán được; lưu cấu hình thử |
| `results/predictions_svd_validation.csv`, `predictions_knn_validation.csv` | Đủ `row_id` và cùng tập validation |
| `results/metrics_svd_validation.csv`, `metrics_knn_validation.csv`, `reduction_validation.csv` | Có chỉ số và số mẫu, nêu rõ reconstruction đo trên gì |
| `figures/variance_by_dimension.png`, `report/tv3_factorization.md` | Biểu đồ có trục/đơn vị và nhận xét ban đầu |

- [ ] **Nghiệm thu tuần 2:** metric tính lại từ file dự đoán khớp; không dùng ma trận điền 0 như thể đó là rating quan sát.

## Tuần 3 — Clustering, tuning và năm fold

### Việc cần làm

- [ ] Tìm tham số SVD trên train, chạy cùng năm fold tiến theo thời gian do TV1 tạo; báo cáo RMSE/MAE `mean ± std`.
- [ ] Lấy embedding/vector học từ train để chạy K-Means và DBSCAN; chuẩn hóa vector nếu cần.
- [ ] Đo Silhouette, Davies–Bouldin trên cùng tập vector hoặc mẫu cố định; với DBSCAN ghi số cụm và tỷ lệ điểm nhiễu.
- [ ] Lưu model SVD, ánh xạ ID, vector phim và hàm suy luận; bàn giao TV5.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `src/factorization/clustering.py`, `predict.py`, `tune.py` | Chạy được clustering, tuning và suy luận theo thứ tự ứng viên |
| `results/metrics_svd_cv.csv`, `tuning_svd.csv`, `clustering_validation.csv` | Có fold, cấu hình, metric, số cụm/nhiễu và thống kê tổng hợp |
| `artifacts/svd_model.pkl`, `svd_id_maps.json`, `item_vectors.npz` | Nạp lại được model, ID mapping và vector phim |
| `figures/movie_clusters.png`, `report/tv3_factorization.md` | Hình minh họa và mô tả cách diễn giải cụm |

- [ ] **Nghiệm thu tuần 3:** TV5 nạp model, dự đoán được phim ứng viên; nhóm khóa tham số SVD trước khi mở test.

## Tuần 4 — Test cuối và báo cáo

### Việc cần làm

- [ ] Dự đoán SVD/KNN trên test theo cấu hình đã khóa; tính RMSE/MAE và thời gian suy luận.
- [ ] Đối chiếu SVD với KNN và baseline điểm trung bình; ghi rõ phạm vi warm-start và fallback.
- [ ] Kiểm tra kết quả giảm chiều/clustering trong báo cáo đúng với train/validation, không tuyên bố phân cụm là kết quả dự đoán rating.
- [ ] Duyệt bảng số liệu của TV5 và hoàn thiện mục SVD, giảm chiều, phân cụm, giới hạn tài nguyên.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `results/predictions_svd_test.csv`, `predictions_knn_test.csv` | Cùng tập test, đúng `row_id` và schema |
| `results/metrics_svd_test.csv`, `metrics_knn_test.csv`, `latency_factorization.csv` | Có RMSE/MAE, thời gian và phần cứng |
| `report/tv3_factorization.md` | Có phương pháp, tham số, bảng/biểu đồ, nhận xét và giới hạn |

- [ ] **Nghiệm thu tuần 4/hoàn thành TV3:** SVD có kết quả kiểm chứng và test trong bảng rating; KNN là baseline rõ ràng; các thí nghiệm giảm chiều/phân cụm có chỉ số riêng.

### Mở rộng nếu phần bắt buộc đã xong

- [ ] Nếu nhóm làm Q-Learning: cài hai baseline ngẫu nhiên/phim phổ biến trong `src/factorization/rl_baselines.py`, xuất `results/rl_baselines.csv`.
