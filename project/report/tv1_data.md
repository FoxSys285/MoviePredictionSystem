# Phần TV1 — Dữ liệu, chia tập và baseline

> Số liệu được tạo từ toàn bộ MovieLens 25M trên máy hiện tại. Nguồn kiểm chứng: `results/data_profile.json`, `data/processed/splits.json`, `results/coverage_validation.csv`, `results/coverage_cv.csv`, `results/metrics_mean_validation.csv`, `results/metrics_mean_cv.csv` và `results/data_audit.md`. Tập test chưa được dùng để đo chất lượng mô hình.

## 1. Nguồn và chất lượng dữ liệu

Nhóm sử dụng [MovieLens 25M của GroupLens](https://grouplens.org/datasets/movielens/25m/), lấy `ratings.csv` làm bảng tương tác chính và `movies.csv` làm bảng thông tin phim. Mỗi rating gồm `userId`, `movieId`, `rating` và `timestamp`. Dữ liệu gốc được giữ tại `../Dataset/ml-25m`; các file xử lý nằm trong `project/data/processed`. [README của GroupLens](https://files.grouplens.org/datasets/movielens/ml-25m-README.html) yêu cầu dẫn nguồn và không phân phối lại bộ dữ liệu nếu chưa được cho phép.

| Chỉ tiêu kiểm tra | Kết quả |
|---|---:|
| Số rating | 25.000.095 |
| Số người dùng có rating | 162.541 |
| Số phim trong `movies.csv` | 62.423 |
| Số phim xuất hiện trong `ratings.csv` | 59.047 |
| Rating thiếu, ID không hợp lệ, rating ngoài thang hoặc timestamp không hợp lệ | 0 |
| Bản ghi trùng hoàn toàn trên `(userId, movieId, rating, timestamp)` | 0 |
| Rating nhỏ nhất/lớn nhất | 0,5 / 5,0 |

Các rating được chấm theo 10 mức cách nhau 0,5 sao. Mức 4,0 sao xuất hiện nhiều nhất (6.639.798 lượt). Trung vị số rating/người dùng là 71, còn trung vị số rating/phim đã được đánh giá là 6. Tỷ lệ cặp user–item có rating trong tập quan sát khoảng **0,26%**, cho thấy ma trận tương tác rất thưa. Các hình `figures/rating_distribution.png`, `interaction_distribution.png` và `genre_distribution.png` minh họa phân bố tương ứng.

Script `src/data/prepare.py` kiểm tra file nguồn, giá trị thiếu/không hợp lệ và trùng lặp; gán `row_id` ổn định sau khi sắp xếp theo `(timestamp, userId, movieId, rating)` và lưu `data/processed/ratings_clean.parquet`. Mọi ô user–item chưa có rating được coi là **chưa quan sát**, không được gán điểm 0. Hash SHA-256 của hai file nguồn được lưu trong `results/data_profile.json` để hỗ trợ chạy lại.

## 2. Chia dữ liệu và xử lý tương tác ít

Sau khi làm sạch, nhóm chia toàn bộ rating theo **thời gian toàn cục**. Các rating trùng một giây ở ranh giới được đặt cùng một phía; do đó tỷ lệ thực tế xấp xỉ 70/15/15.

| Tập | Số rating | Khoảng thời gian UTC |
|---|---:|---|
| Train | 17.500.066 | 09/01/1995 – 22/01/2015 |
| Validation | 3.750.014 | 22/01/2015 – 24/03/2017 |
| Test | 3.750.015 | 24/03/2017 – 21/11/2019 |

Ba tập có cùng schema `row_id, userId, movieId, rating, timestamp` và không giao nhau. `data/processed/splits.json` lưu số dòng/range ID/range timestamp; `cv_folds.json` lưu năm fold tiến theo thời gian bên trong train. Mỗi fold có giai đoạn huấn luyện mở rộng dần và giai đoạn kiểm chứng nằm ngay sau đó. Các model dùng chung những file này để so sánh.

Ngưỡng thử nghiệm để nhận diện người dùng/phim ít tương tác là **ít nhất 10 rating/người dùng và 20 rating/phim trong train**. Ngưỡng chỉ được tính từ train. File `train_core.parquet` là tập train phụ gồm 17.444.718 rating (99,68% rating train), 121.946 người dùng và 12.354 phim. **Tập train đầy đủ vẫn là tập huấn luyện chính**; `train_core.parquet` chỉ dành cho phân tích warm-start hoặc thí nghiệm có ghi rõ. Validation/test không bị xóa các trường hợp cold-start.

Một hệ quả quan trọng của cách chia theo thời gian toàn cục là **3.317.850/3.750.014 rating validation (88,48%) thuộc người dùng chưa có rating nào trong train**. Chỉ 337.481 rating validation (9,00%) thuộc nhóm warm-start theo cả hai ngưỡng trên. Vì SVD và NCF dùng embedding ID không dự đoán trực tiếp được cho user chưa xuất hiện, nhóm cần báo cáo riêng kết quả trên cùng tập warm-start và kết quả toàn validation khi có fallback. Không so sánh điểm toàn tập của các mô hình nếu mỗi mô hình dùng một tập hàng khác nhau. Tập chung `data/processed/validation_warm.parquet` và file `cv_warm_ids.parquet` (cột `fold,row_id`) giúp TV2–TV4 dùng **đúng cùng hàng** khi đánh giá warm-start; `results/coverage_validation.csv` và `coverage_cv.csv` lưu số lượng của từng nhóm.

## 3. Baseline điểm trung bình

Baseline dự đoán mọi cặp user–movie bằng rating trung bình của train, **3,52548945** sao. Đây là mốc đối chứng tối thiểu, không dùng validation/test để học tham số. Mô hình tạo dự đoán cho toàn bộ validation trong `results/predictions_mean_validation.csv`.

| Phạm vi đánh giá | Số rating | RMSE | MAE |
|---|---:|---:|---:|
| Validation toàn bộ | 3.750.014 | 1,091591 | 0,848723 |
| Validation warm-start | 337.481 | 1,033582 | 0,781742 |
| 5 fold tiến theo thời gian trong train, mean ± std | Mỗi fold khoảng 2,92 triệu rating kiểm chứng | 1,054701 ± 0,045710 | 0,834524 ± 0,067250 |
| 5 fold, chỉ warm-start, mean ± std | 139.068–358.189 rating/fold | 1,019768 ± 0,065188 | 0,800120 ± 0,084263 |

Thời gian huấn luyện là thời gian tính trung bình train; độ trễ suy luận được đo bằng thời gian đọc và trả về một batch 100.000 dự đoán trên cùng máy. File metric ghi số dòng, seed, phần cứng và thời gian cụ thể. Baseline không được dùng để suy luận rằng các mô hình phức tạp sẽ chắc chắn vượt trội; kết luận chỉ được rút ra sau khi nhóm chạy cùng giao thức.

## 4. Gợi ý ban đầu cho người dùng mới

`src/data/cold_start.py` cung cấp phương án gợi ý cho user chưa có embedding. Hàm nhận các phim và rating mà người dùng vừa nhập, tính sở thích thể loại từ các rating đó, kết hợp với rating trung bình đã làm trơn và độ phổ biến của phim **chỉ tính từ train**. Các phim người dùng đã chọn bị loại khỏi Top-N. Nếu người dùng chưa nhập phim, hàm sử dụng rating đã làm trơn và độ phổ biến. Kết quả ghi rõ chiến lược `genre_popularity_cold_start` hoặc `popularity_cold_start`, để web không nhầm fallback với dự đoán SVD/NCF.

Chiến lược này phục vụ demo và chưa được tuyên bố là tối ưu. Phần đánh giá Top-N của TV5 cần định nghĩa riêng tập phim ứng viên và thước đo; các rating thiếu không được mặc nhiên coi là phản hồi tiêu cực.

## 5. Kiểm tra và giới hạn

`src/data/audit.py` đã kiểm tra 12 điều kiện: schema và ID duy nhất ở ba tập, ranh giới và thứ tự thời gian trên toàn bộ dữ liệu, tổng số hàng, thống kê user/movie train, năm fold, các tập warm-start chung và khả năng tính lại RMSE/MAE baseline từ file dự đoán. Tất cả đều **PASS** tại `results/data_audit.md`.

Môi trường đã dùng cho phần TV1: Python 3.14.3, DuckDB 1.5.5, pandas 3.0.5, PyArrow 25.0.1, NumPy 2.5.1 và Matplotlib 3.11.0; bộ nhớ DuckDB giới hạn 2 GB và dùng hai luồng theo `configs/default.json`.

Những việc cần cả nhóm hoàn tất sau phần TV1: TV2–TV4 xác nhận nạp đúng dữ liệu và để TV1 soát đầu ra validation; TV5 xác nhận tích hợp hàm cold-start; nhóm chốt mô hình từ validation trước khi TV1 chạy baseline trên test. Vì vậy, trong bản hiện tại **chưa có metric test**.

## 6. Cách chạy lại

Từ thư mục `project`, chạy lần lượt:

```powershell
.\.venv\Scripts\python.exe -m src.data.prepare
.\.venv\Scripts\python.exe -m src.data.split
.\.venv\Scripts\python.exe -m src.data.eda
.\.venv\Scripts\python.exe -m src.data.baseline --split validation
.\.venv\Scripts\python.exe -m src.data.baseline --cv
.\.venv\Scripts\python.exe -m src.data.audit
.\.venv\Scripts\python.exe -m src.data.cold_start --ratings 1:5 296:4.5 --top-n 10
```

Các file Parquet và dự đoán đầy đủ được Git bỏ qua vì dung lượng lớn. Khi chạy lại trên cùng dữ liệu và cấu hình, dùng `--force` cho `prepare`/`split` nếu muốn ghi đè file xử lý đã có. Tập test chỉ được đánh giá sau khi nhóm tạo `report/model_selection.md` và chạy baseline với cờ `--allow-test`.
