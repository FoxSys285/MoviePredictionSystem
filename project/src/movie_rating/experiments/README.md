# Cài đặt thuật toán thực nghiệm

Mỗi thư mục thuật toán có `baseline.py` cho mô hình cơ sở và `improved.py`
cho mô hình cải tiến. Cả tám lớp đã triển khai `fit()`, `predict()` và các hook
early stopping cần thiết cho bộ điều phối.

| Thư mục | Thư viện sử dụng |
|---|---|
| `algorithms/linear_regression/` | scikit-learn |
| `algorithms/xgboost/` | XGBoost |
| `algorithms/svd/` | NumPy, Numba; SciPy cho dữ liệu sparse khi cần |
| `algorithms/ncf/` | PyTorch |

| Thuật toán | Cơ sở | Cải tiến |
|---|---|---|
| Linear Regression | StandardScaler + OLS trên thống kê lịch sử | StandardScaler + Ridge L2, thêm thể loại/năm và tương tác sở thích |
| XGBoost | Histogram boosting trên thống kê lịch sử | Thêm đặc trưng nội dung/sở thích, subsampling và regularization |
| SVD | Global mean + bias user/phim + tích nhân tố ẩn, học SGD | Khởi tạo bias có regularization, nhiều factors hơn và learning-rate decay |
| NCF | Embedding user/phim + MLP, MSE/AdamW | Thêm nhánh tích embedding, MLP sâu hơn, dropout và khởi tạo bias |

Đánh giá hiệu quả từng biến thể bằng Validation/Test. Thống kê đặc trưng được
fit riêng trên từng fold Train; các hàng huấn luyện loại chính rating của mình
khỏi thống kê, kể cả global prior. ID được dùng làm mã tra cứu; SVD/NCF học trên
rating quan sát, không dựng ma trận user × movie đặc hoặc điền ô trống bằng 0.

Pandas dùng để giữ các bảng rating và thông tin phim. Các phụ thuộc đã được khai
báo trong `project/pyproject.toml`; phiên bản cài đặt được khoá trong các file
`requirements-lock.txt` và `requirements-torch-*.txt`. Dùng môi trường `project/.venv`.

## Giao diện cài đặt

- `fit(ratings, movies=None)`: học từ rating Train và thông tin phim nếu cần.
- `predict(pairs)`: nhận bảng có `userId`, `movieId`; trả một mảng điểm theo đúng
  thứ tự các dòng đầu vào.
- `registry.py`: chọn bằng ID `linear`, `xgboost`, `svd`, `ncf` và phiên bản
  `baseline` hoặc `improved`.
- `data.py`: đọc CSV, chia dữ liệu ngoài 70/15/15 và tạo 5 fold trên Train.
- `runner.py`: GridSearchCV, refit, đánh giá Validation và Test.
- `baseline.py`: Global Mean dùng chung để đối chiếu cả bốn thuật toán.

## Hai chỉ số đánh giá

`evaluation.py` đã cài đặt:

- `rmse(actual, predicted)`: căn bậc hai của trung bình bình phương sai số.
- `mae(actual, predicted)`: trung bình trị tuyệt đối sai số.
- `evaluate_predictions(actual, predicted)`: trả cả hai trong một dictionary.

Cả hai tính theo đơn vị sao, càng thấp càng tốt. Đầu vào phải là hai mảng một
chiều, không rỗng, cùng số phần tử, không chứa NaN/Inf. Các hàm đo trực tiếp
điểm được truyền vào. Các mô hình giới hạn dự đoán về 0,5–5; metric không tự
làm tròn hay thay đổi dự đoán.

```python
from movie_rating.experiments.evaluation import evaluate_predictions

scores = evaluate_predictions([3.0, 4.0, 5.0], [2.0, 4.0, 4.0])
# {"rmse": 0.816496580927726, "mae": 0.6666666666666666}
```

`config.py` cung cấp lưới tham số khởi đầu cho từng phiên bản. Có thể truyền
`parameter_grids` riêng vào `run_experiment()`. Scaler chỉ áp dụng cho các mô
hình tuyến tính. SVD/NCF có fallback từ global mean và bias khi gặp ID mới.

## Chia dữ liệu

`split_data(ratings)` trả `train`, `validation`, `test`, chia riêng trong từng
user theo thời gian 70/15/15. Ví dụ 100 rating khác timestamp được chia 70/15/15.
Giữ timestamp trùng trong một tập nên tỷ lệ có thể lệch. User dưới 3 rating hoặc
không có ba giai đoạn tách biệt bị loại và thống kê trong
`splits['train'].attrs['split_report']`, cùng số dòng và tỷ lệ thực tế mỗi tập.
Một cặp user/phim chỉ được xuất hiện một lần trong dữ liệu đầu vào.

`load_data(directory)` đọc `ratings.csv`, `movies.csv` và trả `(splits, movies)`.
Không ghi đè dữ liệu đã chuẩn bị hoặc tự chạy huấn luyện.

`make_cv_folds(train)` chia lịch sử Train thành sáu khối thời gian, tạo năm fold
mở rộng quá khứ để dự đoán khối tiếp theo. Timestamp trùng luôn cùng phía ranh
giới. User cần ít nhất 6 rating Train để tham gia CV; user không có khối đánh giá
ở một fold không đóng góp điểm cho fold đó. Nếu một fold không có dữ liệu đánh
giá thì báo lỗi. Fold chỉ chứa chỉ số dòng của Train, không chứa Validation/Test.

## Quy trình đã chốt

1. `GridSearchCV` dùng 5 fold trên Train, chấm MAE và RMSE, `refit='rmse'`.
   Cung cấp `parameter_grids` cho cả `linear`, `xgboost`, `svd`, `ncf`.
2. GridSearchCV tự refit cấu hình tốt nhất của mỗi thuật toán trên toàn bộ Train.
   Thống kê, scaler và encoder phải fit bên trong implementation từng fold.
3. Global Mean được fit từ Train. Validation so sánh RMSE/MAE của cả bốn mô hình
   và Global Mean. XGBoost/SVD/NCF triển khai `fit_with_validation()` để fit lại từ
   đầu với siêu tham số đã chọn, dùng Validation cho early stopping. Hook phải
   đặt `best_iterations_` bằng **số** epoch/cây, không phải chỉ số bắt đầu từ 0.
4. `refit_train_validation=False` mặc định. Nếu bật, refit cả bốn mô hình và
   Global Mean trên Train + Validation. Giữ nguyên siêu tham số; XGBoost/SVD/NCF dùng
   số cây/epoch từ bước 3 và không dùng Test để dừng sớm. Điểm Validation đã đo
   ở bước 3 được giữ nguyên, không tính lại bằng mô hình đã học Validation.
5. Đến cuối mới tính Test **một lần cho từng mô hình**, gồm cả bốn thuật toán và
   Global Mean. Trả RMSE và MAE; không chọn lại siêu tham số theo Test.

`run_experiment(splits, movies, parameter_grids, variant='baseline',
refit_train_validation=False)` thực hiện quy trình trên và trả kết quả cùng
siêu tham số, điểm CV, Validation, Test và phạm vi refit. Hàm chưa học ensemble.
## Chạy và xem kết quả

Trong `project`, chạy:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m movie_rating.experiments.run --max-users 1000 --variant both
```

`--max-users 0` chọn toàn bộ dữ liệu. Mẫu được lấy ngẫu nhiên theo user với seed
42 và giữ đầy đủ lịch sử trước khi chia. `--refit-85` bật bước refit cuối; mặc
định tắt. Lưới của cả hai biến thể được lưu trước khi đánh giá Test của bất kỳ
biến thể nào. Hai biến thể dùng chung chính xác các tập Train/Validation/Test.

Mỗi lượt lưu trong `artifacts/experiments/<run-id>/`: dữ liệu đã chia, provenance,
cấu hình, `job.json` và `result.json`. Có thể tái sử dụng bộ chia bằng
`--prepared-from <run-id>`. Kết quả ghi rõ số user/rating thực tế, số user bị loại
vì không tách được lịch sử, phạm vi mẫu/toàn bộ và hash của mẫu.

Chạy `run_project.bat`, mở **Thực nghiệm → So sánh và đánh giá kết quả thực nghiệm**
để đọc RMSE/MAE và tải JSON. Trong **Mô hình cơ sở** hoặc **Mô hình cải tiến**,
chọn một hay nhiều thuật toán, rồi nhấn **Chạy thực nghiệm** dưới phần quy trình.
Hai phần mô tả có nút thu gọn/mở rộng. Log, đồng hồ và tiến độ xuất hiện ngay
dưới nút chạy; chuyển mục hoặc tải lại trang vẫn theo dõi được lượt đã khởi chạy.
Web mặc định lấy tối đa 1.000 user với seed 42 và không refit 85%. Phạm vi mẫu
được hiển thị cạnh nút chạy; không phải kết quả trên toàn bộ 25 triệu rating.

`POST /api/v1/experiments/runs` nhận `variant`, danh sách `algorithms`,
`max_users` (10–5.000, mặc định 1.000), `seed` và `refit_85`.
Mỗi mục chỉ có một lượt đang chạy; gửi lại cùng cấu hình trả lượt hiện tại.
Worker chạy riêng, ghi `job.json`, `events.jsonl`, `stdout.log`, `stderr.log`.
`GET /runs/<id>/logs?cursor=0` (cùng prefix API) đọc tiếp bằng byte cursor;
`/runs/<id>/logs/download` tải log đầy đủ. Tiến độ tăng theo số lần fit CV đã
hoàn thành và các giai đoạn sau CV, không ước tính thời gian còn lại.
Bộ chia đã lưu được dùng lại nếu seed, số user và fingerprint nguồn vẫn khớp.
CLI hỗ trợ `--algorithms linear xgboost svd ncf` để giới hạn thuật toán cần chạy.

Tham khảo API chính thức: [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html),
[XGBoost và early stopping](https://xgboost.readthedocs.io/en/stable/python/python_api.html),
[PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html).
