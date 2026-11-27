# Cài đặt thuật toán thực nghiệm

Mỗi thư mục thuật toán có `baseline.py` cho mô hình cơ sở và `improved.py`
cho mô hình cải tiến. Hiện các lớp là khung: `fit()` và `predict()` báo
`NotImplementedError` cho tới khi cài đặt từng thuật toán.

| Thư mục | Thư viện sử dụng |
|---|---|
| `algorithms/linear_regression/` | scikit-learn |
| `algorithms/xgboost/` | XGBoost |
| `algorithms/svd/` | NumPy, Numba; SciPy cho dữ liệu sparse khi cần |
| `algorithms/ncf/` | PyTorch |

Pandas dùng để giữ các bảng rating và thông tin phim. Các phụ thuộc đã được khai
báo trong `project/pyproject.toml`; phiên bản cài đặt được khoá trong các file
`requirements-lock.txt` và `requirements-torch-*.txt`. Dùng môi trường `project/.venv`.

## Giao diện cài đặt

- `fit(ratings, movies=None)`: học từ rating Train và thông tin phim nếu cần.
- `predict(pairs)`: nhận bảng có `userId`, `movieId`; trả một mảng điểm theo đúng
  thứ tự các dòng đầu vào.
- `registry.py`: chọn bằng ID `linear`, `xgboost`, `svd`, `ncf` và phiên bản
  `baseline` hoặc `improved`.
- `data.py`, `runner.py`: khung để bổ sung xử lý dữ liệu và điều phối chạy sau.

## Hai chỉ số đánh giá

`evaluation.py` đã cài đặt:

- `rmse(actual, predicted)`: căn bậc hai của trung bình bình phương sai số.
- `mae(actual, predicted)`: trung bình trị tuyệt đối sai số.
- `evaluate_predictions(actual, predicted)`: trả cả hai trong một dictionary.

Cả hai tính theo đơn vị sao, càng thấp càng tốt. Đầu vào phải là hai mảng một
chiều, không rỗng, cùng số phần tử, không chứa NaN/Inf. Các hàm đo trực tiếp
điểm được truyền vào; việc giới hạn dự đoán về thang sao sẽ được chốt ở bước
triển khai mô hình.

```python
from movie_rating.experiments.evaluation import evaluate_predictions

scores = evaluate_predictions([3.0, 4.0, 5.0], [2.0, 4.0, 4.0])
# {"rmse": 0.816496580927726, "mae": 0.6666666666666666}
```

Giao diện thực nghiệm chưa gọi các lớp này để huấn luyện. Việc nối API và chạy
mô hình sẽ thực hiện sau khi có implementation.
