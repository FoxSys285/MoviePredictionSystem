# MovieLens AI: chạy web và mô hình

## Chạy lần đầu

Chạy các lệnh từ thư mục `project`:

```powershell
npm install
python -m pip install -r requirements-ml.txt
npm run train:models
```

Huấn luyện đọc các split Parquet trong `data/processed/movielens25m/`, giữ dữ liệu gốc nguyên vẹn, rồi lưu model vào `artifacts/models/` và chỉ số vào `results/model_metrics.csv` cùng `results/model_metrics.json`.

Mở ba terminal riêng:

```powershell
npm run api
```

```powershell
npm run dev
```

Sau đó mở URL Vite in ra ở terminal. API chạy ở `http://127.0.0.1:8000`; Vite chuyển tiếp các request `/api` đến API này. Nếu huấn luyện lại, dừng rồi khởi động API lại để nạp các artifact mới.

## Chức năng

- Đăng nhập demo bằng MovieLens User ID `1` đến `5`; các hồ sơ này dùng lịch sử rating thật của user tương ứng trong tập train.
- Tài khoản đăng ký mới có thể thêm phim đã xem, rating từ 0,5 đến 5 sao và thời điểm đánh giá. Lịch sử demo lưu trong `localStorage` của trình duyệt, chưa có xác thực hay cơ sở dữ liệu tài khoản.
- Trang **Gợi ý phim** gọi API để dự đoán Top 10 bằng mô hình đang chọn; phim có trong lịch sử bị loại khỏi kết quả.
- Trang **So sánh mô hình** chạy bốn mô hình trên cùng hồ sơ, hiển thị RMSE/MAE trên test, thời gian huấn luyện, tốc độ suy luận và danh sách gợi ý của từng mô hình. Có thêm baseline trung bình rating theo phim.

## Mô hình và dữ liệu đầu vào

| Mô hình | Cài đặt hiện tại | Đặc trưng dùng khi gợi ý |
| --- | --- | --- |
| Linear Regression | Ridge từ scikit-learn | Rating/hồ sơ user, rating và số lượt của phim, thể loại, thời gian |
| XGBoost | XGBoost Regressor | Cùng bộ đặc trưng hồ sơ, thể loại và thời gian |
| SVD / Matrix Factorization | Latent factors user-item huấn luyện bằng PyTorch | Rating user-item và timestamp; lịch sử nhập trên web được dùng để fold-in, thể loại dùng để điều chỉnh hồ sơ |
| NCF | Neural Collaborative Filtering bằng PyTorch | Embedding user-item và timestamp; lịch sử nhập trên web được dùng để fold-in, thể loại dùng để điều chỉnh hồ sơ |

Khi train, Linear lấy tối đa 1.500.000 rating, XGBoost 600.000, Matrix Factorization 1.500.000 và NCF 600.000. Seed mặc định là `42`. Các số validation/test được tính trên các split thời gian đã xử lý, không trộn ngẫu nhiên các split.

Kết quả lần huấn luyện hiện tại:

| Mô hình | RMSE test | MAE test | Huấn luyện |
| --- | ---: | ---: | ---: |
| Linear Regression | 0,8878 | 0,6745 | 0,67 giây |
| XGBoost | **0,8677** | **0,6564** | 5,13 giây |
| SVD / Matrix Factorization | 1,0875 | 0,8370 | 18,25 giây |
| NCF | 0,9618 | 0,7367 | 8,32 giây |
| Baseline trung bình phim | 0,9566 | 0,7366 | — |

Vì kết quả này, XGBoost được chọn sẵn trên trang gợi ý; người dùng vẫn có thể đổi mô hình. Thời gian huấn luyện phụ thuộc phần cứng. Các chỉ số test đánh giá dự đoán trên holdout với latent factors đã học; bảng gợi ý trên web dùng thêm fold-in theo lịch sử hiện tại.

## Lệnh hữu ích

```powershell
npm run train:models -- --help
```

Các tham số như số rating lấy mẫu, số epoch và seed có thể đặt sau lệnh huấn luyện, ví dụ:

```powershell
npm run train:models -- --seed 42 --linear-rows 1500000 --xgboost-rows 600000 --svd-epochs 5 --ncf-epochs 6
```

Thay đổi model code rồi chạy lại huấn luyện trước khi bật API để cập nhật artifact.
