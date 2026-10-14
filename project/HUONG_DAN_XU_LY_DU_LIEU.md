# MovieLens 25M — xử lý dữ liệu

## Chạy pipeline

Chạy từ thư mục `project`:

```powershell
python -m pip install -r requirements-data.txt
python scripts/prepare_movielens.py
```

Mặc định script đọc `Dataset/` và ghi vào `data/processed/movielens25m/`. Script không sửa CSV nguồn và sẽ dừng nếu thư mục output đã tồn tại. Muốn thử ngưỡng khác, chọn thư mục output mới:

```powershell
python scripts/prepare_movielens.py `
  --min-user-ratings 20 `
  --min-movie-ratings 20 `
  --train-ratio 0.70 `
  --validation-ratio 0.15 `
  --output-dir data/processed/movielens25m-min20
```

Có thể thêm `--skip-genome` để bỏ qua `genome-scores.csv` (15,5 triệu dòng). Mặc định script vẫn xử lý file này vì đây là nguồn đặc trưng nội dung hữu ích cho các mô hình sau.

## Quy tắc xử lý

1. Đọc `ratings.csv`, loại dòng thiếu hoặc sai miền giá trị; nếu trùng cặp user/phim thì giữ đánh giá có timestamp mới nhất.
2. Với mỗi user, sắp theo `timestamp` (nếu trùng thì theo `movieId`) rồi chia 70% train, 15% validation, 15% test.
3. Chạy lọc k-core lặp chỉ trên **train**: giữ user và phim có ít nhất 20 rating trong train. Sau đó bỏ khỏi validation/test những dòng thuộc user hoặc phim không còn trong train. Làm vậy tránh dùng tương tác tương lai để chọn tập train.
4. Giữ rating gốc; bổ sung `rating_norm_01 = (rating - 0.5) / 4.5` và `rating_centered = rating - user_mean_train`. Mean chỉ tính từ train để tránh rò rỉ validation/test.
5. Đánh chỉ số user/phim từ các ID có trong train và tạo CSR sparse matrix cho cả ba split. Ô 0 trong ma trận là “không có rating”, không phải rating bằng 0.
6. Tạo bảng thể loại, tags, links và genome scores cho phim có trong train. Tạo thêm `train_like_transactions.parquet` từ rating train ≥ 4 để dùng thử Apriori.

## Đầu ra chính

- `ratings_train.parquet`, `ratings_validation.parquet`, `ratings_test.parquet`: rating gốc, timestamp, split và hai cột chuẩn hóa.
- `user_item_train.npz`, `user_item_validation.npz`, `user_item_test.npz`: ma trận CSR; đọc bằng `scipy.sparse.load_npz`.
- `user_id_map.parquet`, `movie_id_map.parquet`: ánh xạ ID gốc sang chỉ số ma trận 0-based.
- `user_train_means.parquet`: mean rating train theo user.
- `movies.parquet`, `movie_genres.parquet`, `movie_tags.parquet`, `movie_links.parquet`, `genome_tags.parquet`, `genome_scores.parquet`: metadata/đặc trưng phim.
- `train_like_transactions.parquet`: danh sách movie ID được user đánh giá từ 4 sao trở lên, chỉ từ train.
- `processing_report.json`: số dòng bị loại, số tương tác mỗi split, số user/phim và cấu hình chạy.

Lưu ý: việc chia theo thời gian ở đây là theo thứ tự xem của từng user. Các mô hình phía sau cần dùng đúng ba split này; không gộp lại rồi random split, vì như vậy sẽ làm lộ đánh giá tương lai vào train. Cross-validation nếu cần nên thiết kế theo thời gian/user riêng, không áp k-fold ngẫu nhiên lên các dòng rating.
