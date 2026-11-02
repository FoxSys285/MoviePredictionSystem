# Hợp đồng API v1

Base URL khi chạy local: `http://127.0.0.1:8000/api/v1`.
Swagger `/docs` cung cấp schema và thử request trực tiếp.

| Method | Endpoint | Chức năng |
|---|---|---|
| GET | `/health` | Trạng thái API, dataset, dữ liệu đã chuẩn bị và mô hình |
| GET | `/models` | Bốn thuật toán chính và ba thuật toán tham chiếu, cùng trạng thái artifact |
| GET | `/dataset` | Thống kê toàn bộ CSV, phân bố điểm, thể loại và sáu phim phổ biến |
| GET | `/movies?q=&offset=0&limit=20` | Tìm, lọc và sắp xếp phim; phân trang tối đa 100 mục |
| GET | `/movies/{movie_id}` | Chi tiết phim, phân bố điểm, nhãn, IMDb/TMDB và phim cùng thể loại |
| GET | `/discovery/genres` | Tối đa 5 hàng thể loại, mỗi hàng tối đa 10 phim; tham số limit/movies_per_genre |
| GET | `/recommendations` | Tối đa 20 gợi ý; limit<=20, algorithm=auto hoặc mô hình đã train |
| GET | `/me/ratings` | Lịch sử của tài khoản đăng nhập; offset/limit, tối đa 100 mục/trang |
| GET | `/me/ratings/{movie_id}` | Điểm riêng của tài khoản; rating=null nếu chưa chấm |
| PUT | `/me/ratings/{movie_id}` | Tạo/cập nhật điểm riêng 0,5–5, bước 0,5; phim phải có trong catalogue |
| DELETE | `/me/ratings/{movie_id}` | Xóa đánh giá của tài khoản đăng nhập |
| POST | `/predictions` | Dự đoán điểm theo user/movie/thuật toán |
| GET | `/experiments/latest` | Kết quả validation và test của run mới nhất |
| GET | `/experiments/documents/protocol` | Tải quy trình so sánh thực nghiệm dạng Markdown |
| GET | `/experiments/documents/challenges` | Tải tài liệu khó khăn thực nghiệm dạng Markdown |
| GET | `/experiments/documents/figures-stability` | Tải thiết kế biểu đồ, lặp seed/thời gian và bootstrap |
| GET | `/experiments/documents/results` | Tải báo cáo kết quả và nhận xét |
| GET | `/experiments/exports/stability` | Tải báo cáo độ ổn định JSON |
| GET | `/experiments/exports/seeds` | Tải CSV từng lượt seed |
| GET | `/experiments/exports/temporal` | Tải CSV từng fold thời gian |
| GET | `/experiments/exports/bootstrap` | Tải CSV chênh lệch và CI 95% |
| POST | `/auth/register` | Tạo tài khoản bằng name/email/password (ít nhất 8 ký tự) |
| POST | `/auth/login` | Đăng nhập bằng email/password, cấp cookie HttpOnly |
| GET | `/auth/demo-accounts` | Chuẩn bị và liệt kê tối đa 5 tài khoản demo, mật khẩu demo, userId và số rating đã nhập |
| GET | `/auth/me` | Tài khoản của phiên hiện tại, 401 nếu chưa đăng nhập |
| POST | `/auth/logout` | Thu hồi phiên, xóa cookie |
| POST | `/contact` | Lưu name/email/subject/message vào SQLite; không gửi email |

Request dự đoán:

```json
{"user_id": 1, "movie_id": 1, "algorithm": "svd"}
```

`algorithm` chính: `ridge`, `xgboost`, `svd`, `ncf`.
Các mô hình tham chiếu `baseline`, `item_knn`, `nmf` vẫn được hỗ trợ khi có artifact.
Bỏ trường này để dùng mô hình có RMSE validation thấp nhất.
Response có `predicted_rating`, `algorithm`, `cold_start`, `run_id` và hai ID.
`cold_start` nghĩa là user hoặc movie chưa xuất hiện trong train; đây là dự đoán
bằng fallback, không phải lỗi request.

ID không dương, thuật toán không hợp lệ và tham số phân trang sai trả HTTP 422.
Danh mục và thống kê đọc trực tiếp từ CSV, không yêu cầu `prepare` hoặc `train`.
Thiếu CSV bắt buộc hoặc dữ liệu không hợp lệ trả HTTP 503. Dự đoán trả 503 nếu
chưa huấn luyện mô hình. UI hiển thị lỗi, không tạo điểm hoặc danh mục giả.
Kết quả `/movies` có `total`, `offset`, `limit`, `items`.
Mỗi item gồm `movieId`, `title`, `genres`, `year`, `mean_rating`, `rating_count`,
`imdbId`, `tmdbId`. Điểm trung bình/độ phổ biến tính từ toàn bộ `ratings.csv`;
chưa có rating thì `mean_rating = null`, `rating_count = 0`. Năm chưa xác định là `null`.
Các tham số bổ sung: `genre=Action`, `year=classic|2000|2010|2020|all`,
`min_rating=0..5`, `sort=popular|rating|newest|title`.
`/movies/{movie_id}` bổ sung `rating_distribution` (10 mức 0,5–5), `tags` (tối đa
12 nhãn phổ biến), `imdb_url`, `tmdb_url`, `related` (tối đa sáu phim có thể loại chung).
Phim không tồn tại trả HTTP 404; phim không có nhãn trả `tags = []`.

Cache ở `artifacts/catalog/` được tạo bằng cách đọc CSV theo chunk. Thay đổi
đường dẫn, kích thước hoặc mtime CSV sẽ kích hoạt phân tích lại ở request kế tiếp.
Thống kê này không ghi đè `data/movie_stats.parquet`, vốn chỉ dùng train cho học máy.

CORS đã mở cho địa chỉ Vite local `localhost:5173` và `127.0.0.1:5173`.
Frontend đã triển khai. Bản build được phục vụ cùng FastAPI cổng 8000.
Tài khoản/tin nhắn lưu ở `artifacts/application.sqlite3`. ID tài khoản ứng dụng
không ánh xạ tự động sang ID người dùng MovieLens.

Cookie phiên có hiệu lực 7 ngày, HttpOnly, SameSite=Lax. Môi trường HTTPS có thể đặt
`MOVIE_SECURE_COOKIES=1`; origin frontend tùy chỉnh qua `MOVIE_FRONTEND_ORIGIN`.
Mặc định chấp nhận form từ localhost/127.0.0.1 cổng 5173 và 8000.
Google/Facebook OAuth, email xác nhận và gửi email liên hệ chưa tích hợp.

## Đánh giá, Watch History và gợi ý tài khoản

Cookie hiện có path `/api/v1` để các endpoint cá nhân nhận phiên. `/auth/me` tự đổi
phạm vi cookie cũ `/api/v1/auth`; logout xóa cả hai. PUT/DELETE kiểm tra Origin như form tài khoản.
Backend luôn lấy user từ session, không nhận app user_id do frontend chỉ định.

`/me/ratings` trả `items,total,offset,limit`. Mỗi item có trường catalogue cộng
`user_rating`, `updated_at` (Unix milliseconds). Watch History là danh sách đã chấm điểm,
không phải nhật ký phát video hoặc lịch sử bấm poster. Ghi lại cùng một phim cập nhật một hàng;
`created_at` giữ nguyên, `updated_at` được làm mới. Dữ liệu lưu trong bảng `user_ratings`
riêng ở SQLite; không thay đổi CSV MovieLens hoặc điểm trung bình cộng đồng.

`/recommendations` trả items (tối đa 20), mode, algorithm, requested_algorithm, cold_start,
run_id, history_count,
model_status, available_algorithms, authenticated. Mỗi item có `predicted_rating`:

- `popular`: prior trung bình có làm trơn bởi số lượt; dùng cho guest/user chưa có tín hiệu.
- `content`: cosine với hồ sơ thể loại có trọng số từ rating riêng của tài khoản.
- `model`: dự đoán từ run thật; dùng hồ sơ MovieLens đã học hoặc adapter cho lịch sử hiện tại.

`predicted_rating=null` ở popular/content; không quy đổi độ tương đồng thành điểm /5 giả.
Không có run thì available_algorithms=[], model_status=not_trained. Chọn một mô hình
chưa train trả 503; frontend chỉ cho chọn các mô hình có artifact thật. Phim đã chấm bị loại
khỏi gợi ý. Gợi ý được tính lại khi đánh giá sửa/xóa; mỗi tài khoản có lịch sử riêng.

Các adapter run hiện tại: Baseline ước lượng user bias; Item-KNN dùng residual lịch sử;
SVD ridge fold-in user bias/factors; NMF NNLS không âm với item factors giữ cố định.
Không tái sử dụng MovieLens user có cùng ID với tài khoản website. Nếu không có phim hỗ trợ
nào xuất hiện trong mô hình, trả content/popular với model_status=no_known_history_items.
Metric offline của run gốc không phải kết quả đánh giá adapter cold-user mới.

Ridge và XGBoost tạo đặc trưng từ lịch sử tài khoản hiện tại cùng thống kê phim
đã học trên train. Không dùng ID tài khoản làm ID MovieLens. NCF tối ưu embedding
và bias riêng của user mới bằng rating hiện tại, giữ cố định toàn bộ mạng đã học.

`algorithm=auto` trong recommendations đọc `best_recommendation_algorithm`
(NDCG@10 validation cao nhất); run cũ hoặc không có ranking dùng RMSE validation.
Endpoint predictions vẫn chọn `best_algorithm` theo RMSE. Test không đổi hai lựa chọn.
Khi history_count=0, dù chọn thuật toán nào, mode=popular, algorithm=null,
predicted_rating=null; requested_algorithm vẫn ghi lựa chọn của trình duyệt.
Không thể suy ra sở thích cá nhân chỉ từ tài khoản trống.

Quy trình so sánh theo đề cương nhóm: [experimental-protocol.md](experimental-protocol.md).
`GET /experiments/latest` trả thêm `analysis` (null nếu chưa chạy
`analyze-experiments`), chứa `run_id` và `cold_history.validation/test` cho đồ thị
0/5/10/20 rating; mỗi cohort ghi số user, query và RMSE/MAE của từng mô hình.
Endpoint cũng trả `stability` (null nếu chưa chạy), gồm `status`, `seed_runs`,
`temporal_runs`, `seed_summary.validation/test`, `temporal_summary` và
`bootstrap.validation/test`. Chỉ số summary có `mean`, sample `std`, `n`;
chênh lệch bootstrap giữ dấu âm/dương, với `delta`, `ci_low`, `ci_high`, `users`
và `conclusion`. File đang chạy chỉ chứa các lượt đã hoàn tất; tải lại để cập nhật.
Khó khăn và giới hạn: [experimental-challenges.md](experimental-challenges.md).

## Hồ sơ dataset và tài khoản mới

Auth trả thêm profile_type=personal/dataset, dataset_user_id (null với personal),
imported_rating_count và profile_modified. Endpoint demo chỉ xuất thông tin đăng nhập
synthetic do ứng dụng tạo, không xuất tài khoản thông thường hay hash/salt.

Recommendations thêm strategy, profile_type và dataset_user_id. Với demo nguyên trạng
thuộc đúng nguồn dataset, nếu user nằm trong model.user_ids thì dùng trained_dataset_user;
nếu không hoặc lịch sử đã sửa thì profile_fold_in. Hai nhánh đều loại toàn bộ phim đã chấm.
Chi tiết và giới hạn đánh giá: [user-profiles.md](user-profiles.md).
