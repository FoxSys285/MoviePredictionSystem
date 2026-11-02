# Ứng dụng học máy dự đoán đánh giá phim — MovieLens 25M

Trang **Thực nghiệm** có sidebar 5 nhóm với 11 thuật toán. Chỉ so sánh các chỉ số đã chốt: hồi quy (4), phân loại (4), phân cụm (3), luật kết hợp (3), PCA (3). Validation và Test hiển thị cùng lúc; mỗi chỉ số có đúng một biểu đồ cột kèm giá trị. Xem [hướng dẫn phòng thực nghiệm](docs/experiment-lab.md).

Login có 5 tài khoản demo lấy lịch sử từ MovieLens, hiển thị email/mật khẩu và nút
điền nhanh. [Luồng tài khoản mới và hồ sơ dataset](docs/user-profiles.md) mô tả cách
nhập lịch sử, nhận diện đúng user và lựa chọn chiến lược gợi ý.

Website có gợi ý tối đa 20 phim ở Home/Gallery, đánh giá phim gắn với tài khoản,
Watch History và các hàng phim theo thể loại. Khi chưa có mô hình đã huấn luyện,
gợi ý nội dung/phổ biến được ghi nhãn rõ; không tạo điểm dự đoán giả.

Phần nền cho đồ án: dự đoán **điểm người dùng đánh giá phim** trên thang 0.5–5,
huấn luyện 4 thuật toán và so sánh trên cùng dữ liệu. Giao diện React + TypeScript
đã triển khai theo 5 ảnh UI/UX trong `images`: Gallery, About, Contact, Đăng nhập,
Đăng ký; có thêm trang chủ theo cùng phong cách.

## Cấu trúc

```text
Movie Rating Prediction/
├── dataset/                       # Bốn CSV gốc của MovieLens 25M
│   ├── ratings.csv
│   ├── movies.csv
│   ├── tags.csv                   # Từ khóa cộng đồng trong trang chi tiết
│   └── links.csv                  # Liên kết IMDb và TMDB
└── project/
    ├── configs/default.toml        # Dữ liệu, seed, chia tập, tham số 4 mô hình
    ├── src/movie_rating/
    │   ├── data/                  # Đọc theo chunk, lấy mẫu người dùng, chia thời gian
    │   ├── models/                # Ridge, XGBoost, SVD, NCF và mô hình tham chiếu
    │   ├── evaluation.py          # RMSE, MAE, tốc độ, warm/cold start
    │   ├── training.py            # Huấn luyện, lưu mô hình, bảng so sánh
    │   ├── services/              # Dự đoán dùng chung cho ứng dụng
    │   ├── api/                   # FastAPI, schema và các endpoint
    │   ├── settings.py            # Cấu hình có kiểm tra hợp lệ
    │   └── cli.py                 # prepare / train / evaluate / serve
    ├── frontend/                  # React/TypeScript, trang và component, ảnh WebP
    ├── notebooks/                 # Kế hoạch EDA và phân tích kết quả
    ├── docs/                      # Kiến trúc, phương pháp, API
    ├── tests/                     # Kiểm tra thuật toán và pipeline bằng dữ liệu giả lập
    ├── artifacts/                 # Sinh sau khi chạy; không đưa vào Git
    ├── scripts/                   # Script cài đặt Windows
    ├── pyproject.toml
    └── requirements-lock.txt      # Phiên bản đã cài và kiểm tra trên máy hiện tại
```

## Bốn thuật toán chính

| Thuật toán | Ý tưởng | Vai trò |
|---|---|---|
| Linear / Ridge | One-hot sparse, thống kê train và tương tác hồ sơ thể loại, L2 | Hồi quy tuyến tính |
| XGBoost | Cây boosting trên đặc trưng thống kê/thể loại, không dùng ID số thô | Hồi quy phi tuyến |
| SVD / Funk SVD | Phân rã ma trận với bias, tối ưu bằng SGD trên đánh giá quan sát | Mô hình nhân tố tiềm ẩn |
| NCF | Embedding user/item, MLP trên phần dư bias, MSE, AdamW, early stopping | Deep learning với PyTorch CPU |

Baseline Bias, Item-KNN và NMF vẫn có thể cấu hình làm mốc tham chiếu.
Combobox ở Home/Gallery luôn hiển thị bốn thuật toán chính; chỉ bật mô hình có
artifact đã train. Tài khoản trống nhận gợi ý cộng đồng và lời mời chấm phim,
không có điểm dự đoán cá nhân. Có rating thì tính lại hồ sơ riêng cho từng mô hình.
Auto gợi ý chọn NDCG@10 validation; endpoint dự đoán rating chọn RMSE validation.
SVD ở đây là **biased matrix factorization**, không phải `numpy.linalg.svd`.
NMF không coi các ô chưa đánh giá là điểm 0. Không dùng thư viện `implicit` vì
mục tiêu hiện tại là dự đoán điểm đánh giá tường minh.

## Chạy trên Windows / PowerShell

Python 3.12 đã được dùng để tạo `project/.venv` và cài các thư viện.
Các lệnh dưới đây chạy **trong thư mục `project`**. Không bắt buộc activate môi trường.

Để chạy toàn bộ ứng dụng, mở `../run_project.bat`. File này build giao diện mới,
thay thế server cũ của đúng dự án và chỉ mở trình duyệt khi server/bundle đã được
xác minh. Lỗi trùng cổng với ứng dụng khác được báo rõ, không mở nhầm trang.

```powershell
# Nếu cần tạo lại môi trường từ đầu:
.\scripts\setup.ps1

# Khởi động API ngay cả khi chưa có dataset:
.\.venv\Scripts\movie-rating.exe serve
```

Swagger: <http://127.0.0.1:8000/docs>.
Giao diện đã build: <http://127.0.0.1:8000>.
Nếu chưa có `frontend/dist`, chạy `npm ci` và `npm run build` trong `frontend`
trước khi khởi động backend. Xem [hướng dẫn frontend](frontend/README.md).
Health: <http://127.0.0.1:8000/api/v1/health>.
Khi chưa có dữ liệu/mô hình, health vẫn hoạt động; chức năng phụ thuộc dữ liệu
trả HTTP 503 kèm hướng dẫn chuẩn bị dữ liệu.

Dataset gốc nằm trong `dataset/`; mặc định huấn luyện/website dùng `dataset_cleaned/`.
Website đọc đủ bốn CSV để phục vụ danh mục, chi tiết và thống kê mà không cần
huấn luyện trước. Các số liệu được kiểm tra: **62.423 phim, 25.000.095 đánh giá,
162.541 người dùng và 1.093.360 nhãn**. Xem [phân tích dataset](docs/dataset-analysis.md).

Để xem lại thống kê (tự dùng cache nếu CSV không đổi):

```powershell
.\.venv\Scripts\movie-rating.exe analyze
```

## Bản dữ liệu đã làm sạch

`../dataset_cleaned/` chứa bốn CSV giữ nguyên schema, `README.md` mô tả cách xử lý
và `cleaning_report.json` ghi số dòng thay đổi cùng SHA-256. Dữ liệu gốc không bị
ghi đè. Script kiểm tra ID, điểm, timestamp và liên kết phim; chuẩn hóa văn bản,
loại nhãn trống/placeholder và xử lý trùng lặp trên toàn bộ dữ liệu.

Chạy lại vào một thư mục mới:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/clean_dataset.py --output ../dataset_cleaned_new
```

Đặt `$env:MOVIE_DATASET_DIR = '../dataset_cleaned'` để dùng bản sạch. Với một
thực nghiệm huấn luyện mới, chọn `artifacts_directory` riêng trước khi `prepare`.

Để chuẩn bị và huấn luyện các mô hình:

```powershell
.\.venv\Scripts\movie-rating.exe prepare
.\.venv\Scripts\movie-rating.exe train
.\.venv\Scripts\movie-rating.exe evaluate
.\.venv\Scripts\movie-rating.exe analyze-experiments
.\.venv\Scripts\movie-rating.exe stability
.\.venv\Scripts\movie-rating.exe serve
```

Nếu CSV đặt ở nơi khác, sửa `data.directory` ở `configs/default.toml`,
hoặc đặt `$env:MOVIE_DATASET_DIR` đến thư mục chứa CSV.
Đường dẫn trong cấu hình được tính từ `project/`.

Mặc định lấy toàn bộ lịch sử của khoảng **2% người dùng** để phát triển.
Đổi `user_sample_fraction = 1.0` khi chạy toàn bộ 25M, sau khi kiểm tra RAM và thời gian.
Việc đọc theo chunk giảm bộ nhớ lúc đọc; các tập sau lấy mẫu vẫn được giữ trong RAM
để sắp xếp/chia dữ liệu và huấn luyện. Chưa benchmark toàn bộ 25M trên máy này.

Mỗi phiên bản dữ liệu cần một thư mục artifacts riêng: tạo bản sao file config,
đổi `experiment.artifacts_directory`, rồi chạy:

```powershell
.\.venv\Scripts\movie-rating.exe --config configs/my-experiment.toml prepare
.\.venv\Scripts\movie-rating.exe --config configs/my-experiment.toml train
```

Không ghi đè dữ liệu đã chuẩn bị. Mỗi lần `train` tạo một run riêng và cập nhật
`artifacts/latest.json` khi cả 4 mô hình hoàn tất. `evaluate` lưu kết quả test của
run hiện tại; chạy lại sẽ đọc kết quả đã lưu.

## Đầu ra thực nghiệm

```text
artifacts/
├── catalog/                    # Thống kê toàn bộ CSV dành riêng cho website
│   ├── movies.parquet          # Điểm trung bình, số lượt và phân bố điểm từng phim
│   ├── tags.parquet            # Tối đa 12 nhãn phổ biến nhất mỗi phim
│   └── summary.json            # Tổng quan dataset và dấu vết CSV nguồn
├── data/
│   ├── train.parquet
│   ├── validation.parquet
│   ├── test.parquet
│   ├── movies.parquet
│   ├── movie_stats.parquet        # Điểm trung bình và độ phổ biến chỉ từ train
│   └── manifest.json
├── runs/<run-id>/
│   ├── ridge.joblib
│   ├── xgboost.joblib
│   ├── svd.joblib
│   ├── ncf.joblib
│   ├── validation.json
│   ├── comparison.csv
│   └── test.json                  # Có sau lệnh evaluate
├── latest.json
└── application.sqlite3            # Tài khoản/phiên đăng nhập/tin nhắn, tạo khi sử dụng
```

`validation.json` lưu cấu hình, seed, phiên bản thư viện, dấu vết dữ liệu, điểm số,
thời gian và dung lượng mô hình. `comparison.csv` thuận tiện đưa vào báo cáo.
`analyze-experiments` và `stability` thuộc pipeline huấn luyện phục vụ gợi ý.
Phòng thực nghiệm mới chạy độc lập trong `artifacts/experiment-lab/<run-id>/`,
không thay `latest.json` hoặc các mô hình đang phục vụ ứng dụng.
Chỉ nạp file `.joblib` do dự án tự tạo và tin cậy.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

Kiểm tra dùng dữ liệu giả lập trong thư mục tạm, không thêm dữ liệu mẫu vào `dataset`.
Phân tích toàn bộ dataset đã hoàn tất; kết quả RMSE/MAE chỉ xuất hiện sau khi
chạy `prepare`, `train` và `evaluate`. Thống kê hiển thị trên website không thay
đổi thống kê train hoặc dùng làm đầu vào cho mô hình.

Đọc thêm: [kiến trúc](docs/architecture.md), [phương pháp đánh giá](docs/methodology.md),
[hợp đồng API](docs/api.md), [giao diện và cách chạy](frontend/README.md).

Nguồn dữ liệu và cách trích dẫn:
[MovieLens 25M README — GroupLens](https://files.grouplens.org/datasets/movielens/ml-25m-README.html).
