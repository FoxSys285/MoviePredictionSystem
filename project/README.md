# Ứng dụng học máy dự đoán đánh giá phim — MovieLens 25M

Trang **Thực nghiệm** có sidebar sáu nhóm, chạy thuật toán thật trên MovieLens: Regression, Classification, Clustering, Association Rules, Dimensionality Reduction và Reinforcement Learning. Năm nhóm chạy dùng preset cố định do backend xác thực; tham số chỉ đọc, còn chọn nhóm/thuật toán, hủy chạy, lịch sử và export vẫn có. Pipeline v3 dùng chia thời gian trong từng user và 3 lượt CV thời gian cho Regression/Classification, chọn cấu hình trên Validation, refit Train + Validation. Thực nghiệm lớn có giới hạn RAM và đặc trưng theo batch; xem [vận hành và bộ nhớ](docs/personalized-experiments.md), [hướng dẫn Experiment Studio](docs/experiment-studio.md) và [kết quả xác nhận](docs/experiment-studio-results.md).

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
| NCF | Embedding user/item, MLP trên phần dư bias, MSE, AdamW, early stopping | Deep learning với PyTorch, tự dùng CUDA nếu khả dụng |

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

Mặc định dùng toàn bộ **25.000.095 ratings** (`user_sample_fraction = 1.0`).
Chuẩn bị dữ liệu và chạy thực nghiệm có thể cần nhiều thời gian và RAM; trên máy
16 GB nên chỉ chọn một thuật toán mỗi lần, giữ tuning tắt và theo dõi bộ nhớ.
Giao diện có tùy chọn giới hạn user để chạy thử, nhưng mặc định không lấy mẫu.
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

Đọc thêm: [phân cụm user có lịch sử](docs/clustering-experiments.md),
[kiến trúc](docs/architecture.md), [phương pháp đánh giá](docs/methodology.md),
[hợp đồng API](docs/api.md), [giao diện và cách chạy](frontend/README.md).

Nguồn dữ liệu và cách trích dẫn:
[MovieLens 25M README — GroupLens](https://files.grouplens.org/datasets/movielens/ml-25m-README.html).

## CUDA cho NCF

`scripts/setup.ps1` cài PyTorch 2.14.1 CUDA 13.2 theo `requirements-torch-cuda.txt`. Wheel này
cũng giữ tensor CPU và fallback `cuda` → `cpu`; máy chỉ dùng CPU có thể chọn bản CPU-only bằng
`.\scripts\setup.ps1 -TorchBuild cpu`. Không cài lại PyTorch bằng `pip install -e .` riêng lẻ
nếu muốn giữ build đã chọn; dùng script setup để khôi phục đúng build.

Build được đối chiếu với [bảng cài PyTorch 2.14.1 chính thức](https://pytorch.org/get-started/previous-versions/)
(Windows, CUDA 13.2). [Bảng tương thích NVIDIA](https://docs.nvidia.com/deploy/cuda-compatibility/620/minor-version-compatibility.html)
ghi CUDA 13.x cần driver từ nhánh 580 trở lên; máy này dùng 621.24.
RTX 3050 Laptop được PyTorch nhận ở compute capability 8.6. Python 3.12 wheel đã được cài và
kiểm tra trong `.venv`.

NCF chạy embedding và MLP trên CUDA khi `torch.cuda.is_available()` đúng. Bias baseline, mã hóa
ID và chuẩn bị mảng vẫn ở CPU. Với NCF hybrid giảm chiều, mạng NCF nền chạy theo thiết bị đó;
PCA/TruncatedSVD, tạo feature và nhánh correction MLP chạy CPU.

## Tiếp tục thực nghiệm đã dừng

Thực nghiệm hồi quy/phân loại mới lưu `progress.json` sau mỗi cấu hình Validation, mỗi fold CV
và mỗi refit/Test hoàn tất. NCF còn ghi checkpoint atomic theo epoch, gồm optimizer, early
stopping, trọng số tốt nhất và trạng thái RNG. Khi dừng hoặc worker bị gián đoạn, chọn lần chạy
trong lịch sử rồi nhấn **Tiếp tục từ checkpoint**. Báo cáo và file export chỉ xuất hiện sau khi
tất cả biến thể/thuật toán hoàn tất.

Trước khi dùng lại tiến độ, ứng dụng so khớp dữ liệu/split, seed, tham số, code và phiên bản thư
viện. Nếu khác, tiến độ cũ được giữ nguyên nhưng không tái sử dụng được; hãy tạo run mới. Các run
cũ không có `progress.json` hoặc checkpoint epoch không được khôi phục hồi tố. XGBoost và các fit
không có checkpoint epoch sẽ chạy lại riêng fit đang dở; các cấu hình, fold và refit đã lưu vẫn
được giữ.

Trong Experiment Studio, chọn XGBoost rồi đặt **Số luồng CPU** thành 2, 4 hoặc 8. Mặc định là 4
theo benchmark cohort 4.904 user; `threadpool_limits` được đặt cùng `n_jobs`.
Chạy benchmark có giới hạn user bằng:

```powershell
\.venv\Scripts\python.exe scripts\benchmark_xgboost_threads.py --users 5000
```

Benchmark giữ seed 42, giao thức thời gian từng user, 150 cây, depth 6 và learning rate 0.05.
Nó ghi feature/cache time, fit time, tổng thời gian và RSS đỉnh vào thư mục benchmark mới dưới
`artifacts/benchmarks/`; không chạy toàn bộ MovieLens 25M.

Random Forest trong preset phân loại v2 giữ 100 cây, depth 8 và 4 luồng CPU, nhưng giới hạn
bootstrap ở tối đa **250.000 lượt lấy mẫu có hoàn lại mỗi cây**. Mỗi cây lấy mẫu độc lập từ
toàn bộ Train của lượt fit; fold nhỏ dùng tối đa số hàng hiện có. Thống kê đặc trưng vẫn fit
trên toàn bộ Train; 3 fold thời gian, Validation và Test không bị lấy mẫu nhỏ lại. Giới hạn này
có thể thay đổi chất lượng dự đoán, cần đánh giá lại trên dữ liệu đầy đủ. Tổng pipeline còn
có lượt chọn cấu hình trên Validation và lượt refit Train + Validation ngoài 3 fold CV.

API không dùng preset có thể đặt `parameters.forest_max_samples` thành số nguyên dương;
đặt `null` để dùng bootstrap theo kích thước toàn bộ Train như trước. Cấu hình export lưu mức
giới hạn; log từng fit ghi số lượt lấy mẫu thực tế. Khởi động lại ứng dụng và tạo run mới để
áp dụng thay đổi; checkpoint từ code/preset cũ không dùng chung với cấu hình mới.

Đo thời gian và F1/AUC Validation trên cùng cohort, seed và ngưỡng 0,5:

```powershell
\.venv\Scripts\python.exe scripts\benchmark_random_forest_sampling.py --users 5000 --trees 100
```

Kết quả được lưu vào `artifacts/benchmarks/random-forest-sampling-*.json`. Đây là benchmark
cohort có giới hạn, không thay thế đánh giá 3 fold và Test trên toàn bộ dữ liệu.

Lượt đo ngày 26/11/2026: 544.920 hàng Train, 116.708 hàng Validation, 100 cây; tổng thời gian
fit gồm tạo đặc trưng giảm từ 25,57 xuống 16,16 giây (1,58 lần). F1 Validation ở ngưỡng 0,5
đổi từ 0,79883 thành 0,79863; AUC từ 0,78554 thành 0,78583. Thời gian phụ thuộc máy và tải
hệ thống; không suy ra thời gian hay chất lượng trên toàn bộ 25M từ lượt đo này.
