# Ứng dụng dự đoán đánh giá phim — MovieLens 25M

Backend FastAPI và frontend React/TypeScript phục vụ tìm kiếm phim, chi tiết,
đánh giá cá nhân, tài khoản, lịch sử đánh giá và gợi ý phim. Phần thực nghiệm,
API so sánh mô hình, giao diện thực nghiệm và bộ test cũ đã được xoá.

Giao diện thực nghiệm mới ở `/experiments`, mở từ menu hoặc nút nổi **Thực nghiệm**.
Hiện trang chỉ có tiêu đề và trạng thái trống; chức năng sẽ được bổ sung từng bước.

## Cấu trúc

```text
project/
├── configs/default.toml       # Dữ liệu và cấu hình huấn luyện cho ứng dụng
├── src/movie_rating/
│   ├── data/                  # Làm sạch, đọc và chuẩn bị dữ liệu
│   ├── models/                # Mô hình phục vụ dự đoán/gợi ý
│   ├── services/              # Danh mục, tài khoản, dự đoán và gợi ý
│   ├── api/                   # FastAPI và endpoint của ứng dụng
│   ├── evaluation.py          # Đánh giá mô hình trong pipeline huấn luyện
│   ├── training.py            # Huấn luyện và lưu mô hình cho ứng dụng
│   ├── settings.py            # Đọc và kiểm tra cấu hình
│   └── cli.py                 # analyze / prepare / train / evaluate / serve
├── frontend/                  # React/TypeScript
├── artifacts/                 # Dữ liệu chuẩn bị, cache và SQLite
├── docs/                      # Tài liệu dự án
└── scripts/                   # Cài đặt, khởi động và xử lý dữ liệu
```

## Chạy ứng dụng

Mở `../run_project.bat`. Script build frontend rồi khởi động backend và mở
trình duyệt sau khi xác nhận API và bundle giao diện sẵn sàng.

Các lệnh sau chạy trong thư mục `project`:

```powershell
.\scripts\setup.ps1
.\.venv\Scripts\movie-rating.exe serve
```

Chọn PyTorch CPU bằng `.\scripts\setup.ps1 -TorchBuild cpu` nếu cần.
Giao diện: <http://127.0.0.1:8000>. Swagger: <http://127.0.0.1:8000/docs>.
Xem thêm [hướng dẫn frontend](frontend/README.md).

## Dữ liệu

Mặc định ứng dụng dùng `../dataset_cleaned/`, gồm `ratings.csv`, `movies.csv`,
`tags.csv` và `links.csv`. Dataset có 25.000.095 rating và 62.423 phim.
Xem [phân tích dữ liệu](docs/dataset-analysis.md) và
[quy tắc làm sạch](../dataset_cleaned/README.md).

Danh mục phim hoạt động mà không cần huấn luyện mô hình. Khi chưa có mô hình,
gợi ý dùng thể loại/phổ biến và hiển thị rõ nguồn.

Đọc thống kê hoặc tạo một bản dữ liệu sạch mới:

```powershell
.\.venv\Scripts\movie-rating.exe analyze
.\.venv\Scripts\python.exe -X utf8 scripts/clean_dataset.py --output ../dataset_cleaned_new
```

## Huấn luyện mô hình cho chức năng dự đoán/gợi ý

Các mô hình lõi và pipeline CLI được giữ lại để phục vụ ứng dụng:

```powershell
.\.venv\Scripts\movie-rating.exe prepare
.\.venv\Scripts\movie-rating.exe train
.\.venv\Scripts\movie-rating.exe evaluate
```

`prepare` yêu cầu thư mục dữ liệu đầu ra mới. Cấu hình seed, danh sách mô hình
và thư mục artifacts nằm trong mục `[training]` của `configs/default.toml`.
Tham số mô hình nằm trong `[models.<tên>]`. Có thể dùng `--config` để chọn
file TOML khác hoặc đặt `MOVIE_DATASET_DIR` và `MOVIE_ARTIFACTS_DIR`.

Mô hình được lưu trong `artifacts/runs/`; `latest.json` trỏ tới run đang dùng.
NCF dùng CUDA nếu môi trường PyTorch hỗ trợ. Huấn luyện toàn bộ dataset cần
được cân nhắc theo RAM, GPU và thời gian thực tế.

## Tài khoản và API

Tài khoản, lịch sử đánh giá và tin nhắn được giữ trong
`artifacts/application.sqlite3`. Xem [API](docs/api.md) và
[luồng hồ sơ người dùng](docs/user-profiles.md).

Build giao diện bằng `npm run build` trong `frontend`.
