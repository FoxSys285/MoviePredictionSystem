# MoviePredictionSystem

Ứng dụng học máy dự đoán điểm đánh giá phim trên **MovieLens 25M**, gồm backend
FastAPI, giao diện React/TypeScript và phòng thực nghiệm với 12 phương pháp thuộc
6 nhóm nhiệm vụ. Người dùng có thể tìm phim, xem chi tiết, chấm điểm, quản lý
Watch History và nhận gợi ý theo lịch sử.

## Chạy trên Windows

Yêu cầu Python 3.12 và Node.js/npm. Trong PowerShell tại thư mục repository:

```powershell
cd project
.\scripts\setup.ps1
cd frontend
npm ci
cd ../..
.\run_project.bat
```

Giao diện/API: <http://127.0.0.1:8000>. Swagger: <http://127.0.0.1:8000/docs>.

## Dữ liệu và mô hình

Tải [MovieLens 25M từ GroupLens](https://grouplens.org/datasets/movielens/25m/),
đặt `ratings.csv`, `movies.csv`, `tags.csv` và `links.csv` trong `dataset/`.
Tạo bản dữ liệu sạch trước khi huấn luyện theo
[hướng dẫn dự án](project/README.md#bản-dữ-liệu-đã-làm-sạch).

CSV MovieLens, môi trường Python, `node_modules`, mô hình/hiện vật sinh khi chạy
và cơ sở dữ liệu tài khoản cục bộ không được lưu trong repository. Một bản clone
mới cần chuẩn bị dữ liệu và huấn luyện mô hình để sử dụng đầy đủ chức năng.

## Tài liệu

- [Hướng dẫn cài đặt, cấu hình và huấn luyện](project/README.md)
- [Phòng thực nghiệm: 6 nhóm nhiệm vụ](project/docs/experiment-lab.md)
- [Kiến trúc hệ thống](project/docs/architecture.md)
- [API](project/docs/api.md)
- [Giao diện frontend](project/frontend/README.md)
- [Báo cáo đồ án — Word](Bao%20cao%20do%20an%20-%20Movie%20Rating%20Prediction.docx)
- [Báo cáo đồ án — PDF](Bao%20cao%20do%20an%20-%20Movie%20Rating%20Prediction.pdf)

## Kiểm tra

Trong thư mục `project`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

Trong `project/frontend`, chạy `npm run build` để kiểm tra TypeScript và build
giao diện.
