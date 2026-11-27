# MoviePredictionSystem

Ứng dụng khám phá phim và dự đoán điểm đánh giá trên MovieLens 25M, gồm backend
FastAPI và giao diện React/TypeScript. Người dùng có thể tìm phim, xem chi tiết,
chấm điểm, quản lý lịch sử đánh giá và nhận gợi ý theo lịch sử.

Phần thực nghiệm và bộ test cũ đã được xoá để xây dựng lại từ đầu.

## Chạy trên Windows

Yêu cầu Python 3.12 và Node.js/npm. Tại thư mục repository:

```powershell
cd project
.\scripts\setup.ps1
cd frontend
npm ci
cd ../..
.\run_project.bat
```

`setup.ps1` cài PyTorch CUDA mặc định. Có thể chọn bản CPU bằng
`.\scripts\setup.ps1 -TorchBuild cpu` trong thư mục `project`.

Giao diện/API: <http://127.0.0.1:8000>. Swagger: <http://127.0.0.1:8000/docs>.

## Dữ liệu và chức năng

Dataset gốc ở `dataset/`, bản sạch ở `dataset_cleaned/`. Dữ liệu tài khoản lưu ở
`project/artifacts/application.sqlite3`; cache danh mục và dữ liệu đã chuẩn bị
được giữ lại. Các mô hình phục vụ dự đoán/gợi ý nằm trong
`project/src/movie_rating/models/`.

## Tài liệu

- [Cài đặt và huấn luyện mô hình cho ứng dụng](project/README.md)
- [Kiến trúc hệ thống](project/docs/architecture.md)
- [API](project/docs/api.md)
- [Giao diện frontend](project/frontend/README.md)

Để kiểm tra TypeScript và build giao diện, chạy `npm run build` trong
`project/frontend`.
