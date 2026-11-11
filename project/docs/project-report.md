# Báo cáo đồ án Nhóm 05

Bản bàn giao tại thư mục gốc repository:

- `Nhom05_DoAn.docm`: báo cáo hoàn chỉnh, không còn VBA/ribbon tự chạy; giữ hai trang bìa từ mẫu.
- `Nhom05_DoAn.pdf`: bản kết xuất đã kiểm tra, 59 trang.
- `Nhom05_DoAn_HoanThien.zip`: DOCX chỉnh sửa được, PDF, log CSV và hướng dẫn. DOCX được giữ bên trong ZIP vì trên máy đã quan sát thấy một bản DOCX trung gian bị đổi thành DOCM có mã tự chạy.

Nội dung gồm bốn chương theo yêu cầu, kết luận và hướng phát triển, tài liệu tham khảo, phụ lục; 23 hình và 35 bảng đánh số. Ba mục lục tự động đã cập nhật trước khi xuất PDF. Các thông số A4, lề, Times New Roman, cỡ chữ, giãn dòng và đánh số trang đã kiểm tra trực tiếp trong OOXML. `report-validation.json` lưu kết quả và SHA256 bản bàn giao.

## Nguồn số liệu

- Run ID: `20261111T125529`, tại `project/artifacts/report-temporal/20261111T125529`.
- 49.552 ratings thật, mẫu 300 người dùng; đây không phải benchmark toàn bộ MovieLens 25M.
- Chia toàn cục theo thời gian 70/15/15; giữ timestamp bằng nhau cùng tập. Lọc chỉ phần fit. Năm fold thời gian mở rộng, GridSearchCV cho mô hình sklearn, Optuna cho NCF, Gymnasium cho Q-Learning.
- `results.json`, `metrics.csv`, split Parquet, model joblib và dự đoán Test được giữ cùng run. Bảng/hình sinh trực tiếp từ artifact. Các mốc tiêu chí là mục tiêu dự án, không phải chuẩn chất lượng phổ quát.
- Bộ chạy báo cáo độc lập với lịch sử phòng thực nghiệm UI. Không trộn kết quả của hai giao thức.

XGBoost tốt nhất trong bốn mô hình học dự đoán rating theo Validation; Mean Rating còn có Validation RMSE thấp hơn, nên demo Auto chọn baseline. RF có F1 cao nhưng gần như dự đoán toàn bộ lớp Thích; DBSCAN loại hơn 90% thành noise; Q-Learning chưa vượt baseline. Các điểm hạn chế này được giữ trong báo cáo.

## Tái lập

Sau khi có môi trường và dữ liệu chuẩn bị, thực hiện từ thư mục gốc:

```powershell
project/.venv/Scripts/python.exe -m pip install -r project/requirements-report.txt
project/.venv/Scripts/python.exe -X utf8 project/scripts/report_temporal_experiments.py
project/.venv/Scripts/python.exe -X utf8 project/scripts/report_figures.py
project/.venv/Scripts/python.exe -X utf8 project/scripts/build_project_report.py
powershell -NoProfile -ExecutionPolicy Bypass -File project/scripts/finalize_report.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File project/scripts/package_project_report.ps1
```

`finalize_report.ps1` dùng LibreOffice headless và UNO, không mở Word. Có thể chỉ định `-LibreOfficeProgramDirectory` và `-PythonExecutable`; Python phải khớp ABI của pyuno. Môi trường đã kết xuất thành công dùng LibreOffice 26.8.2.2 và Python 3.13. Bộ tạo nội dung dùng `python-docx`, giữ bìa bằng cách đọc ZIP/XML từ bản gốc, không chạy macro. Trước khi ghi đè, bộ đóng gói kiểm tra file hiện tại còn khớp bản gốc để tránh mất nội dung người dùng sửa.

Các chỉnh sửa nhỏ về ngắt bảng và khoảng cách mục lục trong mã sinh sau lần kết xuất cuối chưa được áp dụng vào bản 59 trang, vì Windows sau đó chặn Python bằng cảnh báo virus. Bản bàn giao vẫn là bản kết xuất thành công đã kiểm tra; các công cụ bị chặn không được chạy lại hay sửa để vượt cảnh báo.

Kiểm thử giao thức báo cáo đã chạy thành công 4 bài trước khi có chặn công cụ: ranh giới thời gian, lọc fit, dự đoán không phụ thuộc rating giữ lại, reward/mask Gymnasium. Demo đã kiểm tra Auto và cả bốn mô hình học, loại phim đã xem và giới hạn rating.

## Web demo của bộ chạy báo cáo

```powershell
cd project
.venv/Scripts/python.exe -m uvicorn report_demo:app --app-dir scripts --host 127.0.0.1 --port 8143
```

Chọn phim đã xem, nhập rating, thêm lịch sử và xem Top-N. Auto chọn theo Validation có xét baseline. Tùy chọn thủ công hỗ trợ khảo sát các mô hình. Ảnh chụp giao diện nằm tại `project/docs/report-assets/report_demo.png`.

## Ghi nhận về file mẫu và máy thực hiện

Phân tích tĩnh file mẫu phát hiện `OnRibbonLoad` giải mã dữ liệu nhúng, cấp bộ nhớ thực thi và gọi `CreateThread`. Đây không phải chức năng định dạng báo cáo. VBA, ribbon và dữ liệu nhúng liên quan đã bị loại khỏi bản bàn giao. Bản gốc được giữ tại `.codex-artifacts/report/Nhom05_DoAn_BanGoc.docm` để bảo toàn đầu vào; không dùng bản này để xem báo cáo.

Windows đồng thời cảnh báo virus với một số chương trình trên máy, bao gồm Python, Git và ripgrep. Không tắt bảo vệ, không sửa hoặc đổi tên chương trình bị chặn. Công việc này chỉ tạo và kiểm tra tài liệu không chứa phần tự chạy; không xác nhận máy tính đã được xử lý mã độc. Nên kiểm tra máy trước khi tiếp tục dùng Word hoặc tái chạy thực nghiệm.
