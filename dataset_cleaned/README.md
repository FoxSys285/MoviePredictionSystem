# Dataset MovieLens đã làm sạch

Nguồn: `dataset/`. Giữ nguyên bốn schema CSV và mã định danh, không lấy mẫu.
Dữ liệu gốc đã được kiểm tra SHA-256 trước và sau khi chạy, không thay đổi.

| Tệp | Dòng gốc | Dòng sạch | Thiếu/sai bắt buộc | Phim không tồn tại | Trùng |
|---|---:|---:|---:|---:|---:|
| movies.csv | 62,423 | 62,423 | 0 | 0 | 0 |
| ratings.csv | 25,000,095 | 25,000,095 | 0 | 0 | 0 |
| tags.csv | 1,093,360 | 1,093,332 | 26 | 0 | 2 |
| links.csv | 62,423 | 62,423 | 0 | 0 | 0 |

Chuẩn hóa Unicode NFC và khoảng trắng; giữ nguyên cách viết tên phim. Thể loại
chưa biết giữ `(no genres listed)`; không tự điền năm, điểm hoặc nội dung phim.
Nhãn được chuyển chữ thường bằng casefold; bỏ nhãn rỗng hoặc chỉ chứa các ký
hiệu thiếu dữ liệu `NA`, `N/A`, `null`, `none`, `nan` (không phân biệt hoa/thường).
Với cùng người dùng và phim, điểm
trùng giữ bản có timestamp mới nhất; với nhãn trùng sau chuẩn hóa cũng giữ lần
gắn nhãn mới nhất. Nhãn giống nhau của những người dùng khác nhau được giữ lại.
Khi timestamp bằng nhau, giữ dòng xuất hiện sau trong CSV gốc.

ID người dùng/phim là số nguyên dương; điểm hợp lệ 0,5–5, bước 0,5; timestamp
giữ dạng Unix giây UTC, không âm và không sau thời điểm làm sạch. Không xóa
đánh giá thấp/cao hợp lệ, phim ít đánh giá hoặc phim chưa có đánh giá.

IMDb ID giữ số 0 ở đầu. Mã IMDb/TMDB còn thiếu để trống, không điền 0 hoặc mã
phỏng đoán; mã ngoại hỏng được xóa nội dung. `links.csv` trùng movieId ưu tiên
dòng có nhiều mã ngoại hợp lệ hơn, sau đó dòng xuất hiện trước.

Chi tiết số lượng, quy tắc và SHA-256: `cleaning_report.json`.
Kết quả kiểm tra độc lập trên toàn bộ CSV đã lưu: `validation_report.json`.
Đã xác nhận không còn cặp user/phim trùng, nhãn trùng của cùng user/phim hoặc
tham chiếu đến phim không tồn tại. Toàn bộ 25.000.095 đánh giá được giữ lại;
đã loại 26 nhãn rỗng/placeholder và 2 nhãn trùng sau chuẩn hóa. Có 107 mã TMDB
còn thiếu được giữ trống; không tự suy đoán mã thay thế.

Tạo lại từ thư mục project (đích mới phải chưa tồn tại):

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/clean_dataset.py --output ../dataset_cleaned_new
```

Để dùng dữ liệu sạch cho một lần chạy ứng dụng/học máy:

```powershell
$env:MOVIE_DATASET_DIR = '../dataset_cleaned'
```

Website mặc định vẫn dùng cấu hình hiện có. Nếu huấn luyện lại, dùng một thư mục
artifacts riêng cho dữ liệu sạch; không ghi đè các tập train/validation/test cũ.
