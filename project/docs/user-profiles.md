# Người dùng mới và hồ sơ có sẵn trong dataset

## Đăng nhập thử

Trang Login hiển thị 5 tài khoản ứng dụng liên kết với 5 userId hợp lệ nhỏ nhất
trong dataset đang cấu hình. Nhấn **Dùng tài khoản này**, rồi **Đăng nhập**.
Email hiển thị dạng `movielens{userId}.{mã nguồn}@demo.movie.local`;
mật khẩu demo chung là `MovieLens@2026`.

MovieLens ẩn danh, không cung cấp tên, email hay mật khẩu gốc. Đây là tài khoản
demo do ứng dụng tạo, dùng chung cho thực nghiệm. Các tài khoản đăng ký thông
thường không được đưa vào danh sách này. Tên miền demo được dành riêng.

Lần mở Login đầu tiên đọc ratings.csv theo chunk, giữ toàn bộ lịch sử của 5 user
được chọn, chỉ nhận phim có trong movies.csv và điểm/timestamp hợp lệ. Nếu một
user đánh giá cùng phim nhiều lần, giữ lần mới nhất. Timestamp giây chuyển sang
millisecond cho website. Không phụ thuộc việc CSV đã được sắp xếp theo user hay chưa.
Dataset nhỏ hơn 5 user hiển thị số user thực có.

Lịch sử được nhập một lần vào SQLite, ánh xạ tường minh bằng bảng dataset_profiles.
Không sửa CSV gốc hoặc dataset_cleaned. Mở lại Login hay khởi động lại backend
không nhập lại phim đã xóa, không ghi đè điểm đã sửa. Thay dataset/path/metadata
CSV sẽ tạo nhóm tài khoản demo mới theo dấu nhận dạng nguồn; giữ nguyên nhóm cũ.
Đánh giá sửa/xóa/thêm trên một demo sẽ được người cùng dùng tài khoản đó nhìn thấy.

Có thể chuẩn bị trước khi mở website, chạy trong project:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/seed_demo_accounts.py
```

Không cần chạy script này mỗi lần; API GET /api/v1/auth/demo-accounts tự chuẩn bị
nếu chưa có. Thiếu dataset không chặn đăng ký/đăng nhập thông thường.

## Quyết định cách gợi ý

| Tình huống | Cách xử lý | strategy trong API |
|---|---|---|
| Khách hoặc tài khoản mới, chưa có đánh giá | Xếp phim theo prior rating cộng đồng có regularization | community_popularity |
| Có lịch sử, chưa có mô hình đã train | Hồ sơ thể loại có trọng số rating − 3, loại phim đã đánh giá | genre_profile |
| Demo chưa thay đổi, đúng nguồn dataset, userId nằm trong train | Gọi model.predict với userId MovieLens được ánh xạ; dùng bias/factors đã học | trained_dataset_user |
| Có lịch sử và model, nhưng là tài khoản mới, demo không có trong mẫu train, hoặc demo đã sửa lịch sử | Fold-in từ điểm hiện tại, giữ nguyên tham số phim đã học | profile_fold_in |
| Lịch sử không có phim model đã học | Fallback theo thể loại/phổ biến; model_status=no_known_history_items | genre_profile hoặc community_popularity |

Fold-in hỗ trợ Ridge, XGBoost, SVD, NCF và ba mô hình tham chiếu Baseline Bias,
Item-KNN, NMF. Ridge/XGBoost tính lại đặc trưng hồ sơ từ rating thật; SVD giải
factor/bias user mới; NCF tối ưu embedding/bias riêng, giữ mạng và item cố định.
Mô hình chỉ sẵn sàng khi đã có artifact của run hoàn tất.
Không có tín hiệu thể loại (ví dụ mọi điểm đều bằng 3) sẽ dùng prior cộng đồng.

App user ID và MovieLens userId là hai không gian khác nhau. Tài khoản mới có
app ID=1 không được kế thừa lịch sử/factors của MovieLens userId=1. Mọi chỉnh sửa
được đọc từ tài khoản của phiên đăng nhập, không nhận userId từ trình duyệt để chọn hồ sơ.

Gợi ý tối đa 20 phim chưa chấm, tính lại sau mỗi đánh giá sửa/thêm/xóa. Giao diện
ghi rõ nguồn hồ sơ và cách gợi ý. Chỉ mode=model có predicted_rating trên thang /5;
content/popular trả null, không tạo điểm dự đoán giả.

Combobox luôn hiển thị bốn mô hình chính và Auto; mô hình chưa train bị vô hiệu hóa.
Tài khoản trống vẫn đổi mô hình được khi đã có artifact, nhưng kết quả giữ mode=popular.
Giao diện giải thích và mời chấm 5 phim; không bắt buộc đủ 5 mới cá nhân hóa.
Sau lần đánh giá đầu, các mô hình dùng lịch sử hỗ trợ nếu có phim đã biết.
Auto chọn theo NDCG@10 validation nếu có; fallback RMSE với run cũ/không có ranking.

## Giới hạn của minh họa

Hồ sơ demo nhập toàn bộ lịch sử có sẵn để thử tương tác; không phải một lần đánh
giá offline độc lập. Với nhánh trained_dataset_user, điểm dự đoán dựa trên phần
train mà mô hình đã học, toàn bộ lịch sử hiện có dùng để loại phim đã chấm.
Với fold-in, các điểm hiện tại là dữ liệu hỗ trợ cho hồ sơ mới.

Để so sánh độ chính xác, phải tách support/query theo thời gian, không dùng nhãn
query để fold-in hay xây feature. Đánh giá cold-user ở các mức 0/5/10/20 rating;
đánh giá known-user bằng train/validation/test và quy trình trong
[experimental-protocol.md](experimental-protocol.md). Metric của run gốc không
đại diện mặc nhiên cho chất lượng của adapter fold-in hay tài khoản demo đã sửa.
