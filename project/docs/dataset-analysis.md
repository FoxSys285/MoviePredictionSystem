# Phân tích dataset MovieLens

Phân tích ngày 24/10/2026, đọc toàn bộ bốn CSV trong `dataset/` theo chunk;
không dùng mẫu người dùng để tính các thống kê dưới đây. Có thể chạy lại bằng
`movie-rating analyze` trong môi trường Python của dự án. Báo cáo máy đọc được
lưu ở `artifacts/catalog/summary.json`.

## Quy mô và phạm vi

| Chỉ số | Giá trị |
|---|---:|
| Phim trong movies.csv | 62.423 |
| Lượt đánh giá | 25.000.095 |
| Người dùng có đánh giá | 162.541 |
| Phim có ít nhất một đánh giá | 59.047 |
| Phim chưa có đánh giá | 3.376 |
| Nhãn trong tags.csv | 1.093.360 |
| Dòng liên kết IMDb/TMDB | 62.423 |
| Thể loại đã xác định | 19 |
| Điểm trung bình | 3,53385 / 5 |
| Mật độ ma trận người dùng × phim | 0,2464% |
| Năm phát hành trích được từ tên phim | 1874–2019 |
| Thời gian đánh giá (UTC) | 09/01/1995–21/11/2019 |

Thang điểm là 0,5–5, bước 0,5. Năm phim là năm trong tên phim, không phải
năm người dùng đánh giá. Website phản ánh phạm vi dữ liệu đến năm 2019.

## Cấu trúc và chất lượng

| Tệp | Cột | Cách sử dụng |
|---|---|---|
| movies.csv | movieId, title, genres | Danh mục, tìm kiếm, năm và thể loại |
| ratings.csv | userId, movieId, rating, timestamp | Điểm, mức phổ biến, phân bố và học máy |
| tags.csv | userId, movieId, tag, timestamp | Nhãn cộng đồng trên trang chi tiết |
| links.csv | movieId, imdbId, tmdbId | Liên kết đến phim tương ứng |

Các cột bắt buộc của phim và đánh giá không có giá trị thiếu, movieId trong
movies.csv không trùng, mọi điểm thuộc thang hợp lệ và không có đánh giá tham
chiếu đến phim ngoài danh mục. Có 412 phim chưa trích được năm, 5.062 phim mang
`(no genres listed)` và 25 nhãn rỗng/thiếu. Các nhãn rỗng được bỏ qua khi hiển thị.
Chưa kiểm tra toàn bộ cặp userId/movieId có trùng lặp trong báo cáo danh mục;
pipeline huấn luyện kiểm tra trùng cặp trên dữ liệu được lấy mẫu.

Dataset không chứa ảnh poster, cốt truyện, diễn viên, đạo diễn hay thời lượng.
Website dùng thông tin thật có trong CSV và cung cấp liên kết IMDb/TMDB để
tham khảo thêm. Không suy diễn các trường còn thiếu thành dữ liệu của dataset.

## Phân bố điểm

| Điểm | Số lượt |
|---|---:|
| 0,5 | 393.068 |
| 1,0 | 776.815 |
| 1,5 | 399.490 |
| 2,0 | 1.640.868 |
| 2,5 | 1.262.797 |
| 3,0 | 4.896.928 |
| 3,5 | 3.177.318 |
| 4,0 | 6.639.798 |
| 4,5 | 2.200.539 |
| 5,0 | 3.612.474 |

Điểm 4 xuất hiện nhiều nhất. Ma trận đánh giá thưa khoảng 99,7536%, phù hợp
với cách biểu diễn sparse và các mô hình nhân tố tiềm ẩn; không nên coi phim
chưa đánh giá là điểm 0. Điểm trung bình cao của phim có rất ít lượt đánh giá
cần được đọc cùng số lượt, không đồng nghĩa với mức phổ biến cao.

## Thể loại và phim phổ biến

Sáu thể loại có nhiều phim nhất: Drama (25.606), Comedy (16.870), Thriller
(8.654), Romance (7.719), Action (7.348), Horror (5.989). Một phim có thể thuộc
nhiều thể loại nên tổng các thể loại không bằng số phim.

| Phim | Số lượt đánh giá | Trung bình /5 |
|---|---:|---:|
| Forrest Gump (1994) | 81.491 | 4,048 |
| The Shawshank Redemption (1994) | 81.482 | 4,414 |
| Pulp Fiction (1994) | 79.672 | 4,189 |
| The Silence of the Lambs (1991) | 74.127 | 4,151 |
| The Matrix (1999) | 72.674 | 4,154 |
| Star Wars: Episode IV - A New Hope (1977) | 68.717 | 4,120 |

## Áp dụng vào website và học máy

Home, Gallery và About lấy số liệu từ API, không dùng danh sách phim/điểm mẫu.
Thang điểm hiển thị và bộ lọc đều là /5. Bấm phim mở `/movies/:movieId`, hiển thị
thông tin, số lượt, biểu đồ phân bố điểm, tối đa 12 từ khóa phổ biến và liên kết
IMDb/TMDB. Phim liên quan được chọn bằng số thể loại chung, sau đó mức phổ biến;
đây là quy tắc khám phá danh mục, không phải đầu ra mô hình đã huấn luyện.

Cache danh mục khoảng 5 MB, tách khỏi `artifacts/data/` dành cho học máy. Điểm
cộng đồng dùng toàn bộ `ratings.csv`; các mô hình, bias và thống kê đầu vào học
máy vẫn chỉ dùng train. Các thống kê website không được dùng để tính RMSE/MAE.
Mặc định pipeline huấn luyện vẫn lấy mẫu khoảng 2% người dùng và chia theo thời
gian 80/10/10; phân tích dataset không tự chạy huấn luyện hoặc tạo kết quả giả.
