# MOVIE — giao diện ứng dụng dự đoán điểm phim

React + TypeScript + Vite, triển khai theo 5 ảnh mẫu trong `../../images`.
Nền đen/đỏ, font Bebas Neue + Be Vietnam Pro được lưu cùng ứng dụng,
poster và banner là hình tạo mới. Không cần mạng để tải font/ảnh khi chạy.

## Chạy nhanh toàn ứng dụng

Frontend đã build vào `dist/`. Trong thư mục **project**:

```powershell
.\.venv\Scripts\movie-rating.exe serve
```

Mở <http://127.0.0.1:8000> (mặc định chuyển đến Gallery).
FastAPI phục vụ cả giao diện lẫn `/api/v1`, Swagger ở `/docs`.
Các đường dẫn `/gallery`, `/movies/:movieId`, `/about`, `/contact`, `/login`, `/register`, `/home`
hỗ trợ mở trực tiếp và tải lại trang.

Nếu lấy mã nguồn mới mà chưa có `dist`, vào `project/frontend` và chạy:

```powershell
npm ci
npm run build
```

Sau đó khởi động lại backend.

File `../../run_project.bat` gọi `../scripts/start.ps1`: build source hiện tại,
nhận diện và dừng server cũ của đúng dự án trên cổng 8000, khởi động server mới,
kiểm tra mã phiên server và bundle JavaScript trước khi mở trình duyệt. Nếu
cổng bị ứng dụng khác chiếm hoặc build/start lỗi, hiển thị lỗi và không mở trang.
Chạy lại file bat sẽ thay server của lần chạy trước, không âm thầm dùng lại server cũ.

Trang mới mở có query `launch` riêng để tránh dùng lại tài liệu cũ. FastAPI trả
HTML với Cache-Control: no-store. Tab đã mở từ trước vẫn cần tải lại trang;
chuyển menu trong SPA không tự thay mã JavaScript đang chạy. Giữ cửa sổ chạy
ứng dụng mở; Ctrl+C dừng server của cửa sổ đó. Thông tin lần chạy được lưu tại
`../artifacts/last-launch-8000.json`, gồm URL, PID, thư mục dự án và bundle đã xác minh.

Kiểm tra khởi động không tự mở trình duyệt (trong project):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start.ps1 -NoBrowser -Detached
```

`-Detached` giữ server chạy sau khi script kết thúc; bình thường không cần tham số này.

## Phát triển giao diện

Terminal 1, trong `project`:

```powershell
.\.venv\Scripts\movie-rating.exe serve
```

Terminal 2, trong `project/frontend`:

```powershell
npm ci
npm run dev
```

Mở <http://127.0.0.1:5173>. Vite chuyển tiếp `/api` đến backend cổng 8000,
bao gồm cookie phiên đăng nhập. `MOVIE_API_PROXY` có thể đổi backend proxy.
Không cần `.env` cho cấu hình mặc định.

## Các trang và tương tác

| Trang | Nội dung |
|---|---|
| `/gallery` | Banner, tìm kiếm, thể loại, năm, điểm, sắp xếp, phân trang và chi tiết phim |
| `/movies/:movieId` | Trang chi tiết riêng: thể loại, điểm /5, số đánh giá, phân bố điểm, nhãn, IMDb/TMDB và phim cùng thể loại |
| `/about` | Giới thiệu, tính năng, ảnh rạp, bốn vai trò trong nhóm |
| `/contact` | Thông tin liên hệ, kiểm tra form và lưu tin nhắn vào SQLite |
| `/login` | Đăng nhập email/mật khẩu, hiện/ẩn mật khẩu, cookie phiên |
| `/register` | Đăng ký, kiểm tra xác nhận mật khẩu, tạo phiên đăng nhập |
| `/home` | Trang chủ bổ sung theo cùng phong cách và lối vào bộ sưu tập |
| `/watch-history` | Phim tài khoản đã đánh giá, điểm riêng, ngày cập nhật, sửa/xóa và phân trang |
| `/genres` | Danh sách thể loại thật; chọn để chuyển Gallery với bộ lọc tương ứng |
| `/experiments` | Khu vực thực nghiệm đã được làm trống, chờ danh sách thuật toán và thiết kế đánh giá mới |

Menu dùng nhãn **Genres**. Nút bình thí nghiệm nổi ở góc dưới phải mở `/experiments`;
nút lên đầu trang được đặt phía trên để không chồng lấp. Menu thu gọn trên màn hình hẹp.
Trang Experiments hiện không hiển thị mô hình, số liệu hoặc kết luận cũ. Nội dung mới sẽ
được bổ sung sau khi danh sách thuật toán được chốt.

Home và Gallery có hàng "Các phim có thể bạn yêu thích", cuộn ngang tối đa 20 phim.
Home có thêm 5 hàng thể loại, tối đa 10 phim mỗi hàng; nút Xem tất cả mở Gallery
đã lọc, nút Tất cả thể loại ở hàng đầu mở `/genres`. Số phim có thể ít hơn giới hạn
nếu catalogue không đủ; không lặp hoặc thêm phim mẫu để đủ số.

Trang chi tiết có chọn sao, điểm nửa sao qua select và lưu/cập nhật đánh giá.
Guest chọn đánh giá sẽ nhận thông báo và liên kết đăng nhập có đường dẫn quay lại.
Watch History yêu cầu đăng nhập; chỉ chứa những phim đã chấm điểm, không tự ghi
lịch sử khi bấm mở phim. Đăng nhập/đăng ký giữ tham số `next` nội bộ an toàn.

Gợi ý cá nhân lấy lịch sử tài khoản thật: khi có model đã train thì có điểm dự đoán
và tên mô hình; khi chưa train thì dùng hồ sơ thể loại từ rating và ghi nhãn nguồn.
Guest/user chưa có lịch sử nhận gợi ý cộng đồng. Những phim đã đánh giá được loại
khỏi hàng gợi ý. Không tạo predicted_rating giả hoặc đồng nhất app user ID với MovieLens.
Phần Experiments trên giao diện đã được làm trống để xây dựng lại. Mô tả thuật toán và
thiết kế đánh giá mới sẽ được bổ sung sau khi nhận danh sách thuật toán.

Nhấn poster hoặc tên phim để chuyển đến `/movies/:movieId`. Trang này có thể
mở trực tiếp, tải lại, dùng nút Back; bộ lọc Gallery được giữ trong URL khi quay lại.
Trang chi tiết không mở form hoặc bảng mô hình. Bánh răng mở riêng thông tin bốn
thuật toán và kết quả validation của run mới nhất nếu đã huấn luyện.

Danh mục và thống kê lấy từ **toàn bộ CSV** trong `dataset/`, hoạt động trước khi
huấn luyện. Thống nhất thang điểm 0,5–5 trên card, bộ lọc và trang chi tiết;
phim chưa có đánh giá hiển thị trạng thái trống. Không dùng điểm mẫu hoặc phim
minh họa thay thế khi API lỗi. Trang Home hiển thị số liệu thực và sáu phim
được đánh giá nhiều nhất; About trình bày phân tích dataset.

Dataset không chứa poster hoặc cốt truyện. Chỉ dùng tranh minh họa có ID phim
khớp với dataset; phim khác dùng pattern SVG với 8 bảng màu và 6 họa tiết hình
học. Phối màu và họa tiết được chọn cố định theo ID phim nên card và trang chi
tiết nhất quán. Nếu ảnh tải lỗi, component cũng tự chuyển sang pattern. Không
cần mạng, sinh ảnh hoặc lưu hàng nghìn poster mới. Thông tin bổ sung có thể xem
qua liên kết IMDb/TMDB từ `links.csv`.

## Tương tác và chuyển động

Card có ánh sáng theo vị trí chuột, nghiêng nhẹ, phóng ảnh/họa tiết và nút xem
chi tiết khi hover hoặc focus bàn phím. Nội dung xuất hiện theo cuộn với
IntersectionObserver dùng chung; header bám trên cùng, thanh tiến độ cuộn và
nút lên đầu trang hỗ trợ duyệt trang dài. Chuyển trang có hiệu ứng xuất hiện;
Back phục hồi vị trí cuộn, phân trang cuộn về bộ lọc.

Gallery có skeleton khi tải, nhãn bộ lọc có thể xóa riêng và phím `/` để tìm
kiếm (không can thiệp khi đang nhập form). Menu tài khoản đóng khi bấm ngoài
hoặc Escape. Hiệu ứng tôn trọng `prefers-reduced-motion`; thao tác cảm ứng
không dùng hiệu ứng nghiêng hay yêu cầu hover.

Ảnh nhóm là chân dung AI hư cấu, hiển thị vai trò thay vì tự đặt tên thành viên thật.
Liên kết mạng xã hội và Google/Facebook thông báo chưa kết nối. Form liên hệ lưu
cục bộ, chưa gửi email ra bên ngoài. Tài khoản và tin nhắn lưu tại
`project/artifacts/application.sqlite3`; mật khẩu băm có salt, cookie HttpOnly.
Luồng tài khoản cục bộ chưa gồm khôi phục mật khẩu hay xác minh email.

## Cấu trúc mã

```text
src/
├── App.tsx                  # Routing, phiên, thống kê dataset và thông báo
├── pages/                   # Gallery, Home, About, Contact, Auth
├── components/              # Layout, card phim, chi tiết, bảng thuật toán, UI chung
│   ├── MovieArtwork.tsx     # Ảnh và fallback khi thiếu hoặc tải lỗi
│   ├── PosterPattern.tsx    # Họa tiết SVG tạo trong ứng dụng
│   └── Experience.tsx       # Chuyển trang, cuộn, phím tắt và phục hồi vị trí
├── data/movies.ts           # Kiểu danh mục thật, tên thể loại và ánh xạ tranh minh họa
├── data/posterPatterns.ts   # Phối màu và họa tiết cố định theo phim
├── lib/api.ts               # HTTP client và kiểu dữ liệu
├── lib/motion.ts            # Observer dùng chung, hiệu ứng chuột và cuộn
├── styles.css               # Tokens, layout và responsive
├── experience.css           # Pattern, hover, animation và reduced motion
└── main.tsx                 # React root, router và font cục bộ
public/assets/               # Ảnh WebP dùng trong giao diện
scripts/prepare-assets.mjs   # Chuyển PNG gốc sang WebP
scripts/check-ui.mjs         # Kiểm tra bằng Chrome/Edge cài sẵn
```

Nguồn ảnh: `images/generated/` ở gốc workspace.
Prompt và phương thức tạo ảnh: [prompts.json](../../images/generated/prompts.json).
Năm ảnh UI/UX gốc được giữ nguyên.

## Kiểm tra

```powershell
npm run build
npm run test:ui
```

`test:ui` tự mở hai tiến trình kiểm thử, dùng dữ liệu tài khoản/tin nhắn trong thư mục
tạm rồi dừng các tiến trình. Cần Python trong `project/.venv` và Chrome hoặc Edge
cài sẵn; đặt `CHROME_PATH` khi cài ở vị trí khác. Không tải trình duyệt mới.
Kiểm tra các trang ở chiều rộng 360, 390, 768 và chụp desktop 1440px; thử đọc CSV
giả lập trong thư mục tạm, tìm/lọc phim, mở và tải lại trang chi tiết, giữ bộ lọc
khi quay lại, phim chưa có điểm, phim không tồn tại, bảng mô hình, liên hệ và tài khoản.
Kiểm tra bổ sung ảnh lỗi chuyển sang pattern, biến thể màu, phím tắt tìm kiếm,
skeleton, hover theo chuột, reveal khi cuộn, thanh tiến độ, nút lên đầu trang,
chuyển trang, phục hồi vị trí cuộn và chế độ giảm chuyển động.
Ảnh chụp lưu trong `test-results/` (không đưa vào Git).
