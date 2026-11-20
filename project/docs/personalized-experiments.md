# Hồi quy, phân loại và giới hạn RAM

## Chạy trên dữ liệu hiện tại

Mở lại `run_project.bat`, vào Thực nghiệm và giữ **Dùng toàn bộ dữ liệu đã chuẩn bị**.
Hồi quy/phân loại/phân cụm tự chuyển dữ liệu chia thời gian toàn cục cũ sang cache riêng,
không xóa data cũ hoặc lịch sử. Không cần chạy lại `prepare` để dùng Lab mới.
Luật kết hợp, giảm chiều và học tăng cường tiếp tục đọc dữ liệu chuẩn bị của chúng.

Giao thức mới: chia lịch sử trong từng user, khoảng 70/15/15, không tách rating
cùng timestamp. Có thể có chồng lấn timeline giữa các user; không diễn giải kết
quả này như benchmark tương lai toàn cục. User cần ít nhất 10 rating và ba đoạn
thời gian tách biệt. Số bị loại được ghi trong JSON và giao diện kết quả.

Cache xác nhận trên dữ liệu hiện tại: đầu vào 25.000.095 rating / 162.541 user;
loại 120.669 rating / 3.185 user do điều kiện lịch sử/timestamp. Còn 24.879.426
rating: Train 17.413.070, Validation 3.722.738, Test 3.743.618. Không lấy mẫu user
trong chế độ toàn bộ; số bị loại không bị giấu trong thống kê nguồn.

## Mô hình và tuning

- XGBoost: 150 cây, depth 6, learning rate 0,05.
- SVD: 32 factors, 15 epochs, learning rate 0,005.
- NCF: 32 factors, tối đa 15 epochs, learning rate 0,003, patience 3.
- Forest: 100 cây, depth 8, min_samples_leaf 5.
- Logistic: C=1, max_iter 600, SAGA, float32 đã chuẩn hóa.

Tuning là tùy chọn; thử cấu hình nhẹ/mặc định/mạnh hơn, không chạy song song.
Logistic thử C bằng 0,1 / 1 / 10 khi C gốc bằng 1. SVD/NCF thử 16 / 32 / 64
factors khi factors gốc bằng 32. XGBoost/Forest thử cây nông hơn/sâu hơn và tăng
số cây ở ứng viên mạnh. Validation chọn RMSE thấp nhất hoặc F1 lớp Thích cao nhất.
Ngưỡng phân loại được chọn trên Validation, không trên Test.

Chỉ cấu hình được chọn chạy ba lượt CV mở rộng lịch sử trong từng user trên
Train; từng lượt fit lại đặc trưng, chỉ dùng quá khứ để chấm đoạn sau. CV là
chẩn đoán sau lựa chọn, không phải nested CV. Sau đó refit Train + Validation,
đánh giá Test một lần; không chọn lại mô hình bằng Test. Không so trực tiếp các
chỉ số với báo cáo cũ dùng IID hoặc thời gian toàn cục.

Đặc trưng mới: tỷ lệ thích và độ phân tán rating theo user/phim, sở thích từ 20
rating gần nhất, cùng các thống kê và thể loại trước đây. Hàng huấn luyện loại
đóng góp nhãn của chính hàng đó; thống kê không đọc rating held-out. Genre tĩnh
của phim mới có thể dùng, nhưng lịch sử rating của phim đó không được dùng.

## Card trạng thái và thời gian chạy

Khi bấm Chạy, card xuất hiện ngay với đồng hồ `00:00`, tăng mỗi giây theo dạng
`mm:ss` (chuyển sang `hh:mm:ss` từ một giờ). Card vàng khi đang gửi/chờ/chạy;
chuyển xanh lá khi thành công. Lỗi có card đỏ, dừng/gián đoạn có màu trung tính.
Đồng hồ dừng theo `finished_at`, giữ nguyên khi tải lại trang hoặc xem lịch sử;
lượt cũ thiếu mốc kết thúc hiển thị `--:--`. Thời gian tính từ lúc gửi yêu cầu,
bao gồm chuẩn bị dữ liệu, CV và refit, không chỉ thời gian fit mô hình.

## Nhật ký hồi quy và phân loại

Khung **Nhật ký thực nghiệm** nằm ngay dưới thanh tiến độ, cao 50% viewport,
có thanh cuộn. Tự cuộn theo dòng mới khi đang ở cuối; cuộn lên để đọc lịch sử
thì giữ vị trí. Nút **Về log mới nhất** bật theo dõi lại. Hoạt động cả desktop,
mobile và giao diện sáng/tối. Hồi quy, phân loại, phân cụm và luật kết hợp dùng chung khung này.

Log ghi thời gian, cấu hình, tạo đặc trưng, Validation, từng lượt CV, refit và
Test. Linear ghi số rating đã tích lũy theo batch; SVD/NCF ghi epoch (NCF có
loss, Validation RMSE và early stopping). XGBoost ghi từng lượt tạo cache
quantile, số rating xử lý và từng cây đang học. Logistic ghi bước scale/SAGA
và số vòng tối ưu sau fit; Forest ghi cấu hình và số cây sau fit. Worker lớn
có thông báo còn chạy/RAM mỗi 30 giây. Lỗi và yêu cầu dừng được ghi riêng.

Mỗi run lưu `artifacts/experiment-lab/<run_id>/events.jsonl`; mở lại lịch sử
hoặc tải bằng nút **Tải log**. Backend đọc theo byte cursor, tối đa 500 dòng
mỗi request; UI chỉ giữ 5.000 dòng gần nhất để giới hạn RAM. File tải giữ toàn
bộ nhật ký. Khóa file bảo vệ các lần ghi đồng thời từ worker và supervisor.

Áp dụng từ các lượt chạy mới sau khi khởi động lại backend. Không tái tạo log
cho lượt chạy đã bắt đầu trước cập nhật. Nếu đang có thực nghiệm dài, chờ xong
rồi mở lại `run_project.bat` để tránh làm gián đoạn lượt đó.

Kiểm tra giao diện: `npm run test:experiment-logs` trong `frontend/`; chạy thật
hai nhóm và kiểm tra chiều cao, scroll/follow, reload lịch sử, tải log, kết nối
lại và việc các nhóm khác chưa hiển thị log.

## Học tăng cường trong Experiment Studio

Nhóm mới dùng ID `dqn` cho policy Q-Learning xấp xỉ bằng mạng nhỏ; `q_learning`
chỉ còn trong kết quả lịch sử. Loader của RL dùng split thời gian trong từng
user giống các thực nghiệm cá nhân hóa. Mỗi episode Train tách support/query theo
timestamp: profile ban đầu chỉ đọc support và reward lấy từ query. Nếu không thể
tạo hai đoạn thời gian hợp lệ, user được loại và số lượng/lý do xuất hiện trong
protocol.

DQN nhận profile sở thích/tần suất, độ đầy đủ lịch sử, action mask, số bước còn
lại và phản hồi quan sát được trước đó. Mạng không nhận user ID. Action là một
phim trong catalog phổ biến từ Train; phim support và phim đã chọn bị mask.
Target network, replay buffer hữu hạn, Huber loss, gradient clipping và epsilon
schedule có cấu hình riêng. Random và Popularity dùng lại đúng episode, catalog
và mask để làm baseline.

Reward mô phỏng từ rating thực, `(rating - 2,75) / 2,25`. Hành động không có
rating quan sát nhận 0 theo quy ước, nên không được diễn giải là dislike hoặc
ước lượng counterfactual. Báo cáo giữ tỷ lệ hành động có phản hồi, coverage,
ranking và bất định ghép cặp cùng average reward. Validation chọn checkpoint;
Test dùng snapshot đóng băng, không cập nhật trọng số hay reducer.

`reduction_mode=compare` so DQN raw và nén trên cùng cohort/episode IDs, seed,
catalog và training-step budget. Reducer chỉ fit profile Train/support; action
mask, số bước, độ đầy đủ lịch sử và observed flag giữ nguyên. Random/Popularity
không tạo bản sao giả chỉ để gắn nhãn representation. Metadata variant lưu
phương pháp, số chiều, feature family, fit scope, fingerprint và model size.

## Giảm chiều theo adapter

`off` giữ thuật toán gốc; `enabled` chạy representation được chọn; `compare`
thêm raw control. PCA/TruncatedSVD fit trong tập Train hoặc fold tương ứng và
Validation chọn variant theo metric tác vụ. TruncatedSVD chiếu không center;
explained variance là `Var(XV)` trên tổng phương sai của đầu vào đã scale,
không phải tỷ lệ năng lượng `E[(XV)^2]`. EVR/reconstruction chỉ dùng chẩn
đoán. Adapter hồi quy/phân loại nén block tương tác user–genre nhưng giữ bias
và thống kê nền; Funk SVD/NCF dùng nhánh hybrid để profile nén thực sự tham gia
dự đoán. Phân cụm fit lại K/eps trong không gian mới. Profile genre gốc vẫn dùng
để diễn giải cụm.

Apriori không giảm chiều ma trận transaction. Biến thể mới fit reducer trên
profile user Train, phân nhóm user rồi khai phá luật riêng từng nhóm; luật toàn
cục vẫn là baseline. Support trong nhóm và support toàn cục được báo riêng.
Nhóm thiếu luật rơi về luật toàn cục rồi Popularity. Ranking dùng cohort tương
lai cố định, báo candidate/group coverage, fallback và positive ngoài catalog.

## Bộ nhớ

- Linear giải OLS từ các tổng và tích ma trận nhỏ theo batch, không tạo ma trận
  đặc trưng toàn bộ dataset. Dùng số học float64 cho các thống kê OLS.
- Logistic/Forest tạo đặc trưng float32 theo batch 65.536 hàng; ma trận trên
  64 MiB dùng memmap trên đĩa, không giữ bản sao của ma trận Validation/Test.
  Logistic scale theo batch; Forest dùng một luồng fit cây để giảm peak RAM.
- Không fit một hồi quy phụ chỉ để tạo đặc trưng phân loại.
- XGBoost từ 250.000 hàng dùng CPU histogram và external-memory quantile pages
  trên XGBoost >=3.0. Cài đặt hiện tại 3.4.1 hỗ trợ đường chạy này. Bản cũ hơn
  dùng QuantileDMatrix theo batch nhưng không có cùng bảo đảm lưu trang ra đĩa.
- NCF giữ chỉ số/nhãn ở CPU và chỉ chuyển minibatch lên GPU. Train-rating MSE
  trên đồ thị dùng tối đa 50.000 hàng cố định; Validation vẫn chấm toàn bộ.
- Thống kê catalog đọc Parquet theo batch; không concat toàn bộ rating chỉ để
  trả thống kê. Không giữ đồng thời mô hình của các fold/cấu hình đã chấm.

Khi dữ liệu nguồn từ 1 triệu rating, Lab chạy job trong tiến trình riêng và
giám sát mỗi 0,25 giây. Ngân sách mặc định 4096 MiB cho bộ nhớ riêng (USS), giảm
theo RAM đang khả dụng và không quá nửa RAM máy. Giữ ít nhất 12% RAM máy, tối
thiểu 1 GiB, cho hệ thống. Vượt ngân sách hoặc thiếu RAM thì dừng worker, không
công bố kết quả dở; ứng dụng vẫn đọc được lỗi và có thể chạy lại. Đây là cơ chế
giám sát, không phải hard limit RAM của Windows, nên có thể vượt ngắn giữa hai
lần lấy mẫu. RSS còn gồm các trang file memmap/cache có thể thu hồi.

File tạm nằm trong thư mục scratch riêng, được dọn khi hoàn tất/lỗi/dừng. Trên Windows, XGBoost cần đường dẫn cache không dấu; tự dùng short path hoặc thư mục tạm không dấu, có thể đặt `MOVIE_SCRATCH_DIR` để chọn ổ lưu cache. Cần
đủ dung lượng ổ đĩa (khoảng 6 GiB cho đặc trưng phân loại 21 triệu hàng, ngoài
cache dữ liệu). Không tăng ngân sách RAM để ép chạy khi đang mở nhiều ứng dụng.

## Kiểm tra có thể lặp lại

Đã chạy Linear toàn bộ dữ liệu đủ điều kiện, job `20261117T063322-aee23787`:

| Chỉ số | Giá trị |
|---|---:|
| Rating dùng trong thực nghiệm | 24.879.426 |
| RMSE Validation | 0,872195 |
| RMSE Test sau refit | 0,876566 |
| MAE Test | 0,665315 |
| Refit Train + Validation | 61,48 giây |
| Peak RAM riêng của worker (USS, cả lần chạy) | 2.468,0 MiB |
| RAM hệ thống khả dụng thấp nhất (lấy mẫu) | 3.702,8 MiB |

Nguồn: `artifacts/experiment-lab/20261117T063322-aee23787/result.json`. Peak đo cả
chuẩn bị dữ liệu/CV/refit; USS và RSS là hai cách đo khác nhau; USS cũng có thể tính trang file chưa chia sẻ. Kết quả trên máy
16 GiB, không bảo đảm mọi thuật toán/cấu hình đều nằm trong cùng mức RAM. Các
mô hình còn lại cần đo riêng; giới hạn worker sẽ dừng nếu không đủ ngân sách.
Không so mức cải thiện trực tiếp với RMSE của giao thức cũ.

Đường XGBoost external-memory cũng đã chạy trên 300.000 rating thật, 10 cây,
depth 2: fit 3,71 giây, RSS đỉnh 691,52 MiB. Đã sửa lỗi cache với đường dẫn
Windows có dấu và xác nhận file tạm được dọn. Đây là kiểm tra đường chạy,
không phải kết quả benchmark XGBoost toàn bộ dữ liệu.

Kiểm thử backend: 128 passed, 1 skipped (phụ thuộc báo cáo tùy chọn chưa cài).
Kiểm tra bổ sung bao gồm timestamp trùng nhau, held-out không đổi cấu hình,
OLS theo batch so với OLS dense, memmap được dọn, XGBoost external-memory và
tiến trình thật bị dừng khi vượt giới hạn RAM giả lập. Frontend đã build và kiểm tra UI sáu nhóm thành công. XGBoost external-memory được chạy thêm trên 300.000 rating thật: fit 3,71 giây, peak RSS khoảng 692 MiB; đã kiểm tra đường dẫn chứa tiếng Việt trên Windows.

Từ thư mục `project`:

```powershell
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe scripts/benchmark_supervised_memory.py --full-data --algorithms linear
.venv/Scripts/python.exe scripts/benchmark_supervised_memory.py --group classification --algorithms logistic random_forest --max-users 150 --tune
```

Script benchmark dùng cùng Lab worker với giao diện, lưu kết quả thật vào lịch
sử `artifacts/experiment-lab/`, gồm peak USS, ngân sách hiệu lực và RAM khả dụng
thấp nhất. Các lệnh giới hạn user là kiểm tra hoạt động, không benchmark 25M.
