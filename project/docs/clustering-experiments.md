# Phân cụm user có lịch sử

Giao thức `past_taste_full_train_future_ranking_v2`: fit cuối trên toàn bộ hồ sơ Train đủ lịch sử.
Mở lại backend khi không còn thực nghiệm đang chạy, tải lại trang Thực nghiệm và
chọn Phân cụm. Không cần prepare lại: nhóm này dùng chung cache chia lịch sử
trong từng user 70/15/15 với hồi quy/phân loại, không sửa dữ liệu/lịch sử cũ.


## Các file được dùng

Khi bật toàn bộ dữ liệu trên workspace hiện tại:

| File | Vai trò |
| --- | --- |
| `artifacts/data-personalized-v2-c9ba624d61cd/train.parquet` | Tất cả 17.413.070 rating Train tạo 159.356 hồ sơ user; cả hai thuật toán fit cuối trên toàn bộ hồ sơ đủ điều kiện. |
| `artifacts/data/movies.parquet` | Metadata thể loại phim, không phải nhãn tương lai. |
| `artifacts/data-personalized-v2-c9ba624d61cd/validation.parquet` | Đánh giá gợi ý/chọn thuật toán. |
| `artifacts/data-personalized-v2-c9ba624d61cd/test.parquet` | Đánh giá cuối. |

Tên cache phụ thuộc chữ ký dữ liệu; JSON `provenance.split_files` và
`provenance.metadata_file` ghi đường dẫn thật của từng lượt. Bảng báo cáo có
`fit_users`, `fit_fraction`, `train_ratings` để kiểm tra mức dùng Train. Ngưỡng
lịch sử vẫn áp dụng, số bị loại công bố rõ. Giới hạn 2.000 user Recall/NDCG và
Silhouette là giới hạn **đánh giá**, không phải giới hạn dữ liệu huấn luyện.

## Hồ sơ sở thích chỉ đọc Train

Với user u, thể loại g, sở thích tương đối là:

`(tổng rating thể loại - số rating thể loại × rating trung bình user) / (số rating thể loại + shrinkage)`.

Mặc định shrinkage = 10. Thể loại chưa xem có điểm 0. Cộng cùng một số vào toàn bộ
rating của một user không đổi vector sở thích. Ghép thêm tần suất xem thể loại,
làm trơn về tần suất chung Train. Chuẩn hóa có sàn độ lệch chuẩn 0,25 cho sở thích
và 0,1 cho tần suất, rồi nhân khối tần suất 0,5 để thể loại hiếm không chi phối.
Scaler chỉ fit trên nhóm user dùng fit trong Train. Loại user dưới ngưỡng ratings
Train được công bố riêng. Không tạo hồ sơ user từ mẩu rating Validation/Test.

## Chọn cấu hình và báo cáo hình học

Tách ngẫu nhiên user từ hồ sơ Train (seed 42): giữ lại khoảng 20%, tối đa 2.000,
toàn bộ phần còn lại dùng khảo sát cấu hình, không cắt ở 20.000 user. Cùng một user giữ lại không nằm
trong mẫu fit ứng viên. Đây là giữ lại user để kiểm tra khả năng gán hồ sơ cùng
chất lượng lịch sử; không phải đánh giá dự đoán sở thích tương lai.

K mặc định 2/3/4/6/8/10. DBSCAN tự chọn eps từ phân vị 50/75/90/98% khoảng cách
láng giềng thứ min_samples trên toàn bộ user fit, gồm chính điểm đó; mặc định min_samples
5/10. Có thể tắt tự động và nhập eps thủ công. Mô hình cần có ít nhất hai cụm hợp
lệ và bao phủ ít nhất 80% user giữ lại trước khi so Silhouette. Khi không cấu
hình nào đạt, vẫn báo cấu hình chẩn đoán ưu tiên có hai cụm hợp lệ rồi bao phủ,
nhưng đánh dấu chưa đủ điều kiện. Sau fit cuối, kiểm tra lại điều kiện trên toàn
bộ hồ sơ Train: DBSCAN có thể hợp nhất thành một cụm, khi đó vẫn không đạt.

Silhouette loại noise, lấy mẫu tối đa 2.000 user; DBI cũng loại noise. Bảng chính
chỉ hiển thị hình học trên lịch sử Train, cùng bao phủ, số user gán/tổng đủ lịch
sử, cụm nhỏ nhất và điều kiện chọn. Cỡ từng cụm có bảng và biểu đồ riêng. Không
gắn nhãn xanh Silhouette/DBI nếu cấu hình chưa đạt điều kiện bao phủ/số cụm.

Kiểm tra ổn định: fit lại trên 80% user (không giới hạn 20.000) với seed 43/44, so nhãn với
mô hình chính trên tối đa 2.000 user. ARI chỉ tính user cùng được gán cụm và có
ít nhất hai cụm ở cả hai phía; báo tỷ lệ bao phủ chung và số lượt so hợp lệ.

## Gợi ý theo cụm và rating tương lai

Dùng số lượt thích (rating ≥3,5) từng phim trong từng cụm trên Train, làm trơn
về mức phổ biến chung với trọng số 20 user. Xếp hạng toàn bộ phim đã xuất hiện
trong Train; loại phim user đã đánh giá trong Train. DBSCAN noise dùng Popularity
dự phòng. Baseline Popularity được học từ cùng user đủ lịch sử, trên cùng Train.

Cohort chọn cố định seed 42, tối đa 2.000 user (cấu hình được, tối đa 5.000) để giữ
RAM. Cả hai thuật toán và baseline chấm đúng cùng các user có ít nhất một phim
tương lai thích, chưa xem trong Train. Phim chỉ xuất hiện tương lai vẫn ở mẫu số
Recall và IDCG, nên tính là miss; không âm thầm bỏ phim mới. Báo số user thực sự
chấm trên từng tập. Recall@10/NDCG@10 lấy trung bình theo user, thêm điểm riêng
user có cụm để không nhầm hiệu quả fallback với hiệu quả phân cụm.

Giữ nguyên lịch sử/mô hình Train để chấm Validation và Test. Cấu hình từng thuật
toán chọn bằng hình học giữ lại/bao phủ; trong các thuật toán đạt điều kiện,
so NDCG@10 Validation với baseline. Test chỉ báo cáo cuối. Đây là đánh giá offline
trên phản hồi quan sát, không suy ra chất lượng online hay mức thích của phim
chưa được chấm. Kết quả này không tự thay đổi mô hình phục vụ ở trang dự đoán.

## Bộ nhớ, xuất dữ liệu và kiểm chứng

- Tích lũy thống kê theo batch, không merge/explode toàn bộ 25 triệu rating.
- Chỉ giữ ma trận user–genre float32, không tạo user–movie dense.
- Trên 20.000 user, K-Means dùng MiniBatchKMeans.partial_fit, đi qua mọi user trong
  mỗi lượt, tổng cộng ba lượt. Không dùng một tập con user thay Train.
- DBSCAN xác định core và nối cụm trên toàn bộ Train. Truy vấn theo batch 512,
  giữ k láng giềng/điểm để xác định core; duyệt cây của các core chưa gán và xây
  lại khi còn dưới một nửa. Không giữ toàn bộ đồ thị khoảng cách trong RAM.
- Thống kê cụm–phim lớn dùng ma trận sparse; chỉ giữ danh sách gợi ý cần thiết cho cohort.
- Khảo sát tuần tự, bỏ mô hình ứng viên sau khi chấm; giữ mô hình cuối từng thuật
  toán đến khi chấm xong rồi giải phóng. Worker vẫn chịu giám sát RAM như các nhóm khác.
- JSON giữ toàn bộ báo cáo. CSV có `record_type=future_recommendation`; Excel
  thêm sheet `Future ranking` và `Cluster search`. Nhật ký từng bước lưu tại `events.jsonl` của lượt chạy.

Chạy lại benchmark theo cùng cơ chế worker/giới hạn RAM của giao diện:

```powershell
.venv/Scripts/python.exe scripts/benchmark_supervised_memory.py --group clustering --max-users 1000
```

## Kiểm chứng train đầy đủ v2

Run `20261117T105742-2e5990ca` ngày 17/11/2026 đã hoàn tất trong khoảng
3 phút 43 giây, gồm khảo sát cấu hình, fit cuối và kiểm tra ổn định.
Cả K-Means và DBSCAN có `fit_users=159356`, `fit_fraction=1.0`,
`train_ratings=17413070`. DBSCAN thực sự xác định core/nối cụm trên toàn bộ
159.356 user, không gán bổ sung từ mô hình chỉ fit 5.000 user.
RAM riêng worker cao nhất 1.088,8 MiB; RAM khả dụng thấp nhất 6.821 MiB.
Bật toàn bộ Train không bảo đảm các cụm tách biệt hơn: DBSCAN lần này vẫn
chỉ có một cụm hợp lệ, noise khoảng 1,01%. Các giới hạn cohort đánh giá vẫn
được công bố riêng và không làm giảm số user fit.

## Lịch sử kiểm chứng v1 (DBSCAN có lấy mẫu fit)

Các số dưới đây thuộc giao thức cũ, không dùng để xác nhận train đầy đủ của v2.

Kiểm tra mẫu ngày 17/11/2026, run `20261117T103253-15f2e07e`: chọn 1.000 user,
còn 974 user/140.898 rating đủ điều kiện; RAM riêng worker cao nhất 493,2 MiB.
K-Means K=2 bao phủ 100%, Silhouette 0,0811, ARI 0,9294: ổn định nhưng cụm chồng
lấn. NDCG@10 Validation 0,07045 so Popularity 0,07121: chưa vượt baseline. Test
0,05430 so 0,05020 chỉ là kết quả mẫu, chưa có khoảng tin cậy và không dùng chọn
lại tham số. DBSCAN gom thành một cụm sau fit cuối, nên không đạt điều kiện dù
bao phủ 99,18%. Không so trực tiếp Silhouette với báo cáo cũ khác đặc trưng/giao thức.

Kiểm chứng toàn bộ dữ liệu, run `20261117T103710-30f5be29`: 24.879.426 rating /
159.356 user đủ lịch sử; worker chạy khoảng 40 giây (phụ thuộc cache/tải máy),
RAM riêng cao nhất 1.161,5 MiB, RAM khả dụng thấp nhất 3.603,9 MiB. DBSCAN fit
5.000 user và gán cho toàn bộ 159.356 hồ sơ; vẫn chỉ một cụm nên không đạt. K-Means
K=2, Silhouette 0,08375, DBI 3,118, bao phủ 100%, ARI ổn định trung bình 0,472.
Tức cụm vẫn chồng lấn và độ ổn định còn hạn chế; không diễn giải là phân cụm tốt.

Ranking dùng 2.000 user seed 42, chấm 1.967 user Validation / 1.953 user Test:

| Tập | Popularity Recall@10 | K-Means Recall@10 | Popularity NDCG@10 | K-Means NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0,05108 | 0,05361 | 0,06578 | 0,06712 |
| Test | 0,04548 | 0,04728 | 0,05945 | 0,06038 |

K-Means cao hơn baseline một ít trên cohort này; chưa kiểm định độ tin cậy.
Không chọn lại tham số theo Test, không so trực tiếp với kết quả khác giao thức.

## Nhật ký thực nghiệm

Khung log dùng chung với hồi quy/phân loại, nằm dưới thanh tiến độ, cao 50% màn
hình và có thanh cuộn. Tự theo dõi dòng mới; cuộn lên sẽ giữ vị trí, nút **Về log
mới nhất** bật theo dõi lại. Có timestamp, màu mức log và nút tải toàn bộ JSONL.

Log gồm nguồn Train/metadata, tạo hồ sơ theo batch, cấu hình và điểm từng ứng
viên, cấu hình chọn, số user fit cuối, bao phủ/noise, ARI từng seed và
Recall@10/NDCG@10 Validation/Test so với Popularity. DBSCAN lớn ghi cả tiến độ
khoảng cách láng giềng, xác định core, nối cụm và gán border. Worker báo còn hoạt
động và RAM mỗi 30 giây. Lỗi/dừng/thành công đều được lưu; mở lại lịch sử đọc
được log. Lượt cũ chưa có `events.jsonl` hiển thị thông báo chưa có nhật ký.

File: `artifacts/experiment-lab/<run_id>/events.jsonl`. Trình duyệt giữ tối đa
5.000 dòng gần nhất; file tải chứa toàn bộ nhật ký. Bật log không đổi cách chia
dữ liệu, số user Train, cấu hình hay công thức chấm điểm.
