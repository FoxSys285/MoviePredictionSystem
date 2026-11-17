# MovieLens Experiment Studio — pipeline v3

## Kiến trúc và phạm vi thay đổi

Kiểm tra ban đầu: backend FastAPI, frontend React/TypeScript/Vite, dữ liệu chuẩn bị
Parquet từ MovieLens 25M, mô hình NumPy/Numba, scikit-learn, XGBoost và PyTorch.
Lab cũ chạy năm nhóm qua `lab_compact.py`, chưa có 3-Fold, refit 85%, AUC,
Q-Learning hoặc đầy đủ thông tin hiệu năng. Các handler cũ vẫn được giữ để tránh
ảnh hưởng mã tham chiếu, nhưng API chạy mới chuyển sang pipeline v3.

Giữ nguyên tài khoản, hồ sơ demo, danh mục phim, lịch sử đánh giá, gợi ý,
các model production đã lưu. Cấu hình chuẩn bị mới dùng chia thời gian trong từng user; dữ liệu cũ được giữ lại. Lab lưu báo cáo vào `artifacts/experiment-lab/<run_id>/` và cache split riêng,
không ghi đè model phục vụ ứng dụng.

| Thành phần | Trách nhiệm |
|---|---|
| `experiment_config.py` | Kiểm tra cấu hình, thuật toán hợp lệ và giới hạn tài nguyên |
| `experiment_data.py` | Dùng split đầy đủ hoặc lấy mẫu user, cache thống kê/Parquet, fingerprint |
| `experiment_supervised.py` | 3-Fold, chọn cấu hình trên Validation, refit và đánh giá Test |
| `experiment_unsupervised.py` | Vector user–genre, clustering, Apriori, PCA và TruncatedSVD |
| `experiment_reinforcement.py` | Q-Learning và hai baseline trên môi trường phản hồi lịch sử |
| `experiment_pipeline.py` | Điều phối nhóm, bảng số liệu và biểu đồ |
| `lab.py` / `api/lab.py` | Worker nền, tiến độ, hủy, lịch sử, xuất dữ liệu |
| `Experiments.tsx` và các component Experiment | Sidebar, cấu hình, báo cáo, biểu đồ |

Các giai đoạn triển khai: kiểm tra kiến trúc → phân tách module → xây giao diện
thống nhất → chuẩn hóa chia tập → triển khai sáu nhóm → lưu báo cáo → tích hợp API
→ kiểm thử đơn vị/API/trình duyệt và chạy MovieLens thật → tài liệu này.

## Khởi chạy và sử dụng

Nếu máy đã có môi trường và frontend build:

```powershell
cd project
.\.venv\Scripts\python.exe -X utf8 -m movie_rating.cli serve
```

Mở <http://127.0.0.1:8000/experiments>. Nếu cài mới, chạy `scripts/setup.ps1`,
`npm ci` và `npm run build` trong `frontend` theo README dự án. Node.js/npm cần
được cài trên máy để dùng `run_project.bat`. Launcher tự tìm Node.js/npm trên
PATH, thư mục cài thông thường, NVM và Laragon (kể cả khi IDE chưa nhận PATH).
Nếu dùng vị trí khác, đặt biến `MOVIE_NODE_DIR` tới thư mục chứa `node.exe`
và `node_modules/npm`. Python yêu cầu 3.12 hoặc 3.13.

1. Chọn một trong sáu nhóm ở sidebar.
2. Xem mục tiêu, dataset đã chuẩn bị và định nghĩa các tiêu chí.
3. Chọn một thuật toán để chạy riêng hoặc nhiều thuật toán để so sánh.
4. Mặc định dùng toàn bộ ratings đã chuẩn bị; có thể bỏ chọn để giới hạn user.
   Seed cố định 42.
5. Chạy, theo dõi trạng thái/tiến độ; có thể dừng sau bước tính toán hiện tại.
6. Xem bảng đánh giá, nhận xét, biểu đồ và cấu hình được chọn.
7. Dùng lịch sử để mở lại báo cáo; tải CSV, Excel hoặc JSON.

Phần cấu hình thông số có thể mở rộng/thu gọn, giữ nguyên giá trị đã chỉnh.
Mỗi tiêu chí có mốc mặc định có thể sửa, giải thích cách đọc và nhãn đánh giá
ngay cạnh kết quả. Xem [phân tích từng mốc](metric-reference-guide.md); các mốc
chỉ diễn giải chỉ số, không thay đổi training hoặc lựa chọn model.

CSV gồm metric, Mean/Std CV, cấu hình và các luật Apriori nếu có. Excel gồm
Metrics, CV folds, Rules (nếu có) và Protocol. JSON giữ toàn bộ số liệu biểu đồ,
các cấu hình đã thử, phân bố nhãn, fingerprint và thông tin môi trường.

Không tự khởi chạy huấn luyện khi chuyển nhóm. Thống kê dataset được cache theo
size/mtime của Parquet; mẫu chia được dùng lại giữa các nhóm có cùng số user.
Lịch sử v2 được giữ và có nhãn giao thức cũ; không so trực tiếp với kết quả v3.

## Dữ liệu và khả năng tái lập

Đầu vào lấy từ `artifacts/data/{train,validation,test,movies}.parquet`. Hồi quy và
phân loại dùng `per_user_temporal_70_15_15_v2`: mỗi user lấy lịch sử đầu để Train,
đoạn tiếp theo Validation, đoạn cuối Test, khoảng 70/15/15. Rating cùng timestamp
không tách qua hai tập. User dưới 10 rating hoặc không có ba khoảng thời gian
hợp lệ được loại và ghi số lượng trong `provenance.exclusions`.

Toàn bộ user đủ điều kiện được dùng khi bật toàn bộ dữ liệu. Nếu giới hạn user,
chọn user bằng seed 42 và áp dụng cùng giao thức thời gian, không chia IID lại.
Timeline giữa các user có thể chồng lấn; đây là đánh giá cá nhân hóa cho user có
lịch sử, không mô phỏng tương lai toàn cục hoặc user mới hoàn toàn.

Dữ liệu cũ chia toàn cục được chuyển một lần sang cache riêng
`artifacts/data-personalized-v2-<signature>/`, không ghi đè dữ liệu hoặc kết quả
lịch sử. Chạy mới tự dùng cache này; không cần xóa thư mục data cũ. `prepare`
cho thư mục artifacts mới dùng cấu hình `split_strategy="per_user_temporal"`;
`global_temporal` vẫn có thể chọn cho nghiên cứu tương lai toàn cục.

[Chi tiết vận hành và RAM](personalized-experiments.md). Thực nghiệm lớn chạy
trong tiến trình riêng. Ngân sách mặc định 4096 MiB cho bộ nhớ riêng của worker,
đồng thời hạ theo RAM khả dụng; để lại ít nhất 12% RAM máy (tối thiểu 1 GiB).
Khi worker vượt ngân sách hoặc RAM khả dụng dưới mức dự phòng, dừng tiến trình,
không công bố kết quả dở. Đây là giám sát mỗi 0,25 giây, không phải hard limit
của hệ điều hành. File cache có thể làm RSS cao nhưng được hệ điều hành thu hồi.

Đặc trưng float32 tạo theo batch 65.536 hàng. Linear tích lũy thống kê OLS nhỏ;
Logistic/Forest dùng file memmap khi ma trận trên 64 MiB; XGBoost lớn dùng trang
quantile lưu đĩa. Phân loại không fit một hồi quy phụ để lấy đặc trưng. NCF chỉ
chuyển minibatch lên GPU. Cache và tiến trình được giải phóng sau mỗi lần chạy.

Các thống kê gồm ratings, users, phim có rating, catalog phim, genres, rating
min/max/mean, sparsity `1 − ratings/(users × movies)` và số bản ghi mỗi tập.
Cache không tạo ma trận user–item dense. Vector user–genre chỉ có số cột bằng số
genres; silhouette lấy mẫu tối đa 2.000 điểm để chặn ma trận khoảng cách bậc hai.

## Giao thức từng nhóm

### Regression

Linear Regression, XGBoost, Funk SVD có bias và NCF dự đoán rating trong [0,5;5].
Linear/XGBoost dùng thống kê user/phim và metadata thể loại; SVD/NCF học từ rating.
Mặc định: XGBoost 150 cây, depth 6, learning rate 0,05; SVD/NCF tối đa 15 epochs,
32 factors, SVD learning rate 0,005, NCF 0,003. NCF dùng patience 3.
Đặc trưng gồm thống kê user/phim, tỷ lệ thích, độ phân tán rating, thể loại và
sở thích từ 20 rating gần nhất trong phần fit. Khi tạo hàng huấn luyện, loại
đóng góp của nhãn chính hàng đó khỏi thống kê và hồ sơ gần đây.

Tuning tắt mặc định để giới hạn thời gian. Khi bật, thử cấu hình nhẹ, mặc định
và mạnh hơn. Chọn RMSE Validation thấp nhất rồi chỉ chạy ba lượt CV cho cấu hình
được chọn. CV mở rộng lịch sử theo thời gian trong từng user trên Train; mỗi
lượt fit lại đặc trưng/mô hình, học quá khứ và chấm đoạn kế tiếp. Báo cáo Mean
và sample Std (ddof=1); đây là CV chẩn đoán sau lựa chọn, không phải nested CV.

NCF dùng Validation cho early stopping; train-rating MSE đo trên tối đa 50.000
hàng cố định để tránh suy luận toàn bộ Train sau mỗi epoch. Validation RMSE vẫn
đo trên toàn bộ Validation theo batch. Trong CV dùng epoch cố định.

Sau lựa chọn, mỗi thuật toán fit mới trên Train + Validation 85%. NCF dùng số
epoch tốt nhất đã khóa, không dùng Test cho Early Stopping. Báo cáo Test cuối
cùng cùng RMSE, MAE, training/inference time và kích thước model. Nhận xét chọn
bằng Validation, đối chiếu CV ± std và chi phí thực đo, không mặc định NCF tốt nhất.

### Classification

Không thích = rating <3,5; Thích = rating ≥3,5. Logistic và Forest dùng cùng
ba lượt CV thời gian như hồi quy, không dùng StratifiedKFold ngẫu nhiên.
Logistic mặc định C=1, max_iter=600, solver SAGA với dữ liệu float32 đã scale.
Forest mặc định 100 cây, depth 8, min_samples_leaf=5, fit từng cây lần lượt.
Tuning thử C nhỏ hơn/lớn hơn hoặc cây nông hơn/sâu hơn. Validation chọn F1 lớp
Thích và ngưỡng xác suất (0,1–0,9), khóa cấu hình/ngưỡng trước refit và Test.

F1, Precision và Recall trong v3 là của **lớp Thích**, khác macro của lab v2.
AUC-ROC dùng xác suất dự đoán, không dùng nhãn cứng. AUC không xác định khi tập
đánh giá chỉ có một lớp. Báo cáo Accuracy, F1, AUC, Precision, Recall, ma trận
[[TN,FP],[FN,TP]], ROC và tỷ lệ nhãn của từng tập. CV dùng ngưỡng 0,5 thống nhất.

### Clustering

Đặc trưng = rating trung bình của từng user theo từng genre. Train loại user
dưới số ratings tối thiểu đã chọn. Genre thiếu được điền bằng rating trung bình
của user; scaler chỉ fit Train. Validation/Test dựng profile riêng từ ratings
giữ lại, không tuyên bố là đánh giá user mới hoàn toàn.

Thử danh sách K và tích eps × min_samples. Ưu tiên silhouette Validation;
nếu Validation không hợp lệ, cấu hình trong mỗi thuật toán dùng Train làm
fallback và báo cáo rõ không đủ cơ sở chọn thuật toán triển khai.
Silhouette/DBI loại noise; số cụm và tỷ lệ noise luôn được báo cáo riêng.
PCA 2D và bảng genre trung bình giúp diễn giải cụm Train.

DBSCAN không có predict chuẩn. Lab sử dụng phép mở rộng: core Train gần nhất
trong eps quyết định nhãn; ngoài eps là noise. Index nearest-neighbor được tính
vào training time, truy vấn gán nhãn mới tính vào inference time.

### Association Rules

Một transaction cho mỗi user Train, chỉ chứa phim rating ≥4. Giữ cả transaction
rỗng trong mẫu số support. Catalog giới hạn top phim thích trong Train, có thể
cấu hình `max_items`, `min_support`, `min_confidence`, độ dài itemset tối đa 2–3.
Apriori sinh itemsets và luật thật; báo số luật và sắp xếp Lift/Confidence/Support.
Execution time gồm khai phá và sinh luật, không báo inference giả.
Không tìm thấy luật là kết quả hợp lệ, không tự chèn ví dụ vào báo cáo.

### Dimensionality Reduction

PCA và TruncatedSVD dùng cùng vector user–genre chuẩn hóa; thử danh sách
components. PCA center nội bộ; TruncatedSVD không center nội bộ nhưng đầu vào
chung đã chuẩn hóa nên kết quả có thể rất gần nhau. Đây là SVD giảm chiều,
khác Funk SVD học rating ở Regression.

Explained variance học trên Train; reconstruction MSE đo trên Validation/Test.
Vẽ variance theo số chiều, biểu diễn trước/sau và so sánh lỗi tái tạo.
Chọn cấu hình ít chiều nhất giữ ≥90% variance, sau đó dùng reconstruction error
Validation; nếu không đạt, chọn lỗi Validation thấp nhất. Dữ liệu không có
phương sai được báo lỗi thay vì công bố variance giả.

### Reinforcement Learning

Môi trường dựa trên rating thực, không sinh rating hay nhiễu nhân tạo.

- State = (userId, bitmask phim đã chọn, genre phim trước, phản hồi thích/không thích).
- Action = tối đa 30 phim phổ biến Train chưa được chọn trong episode.
- Reward quan sát = `(rating − 2,75)/2,25`, nằm trong [-1,1].
- Action thiếu phản hồi lịch sử nhận 0 theo quy ước; không được coi là user ghét phim.
- Policy epsilon-greedy khi học, greedy khi đánh giá. Cấu hình alpha/gamma/epsilon.

Q-table chỉ lưu state đã gặp. Q-Learning, Random và Popularity được đánh giá trên
cùng 100 user-episode × 5 bước cho mỗi split; seed đánh giá tách seed huấn luyện.
Huấn luyện chỉ dùng Train, checkpoint dùng Validation, Test giữ riêng.
Hiển thị reward theo episode, cumulative/average reward và observed-feedback
coverage. Không có propensity nên không thể suy ra giá trị counterfactual hay
hiệu quả online từ kết quả môi trường offline này.

## Đo hiệu năng và vận hành

Training/inference dùng `time.perf_counter`, đơn vị giây, cùng máy CPU. Số mẫu
batch được ghi trong từng dòng; Regression còn báo µs/rating. Fit cuối của mô
hình supervised gồm bước xây đặc trưng. Inference Classification chỉ đo
predict_proba, loại thời gian xây đặc trưng và được ghi rõ trong notes.
Chi phí tuning/CV có ở `tuning`/folds, không gộp vào final training time.

Model size là số byte pickle trong RAM, không phải GPU RAM. RSS là toàn tiến
trình được psutil lấy mẫu, không phải RAM riêng của model. Python, thư viện,
CPU, số core, RAM, hệ điều hành và SHA256 source/data/config được lưu để đối chiếu.
Không hứa thời gian bit-for-bit giống nhau giữa các lần chạy.

Một worker nền cho mỗi process, một job đang hoạt động. Chạy API **một process**
để tránh nhiều worker cùng huấn luyện. Huấn luyện không nằm trong request UI.
Cancel có hiệu lực tại checkpoint giữa bước fit, không ngắt thô một phép fit.
Job đang chạy khi server khởi động lại được hiển thị interrupted. Lỗi có thông
báo riêng và không xuất bản result dở; JSON được ghi tạm rồi thay thế nguyên tử.

## Kiểm chứng

```powershell
cd project
.\.venv\Scripts\python.exe -m pytest tests -q
cd frontend
npm run build
npm run test:experiments
```

Đã xác nhận bộ tests toàn dự án, TypeScript/Vite build và Playwright với sáu nhóm
chạy MovieLens thật, chọn riêng/nhiều thuật toán, lưu và mở lại lịch sử, xuất
CSV/XLSX/JSON, bố cục 390px ở dark/light, cùng khôi phục sau lỗi API.
Các phép kiểm tra chống rò rỉ xác nhận đặc trưng fold không nhận dòng giữ lại,
thay nhãn Test không thay cấu hình được chọn, và refit dùng đúng Train + Validation.
Fixtures sinh trong tests chỉ dùng kiểm tra công thức, không đưa vào báo cáo UI.

Xem [báo cáo chạy xác nhận](experiment-studio-results.md) để lấy run ID, metric
và cấu hình thực tế. Artifact runtime và ảnh trình duyệt không được commit.

Tham khảo phương pháp: [scikit-learn — data leakage](https://scikit-learn.org/stable/common_pitfalls.html),
[cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html),
[DBSCAN core samples](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).
