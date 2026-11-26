# MovieLens Experiment Studio — pipeline v4

## Kiến trúc và phạm vi thay đổi

Kiểm tra ban đầu: backend FastAPI, frontend React/TypeScript/Vite, dữ liệu chuẩn bị
Parquet từ MovieLens 25M, mô hình NumPy/Numba, scikit-learn, XGBoost và PyTorch.
Lab cũ chạy năm nhóm qua `lab_compact.py`, chưa có 5-Fold, refit 85%, AUC,
DQN hoặc đầy đủ thông tin hiệu năng. Experiment Studio hiện có năm nhóm tác vụ
chính và một reducer dùng chung PCA/TruncatedSVD trong các pipeline. Route PCA
độc lập chỉ còn dành để mở/xuất lịch sử cũ; API vẫn giữ handler tương thích.
Các handler lab cũ vẫn được giữ để tránh ảnh hưởng mã tham chiếu; API chạy mới
dùng schema 4. Lịch sử schema 2/3 vẫn đọc được.

Giữ nguyên tài khoản, hồ sơ demo, danh mục phim, lịch sử đánh giá, gợi ý,
các model production đã lưu. Cấu hình chuẩn bị mới dùng chia thời gian trong từng user; dữ liệu cũ được giữ lại. Lab lưu báo cáo vào `artifacts/experiment-lab/<run_id>/` và cache split riêng,
không ghi đè model phục vụ ứng dụng.

| Thành phần | Trách nhiệm |
|---|---|
| `experiment_config.py` | Kiểm tra cấu hình, thuật toán hợp lệ và giới hạn tài nguyên |
| `experiment_data.py` | Dùng split đầy đủ hoặc lấy mẫu user, cache thống kê/Parquet, fingerprint |
| `experiment_supervised.py` | 5-Fold, chọn cấu hình trên Validation, refit và đánh giá Test |
| `experiment_clustering.py` | Sở thích tương đối, bao phủ cụm, độ ổn định và ranking tương lai |
| `experiment_unsupervised.py` | Apriori toàn cục, Apriori phân nhóm và chẩn đoán PCA/TruncatedSVD |
| `experiment_reduction.py` | Reducer dùng chung, metadata/fingerprint và biến thể raw/reduced |
| `experiment_reinforcement.py` | DQN theo thời gian, checkpoint Validation và hai baseline |
| `experiment_pipeline.py` | Điều phối nhóm, bảng số liệu và biểu đồ |
| `lab.py` / `api/lab.py` | Worker nền, tiến độ, hủy, lịch sử, xuất dữ liệu |
| `Experiments.tsx` và các component Experiment | Sidebar, cấu hình, báo cáo, biểu đồ |

Chi tiết DQN, reducer theo adapter, run IDs và các giới hạn pilot được ghi trong
[báo cáo triển khai RL/giảm chiều](rl-dimensionality-reduction-report.md).

Các giai đoạn triển khai: kiểm tra kiến trúc → phân tách module → xây giao diện
thống nhất → chuẩn hóa chia tập → triển khai năm nhóm chính và reducer dùng chung → lưu báo cáo → tích hợp API
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

1. Chọn một trong năm nhóm chính ở sidebar. URL `group=pca` cũ mở trang lịch sử
   reducer độc lập để xem và xuất run đã lưu; trang này không khởi chạy run mới.
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

CSV gồm metric, Mean/Std CV, cấu hình, luật Apriori toàn cục/theo nhóm nếu có.
Excel gồm Metrics, CV folds, Rules, Grouped rules, Future ranking, Cluster search,
Apriori groups và Protocol theo nhóm phù hợp. JSON giữ toàn bộ số liệu biểu đồ,
các cấu hình đã thử, phân bố nhãn, fingerprint và thông tin môi trường.
Phân cụm bổ sung bản ghi `future_recommendation` trong CSV và các sheet
`Future ranking`, `Cluster search` trong Excel.

Không tự khởi chạy huấn luyện khi chuyển nhóm. Thống kê dataset được cache theo
size/mtime của Parquet; mẫu chia được dùng lại giữa các nhóm có cùng số user.
Lịch sử v2/v3 được giữ và có nhãn giao thức cũ; không so trực tiếp với schema 4.

## Dữ liệu và khả năng tái lập

Đầu vào lấy từ `artifacts/data/{train,validation,test,movies}.parquet`. Hồi quy,
phân loại và phân cụm dùng `per_user_temporal_70_15_15_v2`: mỗi user lấy lịch sử đầu để Train,
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
trong tiến trình riêng, không có giới hạn thời gian hoặc ngưỡng RAM tự dừng.
Worker ghi mức dùng RAM riêng và RAM khả dụng của máy mỗi 30 giây; người dùng có
thể dừng thủ công trên giao diện. Hệ điều hành vẫn có thể kết thúc tiến trình nếu
máy cạn bộ nhớ. File cache có thể làm RSS cao nhưng được hệ điều hành thu hồi.

Đặc trưng float32 tạo theo batch 65.536 hàng. Linear tích lũy thống kê OLS nhỏ;
Logistic/Forest dùng file memmap khi ma trận trên 64 MiB; XGBoost lớn dùng trang
quantile lưu đĩa. Phân loại không fit một hồi quy phụ để lấy đặc trưng. NCF chỉ
chuyển minibatch lên GPU. Cache và tiến trình được giải phóng sau mỗi lần chạy.

Các thống kê gồm ratings, users, phim có rating, catalog phim, genres, rating
min/max/mean, sparsity `1 − ratings/(users × movies)` và số bản ghi mỗi tập.
Cache không tạo ma trận user–item dense. Phân cụm có hai khối sở thích/tần suất
theo genre; giảm chiều vẫn dùng một khối rating trung bình theo genre.
Silhouette lấy mẫu tối đa 2.000 điểm để chặn ma trận khoảng cách bậc hai.

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

Phân cụm dùng chia lịch sử trong từng user, chung cache với hồi quy/phân loại.
Hồ sơ chỉ đọc Train: rating thể loại trừ mức chấm chung của user, làm trơn theo
số lượt đánh giá và ghép tần suất xem thể loại. Thể loại chưa xem có sở thích
tương đối 0. Không dựng lại hồ sơ từ mẩu rating Validation/Test.

Thử K=2/3/4/6/8/10; eps DBSCAN tự lấy theo khoảng cách láng giềng Train hoặc
nhập thủ công. Giữ lại user từ hồ sơ Train để chọn cấu hình: ít nhất hai cụm
hợp lệ và mặc định bao phủ ≥80%, sau đó mới so Silhouette. Kiểm tra lại điều
kiện sau fit cuối; không đạt thì chỉ báo kết quả chẩn đoán. Silhouette/DBI loại
noise nên luôn đọc cùng bao phủ, số user được gán và kích thước cụm. Có kiểm tra
ARI/bao phủ chung khi fit lại với seed 43/44. PCA 2D của biểu diễn raw chỉ minh
họa; variant PCA/TruncatedSVD được fit lại và dùng thật để phân cụm.

K-Means và DBSCAN fit cuối trên toàn bộ hồ sơ Train đủ lịch sử. DBSCAN truy vấn
láng giềng theo batch và duyệt core chưa gán để không giữ đồ thị khoảng cách bậc
hai trong RAM. K-Means lớn dùng partial_fit đi qua từng batch của toàn bộ Train.
Tuning cũng dùng toàn bộ phần Train ngoài user giữ lại, không cắt ở 20.000 user.
Tạo hồ sơ theo batch, không explode toàn bộ bảng rating.

Gợi ý từ lượt thích từng phim trong cụm trên Train, làm trơn về Popularity.
So Recall@10/NDCG@10 trên cùng cohort tối đa 2.000 user có phản hồi tương lai,
toàn bộ catalog Train và bỏ phim đã xem. DBSCAN noise dùng Popularity dự phòng,
đồng thời báo điểm riêng user có cụm. Chọn thuật toán bằng NDCG Validation sau
điều kiện phân cụm/bao phủ; Test chỉ báo cáo cuối. [Giao thức, giới hạn RAM và
kết quả chạy toàn bộ dữ liệu](clustering-experiments.md).

### Association Rules

Một transaction cho mỗi user Train, chỉ chứa phim rating ≥4. Giữ cả transaction
rỗng trong mẫu số support. Catalog giới hạn top phim thích trong Train, có thể
cấu hình `max_items`, `min_support`, `min_confidence`, độ dài itemset tối đa 2–3.
Apriori sinh itemsets và luật thật; báo số luật và sắp xếp Lift/Confidence/Support.
Execution time gồm khai phá và sinh luật, không báo inference giả.
Không tìm thấy luật là kết quả hợp lệ, không tự chèn ví dụ vào báo cáo.

### Giảm chiều hỗ trợ từng thuật toán

Ngay sau thẻ thuật toán, một khối dùng chung hai thẻ PCA và TruncatedSVD mô tả
đầu vào/đường dùng theo nhóm. Trạng thái hoạt động và components phản ánh cấu hình
đang chọn. Chế độ mặc định là tắt; `enabled` chỉ dùng representation được chọn,
`compare` thêm raw control. Cấu hình chi tiết nằm trong phần Chạy thực nghiệm.

`reduction_mode=off` giữ đường chạy gốc. `enabled` chạy các representation đã
chọn; `compare` thêm raw control trên cùng split, seed và budget. Variant IDs
ổn định xuất hiện ở bảng, cấu hình, CV, biểu đồ và export. Chọn reducer/chiều
bằng metric tác vụ trên Validation; EVR và reconstruction chỉ là chẩn đoán.

- Linear/XGBoost/Logistic/Forest chỉ nén block tương tác preference-user với
  genre phim, giữ bias/thống kê nền; scaler và reducer fit trong Train/fold.
- Funk SVD giữ latent factors và thêm correction Ridge từ profile nén. NCF giữ
  embeddings, thêm nhánh MLP từ profile nén và dùng Validation cho early stopping.
- K-Means/DBSCAN scale/reduce hồ sơ preference+tần suất; DBSCAN tìm lại eps.
  Profile cụm vẫn mô tả bằng genre gốc, noise dùng Popularity dự phòng.
- DQN chỉ nén profile support; action mask, fullness, bước còn lại, observed flag
  và tín hiệu tương tác vẫn giữ nguyên. Random/Popularity không bị chạy lại giả.
- Apriori giữ transaction boolean. Biến thể giảm chiều phân nhóm profile Train,
  khai phá luật riêng; support trong nhóm và support toàn cục dùng mẫu số riêng.

### Chẩn đoán Dimensionality Reduction

PCA và TruncatedSVD dùng feature family user/profile tương ứng; thử danh sách
components. PCA center đặc trưng đã scale; TruncatedSVD chiếu không center. Với
TruncatedSVD, explained variance là phương sai thực tế của thành phần chiếu
`Var(XV)` chia tổng phương sai của đầu vào đã scale, không phải năng lượng chiếu
`E[(XV)^2]`. Đây là SVD giảm chiều, khác Funk SVD học rating ở Regression.

Explained variance học trên Train; reconstruction MSE đo trên Validation/Test.
Vẽ variance theo số chiều, biểu diễn trước/sau và so sánh lỗi tái tạo.
Chẩn đoán độc lập báo EVR và reconstruction; không xem đó là độ chính xác rating.
Với adapter tác vụ, chọn bằng RMSE/F1/NDCG/reward Validation thay vì ngưỡng EVR.
Reducer chỉ fit trên phần Train/fold và transform phần held-out; rank thấp hoặc
không có phương sai được báo rõ.

### Reinforcement Learning

Môi trường dựa trên rating thực, không sinh rating hay nhiễu nhân tạo.

- DQN là Q-Learning xấp xỉ bằng MLP nhỏ; không đưa userId vào mạng. DQN có target
  network, replay hữu hạn, Huber loss, gradient clipping và epsilon schedule.
- Train episode chia support/query theo timestamp; profile ban đầu chỉ từ support,
  reward chỉ từ query. Profile normalization cũng fit trên support.
- Observation gồm preference/frequency, fullness, mask, bước còn lại, phản hồi
  trước và observed flag.
- Action = tối đa 30 phim phổ biến Train chưa được chọn trong episode.
- Reward quan sát = `(rating − 2,75)/2,25`, nằm trong [-1,1].
- Action thiếu phản hồi lịch sử nhận 0 theo quy ước; không được coi là user ghét phim.
- Epsilon schedule áp dụng khi học; Validation chọn và phục hồi snapshot model/reducer;
  Test dùng checkpoint đóng băng và không cập nhật trọng số.

Các policy dùng cùng catalog, cohort, episode IDs và mask. Chọn cohort có seed;
log số user/step thực học và actual-action denominator. Cumulative reward là phụ,
đọc cùng average reward/action, coverage, ranking và bootstrap ghép cặp theo user.
Q-table `q_learning` trong báo cáo lịch sử không được đổi tên; kết quả mới dùng id
`dqn`. Không có propensity nên reward thiếu=0 không phải đánh giá counterfactual
không thiên lệch hoặc hiệu quả online.

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

## Preset và giao diện Experiment Studio

Giao diện gửi mã preset phiên bản, nhóm, danh sách thuật toán và `reduction_choice`
(`pca`, `truncated_svd` hoặc `off`). Backend xác thực lựa chọn và tự nạp các tham số chuẩn.
Người dùng chọn một trong ba thẻ ở mục Giảm chiều dữ liệu; các tham số huấn luyện khác chỉ đọc.
Lựa chọn được giữ riêng cho từng nhóm trong phiên mở trang và được lưu trong request của run.

| Nhóm | Giảm chiều trong preset chuẩn |
|---|---|
| Hồi quy | Linear: PCA 8 chiều; XGBoost, Funk SVD và NCF: biểu diễn gốc |
| Phân loại | Logistic: PCA 8 chiều; Random Forest: biểu diễn gốc |
| Phân cụm | K-Means và DBSCAN: PCA 8 chiều trên hồ sơ Train |
| Luật kết hợp | PCA 8 chiều để phân nhóm user; Apriori toàn cục và transaction boolean được giữ |
| Học tăng cường | DQN dùng hồ sơ gốc; Random/Popularity không giảm chiều |

Preset chuẩn tắt tuning. Bảng trên mô tả mặc định; có thể thay PCA bằng TruncatedSVD
8 chiều hoặc chọn Không giảm chiều. DQN cũng cho phép chọn PCA/TruncatedSVD 8 chiều;
Random/Popularity không dùng reducer. Các nhánh XGBoost, Funk SVD, NCF và Random Forest
vẫn dùng biểu diễn gốc theo preset. Mỗi lần chạy chỉ dùng biểu diễn đã chọn.
PCA/TruncatedSVD fit bằng dữ liệu Train; mỗi fold CV fit reducer riêng.
Trang “Dimensionality Reduction” cũ chỉ còn để đọc lịch sử đã lưu.

Kết quả hiện theo thứ tự: tóm tắt giảm chiều, bảng theo nhóm, biểu đồ so sánh và 5-fold theo thời
gian trên Train (chỉ hồi quy/phân loại). Metadata và biểu đồ chẩn đoán nằm trong phần chi tiết.
Thiếu trường của báo cáo cũ hiển thị “Không có dữ liệu”; không suy ra kết luận tăng tốc khi thiếu
đối chứng.

## Kiểm chứng

```powershell
cd project
.\.venv\Scripts\python.exe -m pytest tests -q
cd frontend
npm run build
npm run test:experiments
```

Playwright hiện kiểm tra catalog preset thật của backend, payload nhóm/thuật toán/preset gửi từ UI,
trạng thái hai thẻ reducer, cấu hình chỉ đọc, thứ tự kết quả, báo cáo lịch sử thiếu trường và bố cục
390px. Request tạo job được chặn trong bài kiểm tra trình duyệt để không khởi chạy một loạt thí
nghiệm MovieLens kéo dài; backend có test riêng cho việc xác thực preset và dispatch reducer.
PCA cũ được kiểm tra ở trang history-only.
Các phép kiểm tra chống rò rỉ xác nhận đặc trưng fold không nhận dòng giữ lại,
thay nhãn Test không thay cấu hình được chọn, và refit dùng đúng Train + Validation.
Fixtures sinh trong tests chỉ dùng kiểm tra công thức, không đưa vào báo cáo UI.

Xem [báo cáo chạy xác nhận lịch sử](experiment-studio-results.md) để lấy run ID,
metric và cấu hình trước schema 4; [báo cáo RL/giảm chiều](rl-dimensionality-reduction-report.md)
ghi các run ID và kiểm chứng mới. Artifact runtime và ảnh trình duyệt không được commit.

Tham khảo phương pháp: [scikit-learn — data leakage](https://scikit-learn.org/stable/common_pitfalls.html),
[cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html),
[DBSCAN core samples](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).

## Dừng và tiếp tục hồi quy/phân loại

Các run mới lưu kết quả riêng cho từng biến thể giảm chiều, thuật toán, cấu hình Validation,
fold CV và refit/Test trong `progress.json`. Chọn run đã dừng trong lịch sử rồi nhấn **Tiếp tục
từ checkpoint**. Kết quả được lưu giữ theo run ID; `result.json`, CSV/Excel và báo cáo chỉ có
sau khi mọi đơn vị công việc hoàn tất.

NCF lưu model, optimizer, epoch kế tiếp, trạng thái early stopping, trọng số tốt nhất, đường học
và RNG sau mỗi epoch. NCF hybrid lưu riêng checkpoint của embedding/MLP nền và correction branch.
Các fit không có checkpoint epoch, như XGBoost hoặc SVD, chạy lại riêng fit đang dở. Ứng dụng
kiểm tra hash dữ liệu/split, request (gồm seed và tham số), code và phiên bản thư viện; nếu không
khớp, không dùng lại tiến độ đó. Lần chạy cũ không có progress/checkpoint không thể khôi phục.
Nút tiếp tục hiện được cung cấp cho hồi quy và phân loại; các nhóm phân cụm, association, PCA
lịch sử và RL chưa có cơ chế tiếp tục theo từng đơn vị công việc.

Với XGBoost, cấu hình CPU threads nhận 2, 4 hoặc 8; mặc định là 4 theo benchmark cohort
4.904 user trên máy phát triển. `n_jobs` của estimator và native threadpool limit dùng cùng giá
trị. Bộ giới hạn sửa đường dẫn DLL Unicode trên Windows và nạp XGBoost trước khi dò OpenMP.
Benchmark đo feature/cache, fit, tổng thời gian và RSS trên cùng cohort user, split, seed và tham
số. Bảng, số đo lặp lại và giới hạn mẫu được ghi trong `docs/xgboost-thread-benchmark.md`;
benchmark không fit toàn bộ MovieLens 25M.
