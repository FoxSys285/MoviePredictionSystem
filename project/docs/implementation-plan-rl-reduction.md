# Kế hoạch triển khai Học tăng cường và giảm chiều trong Experiment Studio

Ngày lập: 19/11/2026. Trạng thái cập nhật 20/11/2026: **đã triển khai và kiểm
chứng theo thứ tự A rồi B**. Xem [báo cáo triển khai](rl-dimensionality-reduction-report.md)
và [bàn giao/nhật ký kiểm chứng](session-handoff-rl-reduction.md).

Kế hoạch ban đầu được lập trong session trước và thực hiện trong session hiện
tại theo yêu cầu của người dùng. Experiment Studio đã nhận DQN, reducer và
adapter cho từng nhóm; kết quả giữ rõ giới hạn pilot, coverage, các lần kiểm
chứng và các mục lint repository-wide còn tồn tại. Không thay mô hình gợi ý
production hay dữ liệu đã chuẩn bị.

## 1. Kết quả cần giao

1. Hoàn thiện nhóm Học tăng cường trước: dữ liệu, biểu diễn trạng thái, học policy, chọn checkpoint, đánh giá, log trực tiếp, hủy, lịch sử, biểu đồ và xuất báo cáo.
2. Tích hợp PCA/TruncatedSVD vào thực nghiệm của từng nhóm bằng thiết kế phù hợp với từng thuật toán. Có đối chứng không giảm chiều và mô tả vai trò hỗ trợ ngay trên giao diện.
3. Chọn cấu hình bằng Validation, giữ Test cho đánh giá cuối. Không mặc định giảm chiều hoặc RL sẽ tốt hơn; kết quả xấu/không đủ bằng chứng vẫn phải hiển thị đúng.
4. Giữ khả năng đọc lịch sử v2/v3 và tách rõ giao thức mới. Phần tích hợp thuộc Experiment Studio; không tự thay mô hình đang phục vụ tài khoản/gợi ý production.

## 2. Snapshot audit trước khi triển khai

| Thành phần | Thực trạng và vị trí |
|---|---|
| Điều phối đang dùng | `src/movie_rating/experiment_pipeline.py` gọi các handler `experiment_*`; không sửa nhầm handler `lab_*` cũ |
| Dữ liệu hai nhóm | `experiment_data.py:ExperimentData.load()` chỉ xếp rating/classification/clustering vào `historical`; RL/PCA không chuyển sang cache thời gian từng user |
| Dữ liệu đầy đủ hiện có | `artifacts/data/manifest.json` là chia thời gian toàn cục cũ: 17.500.066 / 3.750.014 / 3.750.015 ratings |
| Cold user của dữ liệu cũ | 84,80% user Validation và 88,63% user Test chưa có trong Train |
| Chế độ lấy mẫu RL/PCA | Gom lịch sử rồi `split_ratings()` ngẫu nhiên từng rating. Mẫu 40 user đã kiểm tra đều có Train xảy ra sau đầu Validation/Test |
| Q-Learning hiện tại | State chứa `userId` và bitmask; không chia sẻ giá trị sang user mới. Mặc định 500 episode lấy user ngẫu nhiên |
| Reward hiện tại | Rating quan sát chuẩn hóa về [-1, 1], thiếu phản hồi = 0; đây là quy ước mô phỏng, không phải đánh giá policy không thiên lệch |
| Checkpoint RL | Có vẽ Validation nhưng không phục hồi Q-table tốt nhất |
| Log backend | `experiment_logs.py:LOGGED_GROUPS` chưa chứa reinforcement/pca; `lab.py` và `experiment_worker.py` dựa vào danh sách này |
| Log frontend | `pages/Experiments.tsx` chỉ gắn `ExperimentLog` cho rating/classification/association/clustering |
| Giảm chiều hiện tại | `experiment_unsupervised.py:reduction()` nén hồ sơ user–genre, chưa được dùng làm đầu vào trong các handler nhóm khác |
| Đặc trưng sẵn có | `models/tabular.py:RatingFeatures` có đặc trưng theo batch và cơ chế loại đóng góp nhãn hàng Train; `experiment_clustering.py:TasteProfiles` có sở thích tương đối + tần suất |
| Lịch sử/schema | `LabService.list_runs()` và UI đang có kiểm tra cứng schema 2/3; thay schema phải sửa cả tạo job, liệt kê và render |

Kiểm chứng đã làm ở session lập kế hoạch: 7 tests liên quan qua; tái chạy RL trên mẫu MovieLens 40 user với 50 và 500 episode. Với 500 episode, 408/500 quyết định Test không có Q khác 0 để phân biệt hành động, chỉ 14/500 có phản hồi quan sát. Đây là bằng chứng cho việc cần sửa thiết kế, không phải kết quả giao thức mới.

Working tree trước khi lập kế hoạch đã có sửa đổi của người dùng ở `../run_project.bat` và `scripts/start.ps1`. Không hoàn tác hoặc ghi đè các sửa đổi này.

## 3. Các quyết định thiết kế

### 3.1. Dữ liệu và chống rò rỉ

- RL và thực nghiệm giảm chiều có tác vụ phía sau dùng `per_user_temporal_70_15_15_v2` ở cả chế độ đầy đủ và lấy mẫu. Tái sử dụng `personalized_directory()` và `user_temporal_split()`, giữ các timestamp bằng nhau cùng một tập.
- Apriori toàn cục hiện có giữ phạm vi toàn bộ prepared Train như giao thức riêng. Biến thể Apriori theo nhóm chỉ fit từ cùng Train đó và công bố riêng; không âm thầm đổi phạm vi dữ liệu của đối chứng.
- Không ghi đè Parquet, manifest, model hay result lịch sử. Đưa protocol, nguồn split, exclusion và fingerprint thực tế vào run mới.
- Fit bộ tạo đặc trưng, imputer/scaler, reducer và estimator trong phần fit của từng fold. Không dùng hồ sơ được tổng hợp từ cả Train+Validation+Test.
- Hàng Train dùng đặc trưng đã loại nhãn của chính hàng đó; đặc trưng lịch sử được mô tả là quá khứ phải dùng prefix trước timestamp và không dùng rating cùng timestamp. Không chỉ thêm reducer sau một ma trận đặc trưng đang chứa nhãn.
- Validation/Test chỉ cung cấp nhãn cho chấm điểm, trừ phản hồi đã được môi trường RL trả về ở bước trước của cùng episode. Phản hồi này không được cập nhật trọng số policy khi đánh giá hoặc rò sang episode khác.
- Supervised giữ giao thức hiện tại: chọn cấu hình trên Validation, CV chẩn đoán trên Train, refit Train+Validation với cấu hình/ngưỡng/epoch đã khóa rồi Test.
- RL giai đoạn A dùng model và hồ sơ nền học từ Train cho cả Validation và Test; không refit 85% trong phiên bản đầu. Ghi rõ giao thức này thay vì gắn mô tả refit supervised cho RL.

### 3.2. Hướng sửa RL được chọn

Đổi policy học chính thành **DQN, tức Q-Learning xấp xỉ bằng mạng nhỏ**, để chia sẻ kiến thức qua đặc trưng. Định danh mới `dqn`; không đổi tên Q-table cũ thành DQN. Ba policy mặc định cho run mới: `dqn`, `random`, `popular`. Kết quả `q_learning` cũ vẫn đọc được và có nhãn giao thức cũ.

- Observation gồm hồ sơ sở thích/tần suất từ lịch sử được phép biết, độ đầy đủ lịch sử, mask phim đã chọn/không hợp lệ, số bước còn lại, thể loại và phản hồi trước (có cờ observed). Không đưa `userId` số vào mạng như biến liên tục.
- Mạng nhỏ, gợi ý hai lớp 64 units, đầu ra Q cho catalog action cố định; dùng target network, replay buffer có giới hạn, minibatch, Huber loss, gradient clipping và epsilon schedule có cấu hình. Dùng PyTorch đã có, không cần thêm framework RL.
- Mask action áp dụng cả khi exploration, chọn greedy và tính max-Q mục tiêu. Transition terminal hoặc hết action hợp lệ có bootstrap bằng 0.
- Snapshot checkpoint phải sao chép độc lập trọng số, encoder/reducer và cấu hình. Chọn bằng Validation, phục hồi checkpoint tốt nhất trước Test; Test không tác động lựa chọn.
- Mặc định CPU và mạng nhỏ để so chi phí rõ ràng. Nếu hỗ trợ GPU, ghi thiết bị và không trộn số đo CPU/GPU trong cùng kết luận hiệu năng.
- Giao diện phải giải thích vì sao policy mới dùng đặc trưng và khác Q-table cũ. Không hứa hội tụ hoặc vượt baseline từ việc đổi sang DQN.

### 3.3. Phạm vi mô phỏng RL

- Giữ reward quan sát `(rating - 2.75) / 2.25`; không sinh rating, không thêm nhiễu để tạo kết quả tốt.
- Thiếu phản hồi vẫn là 0 theo quy ước mô phỏng, nhưng phải có `observed_feedback_fraction`, số hành động quan sát được và cảnh báo theo dữ liệu. Không gọi đây là đánh giá counterfactual/không thiên lệch, không tự tạo propensity hoặc IPS.
- Catalog action lấy từ Train, mặc định tối đa 30 phim; cấu hình phải hiển thị giới hạn thực tế. Định nghĩa popularity từ Train giống nhau cho mọi policy.
- Khi đánh giá, loại phim đã xem trong lịch sử nền Train và phim đã chọn trong episode. Không lọc candidate theo việc Validation/Test có rating, vì như vậy policy sẽ biết trước phim có phản hồi.
- Nếu user còn ít hơn horizon action hợp lệ, episode ngắn lại; nếu không còn action, loại khỏi cohort với lý do. Không chia reward cho hằng số 500 khi số bước thực tế khác 500.
- Chọn cohort và danh sách episode bằng seed độc lập policy; các policy dùng đúng cùng episode IDs, user và catalog/mask ban đầu. Báo tổng user, đủ điều kiện, bị loại, được lấy mẫu và tỷ lệ bao phủ.
- Episode Train có support/query tách theo thời gian bên trong Train. Hồ sơ ban đầu chỉ dùng support; reward lấy từ query; phim trong support bị mask. Không dựng state từ chính query ratings rồi cho agent đoán lại.
- Duyệt user đủ điều kiện theo lượt xáo trộn có seed, thay việc bốc lại ngẫu nhiên 500 lần. Mặc định một sweep; nếu giới hạn bước/user, hiển thị số thực sự đã học và tỷ lệ bao phủ, không mô tả là học hết user.
- Validation dùng hồ sơ nền Train và reward Validation; Test dùng cùng hồ sơ nền Train và reward Test. Trọng số đóng băng trong cả hai. Phản hồi tuần tự chỉ có hiệu lực trong episode đó.
- Môi trường có reward tĩnh theo rating và không mô hình hóa thay đổi sở thích/giữ chân người dùng; phải mô tả giới hạn này ngay ở kết luận RL.

### 3.4. Giảm chiều là thành phần dùng chung, có adapter cho từng thuật toán

Tạo module mới, tên đề xuất `experiment_reduction.py`, chứa specification, fit/transform, thống kê, serialization và metadata; không nhúng logic lựa chọn/Test vào transformer.

Contract tối thiểu:

- `ReductionSpec`: mode `off | enabled | compare`, methods trong `pca | truncated_svd`, danh sách components hợp lệ, seed và lựa chọn scale theo feature family.
- `fit(X_fit, feature_schema, fit_scope)`; `transform(X)` theo batch; `inverse_transform` khi cần chẩn đoán; metadata input/output dimensions, EVR Train, chi phí và fingerprint.
- Adapter quản lý feature block nào được nén, block nào giữ lại; không áp PCA vào IDs thô, target, timestamp hoặc action mask.
- `off` giữ đường chạy gốc. `enabled` chạy cấu hình giảm chiều được chọn bằng Validation. `compare` thêm đối chứng nguyên bản trên cùng cohort/folds/seeds/budget. Mặc định cấu hình API cũ là `off`, UI cung cấp chế độ so sánh rõ ràng.
- Phương sai và reconstruction chỉ là chẩn đoán; chọn method/k bằng chỉ số tác vụ cuối và quy tắc tie-break đã công bố. Ngưỡng EVR không được loại cứng mọi ứng viên có chỉ số tác vụ tốt hơn.
- PCA: đầu vào dense số chiều nhỏ, scale/center theo Train. TruncatedSVD: có thể giữ sparse và scale không trừ mean khi phù hợp. Nếu cùng đầu vào đã center, ghi rõ hai phương pháp có thể cho kết quả gần nhau.
- Không âm thầm dùng IncrementalPCA thay PCA hoặc lấy mẫu fit mà vẫn báo fit toàn bộ; nếu cần biến thể, đặt tên/giao thức riêng và công bố số hàng fit.
- Không dựng dense user×item; dùng float32, batch/memmap và ngân sách bộ nhớ sẵn có. Tái sử dụng encoder trong một fold/variant, không dùng encoder fit từ fold tương lai.

### 3.5. Thiết kế cho từng nhóm và thuật toán

| Thuật toán | Tích hợp dự kiến | Chọn bằng Validation và đánh giá cuối |
|---|---|---|
| Linear Regression | Nén block đặc trưng sở thích/tương tác user–phim; giữ bias/thống kê nền. Đặc trưng tương tác phải biểu diễn được việc cùng phim hợp user này nhưng không hợp user khác | RMSE chính, MAE và chi phí phụ; raw và reduced dùng cùng nhãn/folds |
| XGBoost | Adapter đặc trưng tương tự Linear, giữ đường chạy gốc. Không mặc định PCA có lợi cho cây | RMSE/MAE, thời gian, kích thước; kết quả kém hơn phải được giữ |
| Logistic Regression | Nén block sở thích/tương tác; scaler/reducer fit riêng trong fold, threshold chọn trên Validation | F1 lớp Thích, AUC/Precision/Recall, cùng threshold protocol |
| Random Forest | Cùng feature family để so công bằng, giữ nhánh không giảm chiều | Như Logistic; không gán cải thiện giả định |
| Funk SVD | Giữ latent factors user–item nguyên bản; thêm biến thể **SVD hybrid + hồ sơ nén**, ví dụ `mu + b_u + b_i + p_u·q_i + z_u^T W g_i`, với `g_i` là genre phim và `z_u` từ hồ sơ hợp lệ | RMSE/MAE và chi phí, ghi rõ hybrid không phải TruncatedSVD thay Funk SVD; không nén ID |
| NCF | Biến thể hybrid ghép vector hồ sơ nén và metadata phim vào mạng tương tác; giữ embeddings nguyên bản, fit lại toàn bộ đầu vào/mạng trong fold | RMSE/MAE, early stopping và refit đúng epoch đã khóa |
| K-Means | Scale/reduce hồ sơ sở thích + tần suất của `TasteProfiles`, rồi fit K-Means; hồ sơ mô tả cụm vẫn trình bày theo genre gốc | Điều kiện số cụm/coverage rồi NDCG Validation; Silhouette/DBI là bổ sung |
| DBSCAN | Cùng biểu diễn như K-Means nhưng tìm lại eps/min_samples trong không gian mới; không tái dùng eps của không gian gốc | Coverage/noise/stability và future ranking; noise có fallback như hiện tại |
| DQN | Chỉ nén block hồ sơ sở thích/tần suất; giữ mask, bước còn lại, observed flag và thông tin động cần thiết. Mạng có kích thước đầu vào phù hợp từng variant | Reward mô phỏng Validation; đồng thời coverage, ranking, uncertainty và chi phí; giữ nguyên giới hạn mô phỏng |
| Random / Popularity | Không cần giảm chiều vì policy không đọc vector user. Hiển thị "Không áp dụng" và tái dùng kết quả đối chứng đúng cùng kịch bản | Không chạy lại rồi báo giả rằng giảm chiều làm policy thay đổi |
| Apriori | Không PCA transaction nhị phân. Tạo biến thể **PCA/SVD → phân nhóm hồ sơ user Train → Apriori theo nhóm**, cạnh Apriori toàn cục | Tách support trong nhóm/toàn cục, kích thước/coverage nhóm; downstream ranking trên cohort tương lai xác định riêng |

SVD/NCF hybrid phải triển khai bằng adapter/biến thể trong lab để hạn chế ảnh hưởng production. Không được đánh dấu hoàn tất tích hợp nếu chỉ hiện toggle nhưng model bỏ qua vector nén.

Apriori theo nhóm: giữ transaction rỗng trong mẫu số phù hợp; nhóm quá ít user/phim không khai phá hoặc fallback toàn cục và công bố số lượng. Fit phân nhóm/reducer/luật chỉ trên Train. User đánh giá chỉ được gán nhóm bằng lịch sử Train; user chưa có hồ sơ dùng global fallback. Nếu prepared split có nhiều cold user, hiển thị cohort và coverage, không âm thầm bỏ để tăng điểm. Không so trực tiếp support có mẫu số khác nhau như cùng một chỉ số.

### 3.6. Giao diện và báo cáo

- Giữ mục Dimensionality Reduction làm nơi giải thích/chẩn đoán nén độc lập. Thêm lựa chọn tác vụ đích và điều hướng có cấu hình sang thực nghiệm nhóm; kết quả reconstruction không được gọi là độ chính xác rating.
- Trong từng nhóm có khối "Giảm chiều hỗ trợ thuật toán": mục tiêu, đầu vào, đầu ra, vị trí trong pipeline, tùy chọn phương pháp/components và so sánh nguyên bản. Mô tả riêng cho SVD/NCF hybrid, Apriori phân nhóm và baseline không áp dụng.
- Pipeline trực quan phải phản ánh biến thể đang chọn: chia tập → đặc trưng Train → scaler/reducer → thuật toán → chọn Validation → Test. Không hiển thị refit/CV cho nhóm không thực hiện bước đó.
- Bảng kết quả phân biệt thuật toán, representation/variant, method, k, fit scope và split. Không đổi tên hiển thị rồi giữ `algorithm` bị trùng khiến `lookup[(algorithm, split)]` ghi đè.
- Thêm `variant_id` ổn định và `base_algorithm`; thống nhất cách dùng ở charts, best_configs, CV, metric references, CSV/XLSX/JSON. Không suy config từ prefix tên một cách mong manh.
- `schema_version=4` cho job/result mới và protocol RL/reduction riêng. Đọc/render/export lịch sử 2/3/4; không gắn định nghĩa metric mới lên report cũ.
- Bảng RL có reward/action, reward/episode, actual steps, observed-feedback coverage, cohort/user coverage và ranking nếu xác định được. Cumulative reward là số bổ sung vì phụ thuộc số bước.
- Báo uncertainty bằng chênh lệch policy ghép cặp theo episode/user và bootstrap theo user nếu có đủ user; không coi episode lặp cùng user là các người dùng độc lập. Không tuyên bố có ý nghĩa thống kê khi thiếu dữ liệu.
- Recall/NDCG trong candidate catalog giới hạn phải ghi `candidate_pool_size`, eligible positives và coverage; không gọi là đánh giá toàn bộ catalog. Thiếu positive/episode/action hợp lệ dùng N/A + lý do.
- Timing tách preprocessing, fit policy/model, checkpoint Validation, inference và tổng. Model bytes gồm các thành phần cần suy luận. Baseline báo thời gian khởi tạo, không giả làm thời gian học.
- Log dùng lại `RunLogger`/API cursor/`ExperimentLog`, thêm RL và PCA vào capability chung backend/frontend. Không tạo kênh log song song.
- Log RL gồm nguồn/protocol, số user/rating, support/query, catalog/cohort, sweeps/steps/epsilon/loss/coverage, Validation, checkpoint, Test, so baseline, thời gian/RAM, lưu kết quả, hủy/lỗi. Log reducer gồm fit scope, input→output dimensions, EVR Train, chi phí, downstream metric và cấu hình được chọn.
- Throttle log theo checkpoint hoặc thời gian; giữ phản hồi hủy trong bước dài và heartbeat hiện có. Không log mỗi hàng rating.

## 4. Hợp đồng cấu hình và kết quả cần chốt trước khi viết handler

Tên trường sau là thiết kế đề xuất, có thể chỉnh tên cho phù hợp code nhưng phải thống nhất Python/TypeScript/export/tests:

```json
{
  "reduction": {
    "mode": "compare",
    "methods": ["pca"],
    "components": [4, 8]
  },
  "rl": {
    "training_sweeps": 1,
    "max_training_steps": null,
    "action_limit": 30,
    "horizon": 5,
    "evaluation_users": 1000,
    "replay_capacity": 50000,
    "batch_size": 128,
    "checkpoint_interval_steps": 5000
  }
}
```

Phải bổ sung/validate learning rate DQN, gamma, epsilon start/end/decay, target update, replay warmup và hidden sizes. Learning rate Q-table cũ 0,2 không được dùng mặc định cho Adam DQN; mặc định mạng đề xuất 0,001. Alpha/gamma/epsilon cũ của report lịch sử giữ nguyên để đọc.

Metadata tối thiểu cho variant: `variant_id`, `base_algorithm`, `representation`, `reduction_method`, `components`, `input_dimensions`, `output_dimensions`, `feature_family`, `fit_scope`, `protocol`, `sample_sha256`, chi phí, selection metric và checkpoint đã khóa. Cache key phải chứa protocol, feature semantics, fold/fit fingerprint và ReductionSpec; không dùng chung cache chỉ vì cùng max_users.

## 5. Thứ tự thực hiện và checkpoint bàn giao

Mỗi checkpoint cập nhật `session-handoff-rl-reduction.md`: file thay đổi, test đã chạy, lệnh/kết quả, artifact thực và việc còn lại. Không coi xong UI là xong pipeline.

| Mốc | Việc cần thực hiện | Điều kiện qua mốc |
|---|---|---|
| A0 | Đọc lại code/working tree; chốt protocol, schema 4, variant IDs và config. Bổ sung tests schema/history trước thay đổi | Payload cũ có default hợp lệ; lịch sử 2/3 vẫn xuất hiện và mở được |
| A1 | Sửa loader RL sang temporal, dựng support/query episode và catalog/cohort/mask công bằng | Tests chứng minh timestamp, target exclusion, không lọc candidate bằng nhãn held-out, không trùng phim |
| A2 | DQN mạng nhỏ, replay/target, sweep coverage, checkpoint Validation, Test đóng băng, baseline | Test Bellman/terminal/mask/checkpoint/serialization; chạy dữ liệu MovieLens thật quy mô nhỏ, số liệu có provenance |
| A3 | UI RL, log đầy đủ, biểu đồ, hủy/lỗi/RAM/history/export | Theo dõi log trực tiếp, mở lại/tải log, hủy không xuất result dở. Kết thúc giai đoạn A trước tích hợp giảm chiều |
| B0 | Shared reducer, feature adapters, config/metadata/timing, log PCA; chẩn đoán độc lập | Tests fit isolation, dimensions/rank/no variance/cold profile/batch/serialization; off giữ hành vi gốc |
| B1 | Linear/XGBoost/Logistic/Forest: raw vs reduced, tuning, CV, refit, threshold | Mỗi fold fit mới reducer; đổi nhãn Test không đổi best config; có downstream metrics thực |
| B2 | SVD hybrid và NCF hybrid, hỗ trợ features hợp lệ và size accounting | Vector nén thực sự tham gia predict; gradient/fit hoạt động; baseline không bị thay nghĩa |
| B3 | K-Means/DBSCAN trên biểu diễn nén, tìm lại thông số và ranking | Coverage/noise/stability, profile genre gốc, common cohort và fallback được giữ |
| B4 | Ghép reducer vào block hồ sơ của DQN đã sửa ở A | Raw/reduced dùng đúng cùng episode/cohort/budget; mask và trạng thái động không bị nén |
| B5 | Apriori theo nhóm và đối chứng global, mô tả metric/coverage rõ | Không PCA basket; support đúng mẫu số, fallback và held-out ranking có provenance |
| C0 | Hoàn thiện mô tả từng thuật toán, pipeline UI, schema/history/chart/export/translations | Không trùng variant; JSON/CSV/XLSX chứa đúng cấu hình/phương pháp/k/split; mobile và dark/light dùng được |
| C1 | Hồi quy toàn dự án, benchmark thật có giới hạn, tài liệu cuối | Báo rõ cái đã kiểm chứng, số liệu thật, giới hạn RAM/coverage và phần chưa đạt; không tự công bố cải thiện |

Các file chính dự kiến:

- Backend: `experiment_config.py`, `experiment_data.py`, `experiment_pipeline.py`, `experiment_reinforcement.py`, `experiment_supervised.py`, `experiment_clustering.py`, `experiment_unsupervised.py`, module reducer mới, model/adapter lab mới, `lab.py`, `lab_catalog.json`, `experiment_logs.py`, `experiment_worker.py` nếu cần.
- Frontend: `pages/Experiments.tsx`, `components/ExperimentConfig.tsx`, `components/ExperimentResults.tsx`, `components/ExperimentLog.tsx`, `components/ExperimentCharts.tsx`, `components/MetricReferences.tsx`, `lib/lab.ts`, metric references và bản dịch tương ứng. Tách component giải thích/pipeline nếu file hiện tại quá lớn.
- Validation: thêm `tests/test_experiment_reinforcement.py`, `tests/test_experiment_reduction.py`, tests adapters/hybrids khi phù hợp; mở rộng `test_lab.py`, `test_experiment_logs.py`, `test_personalized_experiments.py` và frontend scripts.
- Documentation: cập nhật `experiment-studio.md`, `personalized-experiments.md`, `metric-reference-guide.md`; viết tài liệu mới về RL và tích hợp giảm chiều. Không sửa số liệu báo cáo cũ thành số liệu mới.

## 6. Kiểm thử bắt buộc

1. Temporal split đầy đủ/lấy mẫu của RL: không IID, timestamp ties không bị tách, cold/excluded cohorts công bố đúng.
2. Support/query Train: thay rating query không làm đổi state ban đầu; query item không bị tiết lộ bằng candidate filter.
3. DQN: terminal không bootstrap; invalid action không được chọn và không vào max target; all-invalid không sinh NaN; checkpoint là bản sao độc lập và được phục hồi thật.
4. Eval: cùng episode IDs/cohort/mask cho policy; Test không cập nhật optimizer/replay/scaler/reducer; episodes thiếu bước dùng denominator thực; unknown observed flag không nhầm dislike.
5. Đổi nhãn Test không đổi cấu hình/reducer/checkpoint/chọn policy. Validation chỉ ảnh hưởng lựa chọn ở những bước đã khai báo.
6. Reducer chỉ thấy đúng fit indices trong mỗi fold; cache không xuyên fold; Train target không có trong feature row; held-out label không đi vào feature query.
7. Chế độ off giữ đường chạy cũ; enabled/compare thật sự dùng vector nén trong từng adapter; shapes/rank/zero variance/bad components/cold users được xử lý rõ.
8. SVD/NCF hybrid có dự đoán thay đổi hợp lý khi thay đặc trưng nén trong fixture có tín hiệu, roundtrip serialization giữ dự đoán; không kiểm tra bằng việc lặp lại implementation.
9. DBSCAN retune theo representation; Apriori support đúng mẫu số, không mất transaction rỗng; không có user/positive hợp lệ thì N/A có lý do.
10. Log paging/Unicode/reopen/download/concurrent worker, cancel/failure/memory stop và không xuất bản kết quả dở cho RL/reduction.
11. History schema 2/3/4 và exports có đủ variant/reduction/protocol; biểu đồ không ghi đè hàng khi nhiều k/method.
12. Chạy tất cả tổ hợp được hỗ trợ trên fixture, rồi pilot MovieLens thật có số user/budget công bố. Kiểm tra đường full-data và giới hạn RAM với cấu hình nhỏ; không suy ra benchmark đủ 25M nếu chỉ chạy sample. Không đặt điều kiện phải thắng baseline để tests qua.

Lệnh trong thư mục `project`:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest tests -q
.\.venv\Scripts\python.exe -m ruff check src tests
```

Chạy lint trước trên file thay đổi; nếu chạy toàn dự án gặp lỗi có sẵn, phân biệt lỗi có sẵn và lỗi mới, không sửa lan rộng ngoài phạm vi.

Lệnh trong `project/frontend`:

```powershell
npm run build
npm run test:experiments
npm run test:experiment-logs
npm run test:association
npm run test:metric-references
```

Đọc yêu cầu backend/port của từng script trước chạy. Không restart/kill server đang có job của người dùng; dùng instance/port riêng nếu script hỗ trợ. Khi Node không có trên PATH, tham khảo khả năng tìm Node trong launcher hiện tại thay vì sửa launcher ngoài phạm vi.

## 7. Điều kiện hoàn thành và báo cáo cuối

- Hoàn thành A trước B; tất cả nhóm/adapter trong bảng có triển khai thực và mô tả đúng phạm vi hỗ trợ.
- RL có luồng train/Validation/checkpoint/Test, log và lifecycle thống nhất; coverage thấp phải nhìn thấy, không bị che bởi một reward lớn hơn.
- Mọi so sánh raw/reduced cùng cohort và giao thức; cấu hình đã khóa trước Test. Chi phí preprocessing/fit/eval không bị trộn sai nghĩa.
- Không có feature/target leakage theo tests; history cũ giữ nguyên; không ghi đè dữ liệu/model production.
- Cập nhật checklist và report các lệnh test, run IDs, cấu hình và kết quả thực, hạn chế còn lại. Không đánh dấu hoàn tất chỉ vì thêm lựa chọn UI hoặc chỉ chạy một nhóm thành công.

## 8. Nguồn phương pháp

- [PyTorch: DQN, replay memory và target network](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).
- [scikit-learn: Pipeline và composite estimators](https://scikit-learn.org/stable/modules/compose.html).
- [scikit-learn: tránh data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
- [scikit-learn: TruncatedSVD và centering](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).
- [Li et al.: giới hạn và điều kiện đánh giá policy offline từ phản hồi từng phần](https://arxiv.org/abs/1003.5956).

Các đường dẫn tài liệu online có thể đang mô tả phiên bản thư viện mới hơn môi trường dự án; session triển khai phải đối chiếu API với phiên bản đã cài, không nâng dependency chỉ để khớp ví dụ.

## 9. Kết quả follow-up và đóng checklist — 20/11/2026

Hai giai đoạn A rồi B đã triển khai. Follow-up sửa thêm các lỗi phát hiện khi chạy
toàn pipeline: NCF hybrid tách đường cong correction khỏi rating MSE và khóa epoch
được chọn bằng Validation cho CV/refit; K-Means và DBSCAN có representation tìm
tham số và representation fit cuối độc lập; TruncatedSVD báo phương sai thật của
component chiếu chia cho tổng phương sai input đã scale. PCA được ẩn khỏi năm nhóm
chính nhưng route lịch sử và export cũ vẫn đọc được. Hướng dẫn PCA/TruncatedSVD
song ngữ đặt sau thẻ thuật toán và phản ánh mode/method/components đang chọn.

Kiểm chứng bổ sung: test pipeline NCF chạy ở `enabled` và `compare`; test
clustering chạy riêng/từng thứ tự K-Means/DBSCAN cho PCA và TruncatedSVD; phép đo
TruncatedSVD được đối chiếu với scikit-learn trên input có mean khác 0. Browser
test mới kiểm tra năm nhóm, thứ tự/hành vi guide, responsive, song ngữ và route
PCA lịch sử với CSV/XLSX/JSON. Kết quả cùng lần chạy MovieLens và số kiểm thử cuối
được ghi trong [bàn giao](session-handoff-rl-reduction.md) và
[báo cáo](rl-dimensionality-reduction-report.md).

Giới hạn vận hành còn lại: real-worker browser suite không được khởi động lại ở
follow-up vì RAM sẵn có thấp hơn ngưỡng guard của worker; không bỏ qua guard.
Pilot được chạy trực tiếp qua `experiment_pipeline.execute`, không tạo run ID hay
giả lập log worker. Các thay đổi người dùng trong `run_project.bat` và
`project/scripts/start.ps1` được giữ nguyên.
