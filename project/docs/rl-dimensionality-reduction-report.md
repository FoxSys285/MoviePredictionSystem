# Báo cáo triển khai Học tăng cường và giảm chiều

Ngày hoàn tất kiểm chứng: 20/11/2026. Phạm vi: Experiment Studio trong
`project/`; không thay mô hình gợi ý production hay dữ liệu đã chuẩn bị.

## Kết quả triển khai

### Giai đoạn A — Học tăng cường

- Job mới dùng ID `dqn`; Q-table `q_learning` trong schema 2/3 vẫn đọc như kết
  quả lịch sử. Mạng DQN nhận profile dùng chung, không nhận user ID.
- Loader RL dùng `per_user_temporal_70_15_15_v2`. Episode Train chia support/query
  theo timestamp; profile ban đầu chỉ fit từ support. Catalog, candidate mask,
  episode IDs và seed được chia sẻ giữa các representation/policy.
- DQN có target network, replay buffer hữu hạn, Bellman target có action mask,
  terminal không bootstrap, Huber loss, gradient clipping và epsilon schedule.
  Snapshot checkpoint gồm trọng số và reducer; Validation chọn checkpoint, Test
  dùng checkpoint đóng băng.
- Random/Popularity chạy cùng episode để đối chiếu. Kết quả có observed feedback,
  coverage, ranking, model size, chi phí, cohort, exclusion và bootstrap ghép cặp.
  Reward thiếu phản hồi bằng 0 theo quy ước mô phỏng, không phải dislike hay
  đánh giá counterfactual.
- RL và PCA vào chung worker log/lifecycle, lịch sử, JSON/CSV/XLSX; log ghi rõ
  support/query, reducer, checkpoint Validation và từng lượt đánh giá đóng băng
  Validation/Test.

### Giai đoạn B — reducer và adapter

Reducer dùng chung ở `experiment_reduction.py` hỗ trợ PCA đã center và
TruncatedSVD không center, fit theo batch trên ma trận đặc trưng có số chiều nhỏ,
kiểm tra rank/variance/shape, `transform`/`inverse_transform`, metadata và
fingerprint. Variant ID ổn định nối xuyên bảng, CV, chart, log và export.

| Nhóm / thuật toán | Dữ liệu được nén | Chọn và đánh giá |
|---|---|---|
| Linear Regression, XGBoost | Block tương tác user-preference × genre phim; giữ bias/thống kê nền | RMSE Validation; reducer fit lại trong mỗi fold, refit Train + Validation rồi Test |
| Logistic Regression, Random Forest | Cùng block tương tác; scaler/reducer fit riêng trong Train/fold | F1/threshold Validation; refit theo cấu hình đã khóa |
| Funk SVD hybrid, NCF hybrid | Profile user nén làm nhánh hiệu chỉnh/MLP; giữ latent factors/embeddings | RMSE Validation; NCF early stopping/refit theo epoch đã khóa |
| K-Means, DBSCAN | Profile sở thích tương đối + tần suất user | Tìm lại K/eps trong representation mới; coverage/stability và NDCG Validation; giữ diễn giải genre gốc |
| DQN | Profile support sở thích/tần suất | Cùng episode/cohort/budget; mask/trạng thái động giữ nguyên; Validation chọn checkpoint/variant |
| Random, Popularity | Không áp dụng: policy không đọc profile | Dùng một baseline chung, không nhân bản theo representation |
| Apriori | Profile user Train để phân nhóm; transaction boolean giữ nguyên | Apriori theo nhóm và baseline toàn cục; support theo mẫu số riêng, NDCG Validation, fallback và candidate coverage |
| PCA, TruncatedSVD diagnostic | Profile user–genre | EVR và reconstruction là chẩn đoán; không gọi là độ chính xác rating |

Chế độ API mặc định là `off`. `enabled` chạy representation được chọn;
`compare` thêm raw control. Phương pháp/số chiều được chọn bằng metric tác vụ trên
Validation, không bằng ngưỡng EVR. Khi chiều yêu cầu vượt rank/rows/features hoặc
feature family không có phương sai, reducer trả lỗi có thông tin thay vì sinh
kết quả giả.

## Kiểm chứng

Các fixture kiểm tra support/query, action mask, Bellman terminal, checkpoint
độc lập, frozen evaluation, serialization, split/fold isolation, Test-label
invariance, Logistic/Forest, Linear/XGBoost, SVD/NCF hybrid, K-Means/DBSCAN,
Apriori support/denominator/fallback/ranking, lịch sử schema 2/3/4 và export
grouped rules.

Các browser script chạy backend riêng trên port 8139–8141, dùng Chromium/Edge
headless và ghi ảnh/run manifest vào `frontend/test-results/` (artifact test,
không phải benchmark production). Sáu nhóm được mở lại từ lịch sử và tải
CSV/XLSX/JSON. Nhóm association có thêm full-Train test cho profile PCA, Apriori
theo nhóm, global baseline, support/fallback, log và export. Nhóm RL có test
log live và so sánh DQN raw/PCA trên sample MovieLens. Bài browser sáu nhóm dùng
poll đúng run ID; mỗi lượt chỉ được kết luận sau khi job đó ở trạng thái cuối.

Kiểm chứng cuối ngày 20/11/2026:

- `python -X utf8 -m pytest tests -q`: **206 passed, 1 skipped**, 15 warnings,
  70,60 giây.
- `ruff check --ignore E501` trên toàn bộ file Python/test đã sửa: **All checks
  passed**. Ruff toàn repository còn 364 mục (360 E501, 3 I001, 1 F841).
- `npm run build`: thành công; Vite còn cảnh báo bundle JavaScript lớn hơn
  500 kB.
- `npm run test:experiments`: sáu nhóm, training thật, CV, lịch sử, export,
  mobile/dark-light và xử lý lỗi đều qua.
- `npm run test:experiment-logs`: log live, paging/scroll, pause/follow,
  reconnect, download và group scope đều qua.
- `npm run test:association`: full prepared Train, PCA profile, 6.944 luật nhóm,
  baseline, ranking, fallback, log và export đều qua.
- `npm run test:metric-references`: toàn bộ kiểm tra metric reference qua.

## Pilot MovieLens thật

Chạy DQN `compare`, PCA 1 component, 40 user, một sweep, 100 training steps,
40 Validation/Test user, action limit 30, horizon 5. Raw/PCA có cùng SHA256 episode
IDs cho Validation/Test và cùng budget. Dùng 8 transition replay warmup và batch 8.

| Variant/policy | Validation reward/action | Test reward/action | Test observed feedback | Test model bytes |
|---|---:|---:|---:|---:|
| DQN raw | 0,01731 | 0,02513 | 3,52% | 50.081 |
| DQN PCA-1 | 0,01117 | 0,01173 | 2,51% | 42.043 |
| Random | 0,02680 | 0,01340 | 2,01% | 0 |
| Popularity | 0,01396 | 0,04355 | 6,03% | 0 |

Validation chọn Random trong pilot này; Test không tham gia lựa chọn. Test paired
bootstrap so Popularity cho DQN raw có mean delta −0,01833, CI95
[−0,04557; 0,01500]; DQN PCA-1 có mean delta −0,03167, CI95
[−0,06111; −0,00333]. Chỉ 21/40 user Train đi qua 100 step budget
(training-user coverage 52,5%). Coverage phản hồi Test chỉ 2,01–6,03%, nên
reward là phép đo mô phỏng rất thưa, không chứng minh hiệu quả policy. Kết quả
này không cho thấy giảm chiều hay DQN thắng baseline.

Run DQN: `20261120T041717-4d86bf6c`, artifact tại
`artifacts/experiment-lab/20261120T041717-4d86bf6c/`. Run này dùng sample
3.657/791/789 rating ở Train/Validation/Test từ 40 user.

Đã chạy thêm Apriori nhóm PCA-1 trên toàn bộ prepared Train, run
`20261120T043856-fd7cd06f`: 17.500.066 rating, 122.029 user, 80 phim catalog,
121.983 profile user được reducer fit (46 bị loại vì không đủ hồ sơ), 38→1 chiều.
Global baseline khai phá 3.495 luật; hai nhóm tạo 6.944 luật. NDCG@10 Validation
là 0,33653 cho global và 0,33448 cho grouped; Test lần lượt 0,28259 và 0,28171.
Grouped assignment coverage là 15,35% Validation và 11,09% Test; fallback về luật
toàn cục/Popularity là 1.688 và 1.771 user. Candidate-positive coverage lần lượt
17,57% và 15,42%. Test chỉ để xác nhận cuối; Validation chọn global baseline.
Kết quả này xác nhận đường chạy full-Train, không phải bằng chứng Apriori nhóm
cải thiện ranking. Worker ghi peak RSS 1.214 MiB cho baseline global; tổng fit
biến thể nhóm mất 32,17 giây, reducer fit 0,029 giây.

Run IDs cuối cùng trong browser smoke test: rating
`20261120T043459-c2d74df3`, classification `20261120T043526-b274fccf`, clustering
`20261120T043541-b0cb4bab`, association toàn cục `20261120T043553-37062242`, PCA
`20261120T043604-1b4eec11`, reinforcement `20261120T043615-6c01fa08`. DQN live
log run là `20261120T043755-94cd646d`; association grouped full-Train run là
`20261120T043856-fd7cd06f`. Manifest tương ứng ở
`frontend/test-results/lab-real-runs.json`,
`frontend/test-results/experiment-logs-check.json` và
`frontend/test-results/association-check.json`. Runtime artifacts và screenshot
được giữ trong workspace để xem lại, không thêm vào git.

## Follow-up sửa lỗi và kiểm chứng — 20/11/2026

Trong lần tích hợp cuối, ba lỗi pipeline được sửa. NCF hybrid dùng schema riêng
cho curve correction, giữ chart raw ở Rating MSE, đồng thời khóa epoch base và
correction theo Validation trước CV/refit. K-Means/DBSCAN giờ fit representation
selection và final riêng theo từng thuật toán, vì vậy chọn cả hai hoặc đảo thứ tự
không làm candidate/final fit nhiễm lẫn. TruncatedSVD tiếp tục transform không
center; explained variance là phương sai của component chiếu trên input đã scale
chia cho tổng phương sai input, được đối chiếu với scikit-learn trên dữ liệu có
mean khác 0.

PCA không còn là nhóm chính để tạo run mới. Route `group=pca` vẫn mở lịch sử và
result cũ, gồm export CSV/XLSX/JSON. Guide song ngữ PCA/TruncatedSVD được dùng
chung cho năm nhóm chính, đặt sau thẻ thuật toán và phản ánh cấu hình reducer.
Browser test fixture bao phủ thứ tự, trạng thái, responsive, ngôn ngữ và route
lịch sử này.

### Pilot pipeline sau sửa lỗi

Chạy trực tiếp `experiment_pipeline.execute` trên sample MovieLens 40 user / 5.237
ratings theo `per_user_temporal_70_15_15_v2` (Train/Validation/Test:
3.657/791/789 ratings). Đây là lần chạy kiểm tra adapter, không qua UI worker và
không có run ID.

| NCF variant | Test RMSE | Ghi chú |
|---|---:|---|
| raw | 0,999401 | Validation chọn PCA-1; raw curve là Rating MSE |
| PCA-1 hybrid | 1,000999 | Correction MSE riêng; không chọn bằng Test |
| TruncatedSVD-1 hybrid | 0,999419 | Correction MSE riêng; không chọn bằng Test |

Validation chọn `ncf_pca_1` với RMSE 0,9483; mỗi variant Test được báo riêng.
K-Means và DBSCAN cùng chạy trong `compare` với raw/PCA-1/TruncatedSVD-1, tổng
12 lượt tìm cấu hình. DBSCAN PCA-1 được chọn trong các phương án đủ điều kiện
(Validation NDCG@10 0,049711). Trên Test, DBSCAN PCA-1 và DBSCAN TruncatedSVD-1
cùng có coverage 0,925 và 5 cụm; K-Means không đạt điều kiện qualification của
clustering selection. Các số này chỉ mô tả sample, không chứng minh lợi ích giảm
chiều. JSON chi tiết: `frontend/test-results/rl-reduction-pilot-followup.json`
trong workspace.

### Kiểm chứng follow-up

- Backend toàn suite: **217 passed, 1 skipped, 21 warnings in 142.09s**; NCF `execute()` test bao phủ
  `enabled` và `compare`, clustering order/algorithm subset **42 passed**.
- Ruff file Python/test đã sửa (bỏ qua E501): **All checks passed**.
- `npm.cmd run build`: pass; Vite báo bundle 520 kB vượt ngưỡng cảnh báo 500 kB.
- `npm.cmd run test:reduction-guide`: pass cho năm nhóm, icon theo ID, guide
  đúng vị trí, mode/method/components, song ngữ, mobile và PCA history exports.
- `node --check` các browser script liên quan: pass.

Real-worker browser suite không được chạy lại vì RAM khả dụng thấp hơn guard của
worker (xấp xỉ 2,9 GiB). Guard được giữ nguyên; pilot dùng lời gọi pipeline trực
tiếp. Launcher người dùng `run_project.bat` và `project/scripts/start.ps1` được
bảo toàn.

## Giới hạn

- Pilot RL giới hạn 40 user và 100 bước; không phải benchmark toàn bộ MovieLens
  25M. Không suy rộng reward hoặc coverage của sample này ra sản phẩm.
- Reward offline chỉ đọc rating được quan sát, không có propensity/IPS. Phản hồi
  thiếu được quy ước bằng 0; độ thưa được trình bày kèm kết quả.
- PCA/TruncatedSVD chạy trên feature family nhỏ (genre/profile), không dựng ma
  trận dense user×item. Reducer giữ covariance theo chiều đặc trưng nên chi phí
  fit tăng theo bình phương số cột.
- Luật theo nhóm có thể thiếu luật; nhóm rỗng/nhỏ fallback về luật toàn cục rồi
  Popularity. Phim ngoài catalog không gợi ý được và được tính vào candidate
  coverage.
- Ruff repository-wide hiện chưa sạch: 364 mục gồm 360 `E501`, 3 `I001` và
  1 `F841`. Tất cả file Python/test trong phạm vi thay đổi qua Ruff khi bỏ qua
  `E501`; không mở rộng chỉnh sửa lint ra ngoài phạm vi tính năng.

## Tài liệu liên quan

- [Experiment Studio pipeline v4](experiment-studio.md)
- [Split cá nhân hóa, log và RAM](personalized-experiments.md)
- [Association và Apriori theo nhóm](association-experiments.md)
- [Metric reference](metric-reference-guide.md)
- [Kế hoạch ban đầu](implementation-plan-rl-reduction.md)
- [Bàn giao và nhật ký kiểm chứng](session-handoff-rl-reduction.md)
