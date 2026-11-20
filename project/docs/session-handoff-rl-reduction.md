# Bàn giao cho session triển khai RL và giảm chiều

Ngày cập nhật: 20/11/2026. Workspace: `E:\.HUIT\Học máy\MoviePredictionSystem`.

## Yêu cầu người dùng

Session trước lập kế hoạch; session hiện tại được yêu cầu thực hiện theo thứ tự: sửa nhóm Học tăng cường trước, có log và quy trình như các nhóm khác; sau đó đưa giảm chiều vào thực nghiệm từng nhóm/thuật toán theo thiết kế phù hợp, có mô tả hỗ trợ, so sánh và kiểm chứng. Công việc dưới đây đã hoàn tất và có report cùng run artifacts để xem lại.

Đọc tài liệu chính trước khi triển khai: [implementation-plan-rl-reduction.md](implementation-plan-rl-reduction.md).

## Trạng thái thực tế

- [x] Audit handler đang chạy, loader, dữ liệu Parquet, Q-table, giảm chiều, log backend/frontend và tests liên quan.
- [x] Lập kế hoạch, hợp đồng dự kiến, thứ tự mốc và tiêu chí nghiệm thu.
- [x] A0: schema/config/variant/history.
- [x] A1: dữ liệu temporal, support/query, cohort/mask RL.
- [x] A2: DQN chia sẻ đặc trưng, checkpoint và đánh giá đóng băng.
- [x] A3: giao diện/log/lifecycle/export RL.
- [x] B0: shared reducer và adapter contract.
- [x] B1: Linear/XGBoost/Logistic/Random Forest.
- [x] B2: SVD hybrid/NCF hybrid.
- [x] B3: K-Means/DBSCAN.
- [x] B4: giảm chiều hỗ trợ DQN.
- [x] B5: Apriori theo nhóm và baseline toàn cục.
- [x] C0: UI/mô tả/history/schema/export/translations.
- [x] C1: regression tests, pilot dữ liệu thật, tài liệu và báo cáo cuối.

Các mốc A→B→C đã hoàn tất theo đúng thứ tự. Kết quả chi tiết, phương pháp,
pilot và giới hạn: [rl-dimensionality-reduction-report.md](rl-dimensionality-reduction-report.md).

## Trình tự triển khai đã thực hiện

1. Đọc hướng dẫn và kiểm tra working tree. Giữ nguyên thay đổi người dùng ở `run_project.bat` và `project/scripts/start.ps1`.
2. Đọc `experiment_pipeline.py`, `experiment_data.py`, `experiment_reinforcement.py`, `experiment_config.py`, `experiment_logs.py`, phần lifecycle/history/export trong `lab.py`, và `frontend/src/pages/Experiments.tsx`.
3. Thực hiện A0→A3 trước. Không làm riêng logging rồi coi đã sửa thiết kế RL. Không bắt đầu bằng việc chỉnh `lab_reinforcement.py` cũ vì API chạy handler `experiment_reinforcement.py`.
4. Sau khi A qua kiểm thử, triển khai B theo adapter từng nhóm. Không áp PCA vào transaction Apriori, raw user IDs, hoặc thay Funk SVD bằng TruncatedSVD.
5. Cập nhật handoff/report bằng bằng chứng kiểm chứng và run IDs sau khi hoàn tất.

## Các điểm đã xử lý và cần giữ khi bảo trì

- Log cho cả sáu nhóm đã được khai báo tại `experiment_logs.py:LOGGED_GROUPS` và render trong `Experiments.tsx`; bài browser log kiểm tra reconnect, paging và tải xuống.
- History/backend/frontend đọc schema 2/3/4; job mới dùng schema 4 và giữ Q-table cũ như lịch sử.
- Loader dùng split thời gian từng user cho RL; luồng sample không gom rồi chia IID lại. Không sửa Parquet/manifest đã chuẩn bị.
- Job mới dùng ID `dqn`, còn `q_learning` giữ tên và protocol cũ. DQN không nhận raw user ID; cấu hình dùng learning rate cho mạng, không kế thừa learning rate 0,2 của Q-table.
- Rating thiếu được gắn cờ không quan sát, không suy thành dislike. Candidate held-out không lọc theo việc có rating.
- Feature Train loại nhãn của chính hàng; episode Train khởi tạo từ support, không dùng query reward trước khi hành động.
- Validation/Test đóng băng trọng số, scaler và reducer; phản hồi tuần tự chỉ tồn tại trong episode đang đánh giá.
- Raw/reduced dùng chung cohort/fold/episode/seed; không yêu cầu reduced thắng. Variant IDs phân biệt bảng, CV, chart, log và export.
- Action mask, bước còn lại và observed flag được giữ nguyên ngoài khối được giảm chiều. Training báo số bước/user thật sự đạt ngân sách.
- Reducer fit theo batch, không dựng ma trận dense user×item; replay buffer hữu hạn. Pilot cho biết coverage Train 21/40 user.
- SVD/NCF có nhánh hybrid dùng profile nén trong dự đoán; baseline latent factor/embedding vẫn giữ nguyên.
- Apriori giữ transaction boolean; báo support trong nhóm/toàn cục bằng mẫu số riêng và fallback theo luật global rồi Popularity.

## Kết quả kiểm tra từ phiên audit ban đầu

Lệnh đã qua trong `project`:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_lab.py -q -k "(six_groups and (pca or reinforcement)) or user_genre or shared_split or offline_rewards or full_data_mode or sampled_data_mode"
```

Kết quả: 7 passed, 29 deselected. Đây là kiểm tra audit trước triển khai, không
phải bằng chứng cho giao thức mới; tests theo mục 6 được bổ sung và chạy trong
nhật ký hoàn tất bên dưới.

Tại thời điểm audit ban đầu đã kiểm tra mẫu MovieLens 40 user, RL 50/500 episode
và thống kê split đầy đủ; chưa benchmark toàn bộ 25M theo giao thức mới và chưa
có model/checkpoint mới để tiếp tục huấn luyện.

## Nhật ký triển khai — hoàn tất 20/11/2026

### Giai đoạn A — Học tăng cường

- Files chính: `src/movie_rating/experiment_config.py`, `experiment_data.py`,
  `experiment_reinforcement.py`, `experiment_pipeline.py`, `experiment_logs.py`,
  `lab.py`; UI ở `frontend/src/components/ExperimentConfig.tsx`,
  `ExperimentResults.tsx` và `frontend/src/pages/Experiments.tsx`.
- Thay Q-table mới cho job mới bằng DQN `dqn`; vẫn đọc Q-table `q_learning` lịch
  sử. Train chia support/query theo thời gian; DQN dùng profile chung, action
  mask, replay/target network, checkpoint Validation và Test đóng băng. Random /
  Popularity dùng chung cohort/episode. Log live ghi reducer, Validation/Test,
  observed feedback và coverage.
- Pilot raw/PCA cùng SHA256 episode IDs, 40 user, 100 steps, horizon 5, catalog
  30. DQN compare run `20261120T041717-4d86bf6c`; 21/40 Train user được học trong
  step budget (52,5%). Validation chọn Random; reward Test thưa (observed
  feedback 2,01–6,03%). Không kết luận DQN hay PCA thắng baseline.

### Giai đoạn B — giảm chiều theo adapter

- Reducer chung: `src/movie_rating/experiment_reduction.py`; adapter ở
  `experiment_supervised.py`, `experiment_clustering.py`,
  `experiment_unsupervised.py` và `experiment_reinforcement.py`.
- Linear/XGBoost và Logistic/Forest nén block tương tác, fit theo fold; SVD/NCF
  có nhánh hybrid dùng profile nén; K-Means/DBSCAN refit/tune trong biểu diễn
  mới; Apriori nhóm PCA nén profile user để phân nhóm, giữ transaction boolean.
  DQN raw/reduced dùng cùng episode. PCA/SVD diagnostic vẫn tách khỏi metric
  downstream.
- Full-Train grouped Apriori run `20261120T043856-fd7cd06f`: 17.500.066 rating,
  122.029 user, 6.944 luật nhóm. NDCG@10 Validation global/grouped 0,33653 /
  0,33448; Test 0,28259 / 0,28171. Nhóm có coverage 15,35% / 11,09%; fallback
  1.688 / 1.771 user. Validation giữ global baseline; đây là kiểm chứng đường
  chạy, không phải kết quả cải thiện.

### Giai đoạn C — UI, tài liệu và kiểm chứng

- History/schema v2/3/4, variant ID trong bảng/CV/chart/export, grouped rules,
  protocol/fit scope, mô tả tiếng Anh/Việt, catalog và metric references đã cập
  nhật. Các tài liệu liên quan và report mới đã cập nhật; không đổi số liệu báo
  cáo lịch sử.
- `python -X utf8 -m pytest tests -q`: **206 passed, 1 skipped**, 15 warnings.
- Ruff trên file Python/test đã sửa với `--ignore E501`: **All checks passed**.
  Toàn repository còn 364 mục Ruff (360 E501, 3 I001, 1 F841).
- `npm run build`: pass; có cảnh báo bundle >500 kB. `npm run test:experiments`,
  `npm run test:experiment-logs`, `npm run test:association` và
  `npm run test:metric-references`: đều pass.
- Sáu group browser run IDs: rating `20261120T043459-c2d74df3`, classification
  `20261120T043526-b274fccf`, clustering `20261120T043541-b0cb4bab`, association
  global `20261120T043553-37062242`, PCA `20261120T043604-1b4eec11`, RL
  `20261120T043615-6c01fa08`. Live-log RL run
  `20261120T043755-94cd646d`; manifests ở `frontend/test-results/`.
- User edits trong `run_project.bat` (`chcp 65001`) và
  `project/scripts/start.ps1` (readiness timeout 120 giây cùng status message)
  được giữ nguyên.

Giới hạn còn lại: pilot DQN giới hạn 40 user/100 bước và reward offline không có
propensity/IPS; Apriori grouped có coverage thấp và không thắng global baseline
ở lần chạy này; repository-wide Ruff chưa sạch như số liệu phía trên. Không có
benchmark đầy đủ cho mọi cấu hình trên toàn MovieLens 25M.

## Follow-up kiểm chứng — 20/11/2026

### Sửa lỗi và hành vi giao diện

- NCF giảm chiều từng phát lỗi schema `train_rating_mse` khi chart dùng curve
  của nhánh hybrid. Curve raw tiếp tục là Rating MSE; curve hybrid ghi rõ
  Correction MSE. Epoch của base NCF và correction branch được chọn độc lập trên
  Validation, rồi khóa cho CV/refit. Test `execute()` ở `enabled`/`compare` xác
  nhận log chart và epoch truyền vào.
- K-Means/DBSCAN từng chia sẻ biểu diễn cuối sau khi quét candidate, khiến kết
  quả phụ thuộc thứ tự danh sách thuật toán. Selection space và final-fit space
  nay được fit riêng theo từng thuật toán. Fixture PCA/TruncatedSVD kiểm tra mỗi
  thuật toán độc lập, kết hợp và đảo thứ tự.
- TruncatedSVD vẫn không center input; explained variance nay là phương sai
  `X @ components.T` chia cho tổng phương sai từng feature của input đã scale.
  Test đối chiếu trực tiếp `explained_variance_` của scikit-learn trên input có
  mean khác 0.
- Sidebar còn năm nhóm chính. PCA cũ là trang history-only; không tạo job PCA
  mới từ route cũ. Guide giảm chiều dùng chung, song ngữ, nằm giữa thẻ thuật toán
  và cấu hình/metric; trạng thái card bám mode, method và số components.

### Kết quả kiểm chứng follow-up

- Backend suite sau follow-up: **217 passed, 1 skipped, 21 warnings in 142.09s**;
  targeted NCF pipeline
  test xác nhận cả hai mode `enabled`/`compare`; clustering subset: **42 passed**.
- Ruff các file Python/test đã sửa với `--ignore E501`: **All checks passed**.
- Frontend: `npm.cmd run build` thành công (cảnh báo bundle JS 520 kB); browser
  test `npm.cmd run test:reduction-guide` pass: năm nhóm, icon theo group ID,
  vị trí/active state, song ngữ, mobile và link export PCA history CSV/XLSX/JSON.
  `node --check` các script browser liên quan cũng pass.
- Pilot trực tiếp qua `experiment_pipeline.execute`: MovieLens 40 user, 5.237
  ratings, split temporal `per_user_temporal_70_15_15_v2` với 3.657/791/789
  Train/Validation/Test. NCF compare raw/PCA-1/TruncatedSVD-1 Test RMSE lần lượt
  0,999401 / 1,000999 / 0,999419; Validation chọn PCA-1 (RMSE 0,9483). Raw
  chart ghi Rating MSE; hai hybrid charts ghi Correction MSE.
- Cùng sample, K-Means + DBSCAN compare chạy 12 lượt tìm cấu hình; DBSCAN PCA-1
  được chọn theo Validation NDCG@10 0,049711 trong các phương án đủ coverage/số
  cụm. Test coverage của DBSCAN PCA-1 và TruncatedSVD-1 là 0,925; cả hai có 5
  cụm. Test chỉ dùng báo cáo cuối. Không kết luận giảm chiều tốt hơn từ pilot
  nhỏ này. Kết quả lưu tại
  `frontend/test-results/rl-reduction-pilot-followup.json` trong workspace,
  không có run ID vì không chạy qua UI worker.

Real-worker browser suite không được chạy lại trong follow-up: bộ nhớ khả dụng
thấp hơn mức guard khoảng 2,9 GiB; worker bị guard từ chối và không được bypass.
Các browser runs đã ghi trong phần trước vẫn là bằng chứng cho log lifecycle của
UI. Launcher không bị chỉnh sửa trong follow-up; nội dung người dùng tại
`run_project.bat` và `project/scripts/start.ps1` vẫn còn nguyên.
