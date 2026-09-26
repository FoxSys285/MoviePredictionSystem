# TV1 — Dữ liệu, chia tập, baseline và cold-start

**Thành viên:** ................................  
**Thời gian:** 4 tuần  
**Quy trình:** 1. Thu thập dữ liệu → 2. Chia tập → 3. Huấn luyện baseline → 4. Dự đoán → 5. Đánh giá.

Đánh dấu `[x]` khi **đã có file, chạy/đọc kiểm tra được và người nhận đã xác nhận**. Mọi đường dẫn trong bảng tính từ thư mục `project/`. Dữ liệu MovieLens gốc đã ở `../Dataset/ml-25m`; không chép lại vào Git.

## Tuần 1 — Thu thập dữ liệu và chia tập

### Việc cần làm

- [ ] Đọc README của MovieLens 25M; kiểm tra đủ `ratings.csv`, `movies.csv`, định dạng cột, số dòng, rating và timestamp.
- [ ] Kiểm tra giá trị thiếu, bản ghi trùng, rating ngoài khoảng 0,5–5; ghi số dòng trước/sau mỗi quy tắc làm sạch.
- [ ] Thống kê số user, số phim, phân bố rating và số lượt tương tác; tạo biểu đồ EDA có tiêu đề, nhãn trục và đơn vị.
- [ ] Tạo `row_id`; sắp xếp theo `(timestamp, row_id)`; chia train/validation/test mục tiêu 70/15/15 theo thời gian.
- [ ] Từ **train**, thống nhất ngưỡng tương tác ít, tạo năm lần chia tiến theo thời gian trong train; ghi tỷ lệ warm-start và ID chưa thấy.
- [ ] Gửi TV2–TV5 schema, số dòng, mốc thời gian và cách nạp các tập chia. Cả nhóm xác nhận trước khi huấn luyện chính thức.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `src/data/eda.py` | Chạy lại được thống kê và biểu đồ từ dữ liệu gốc |
| `src/data/prepare.py`, `src/data/split.py` | Làm sạch, gán `row_id`, chia theo thời gian bằng seed/quy tắc cố định |
| `data/processed/train.parquet`, `validation.parquet`, `test.parquet` | Có cùng schema; `row_id` không trùng giữa các tập; thời gian không đảo ngược |
| `data/processed/splits.json`, `cv_folds.json`, `dataset_manifest.json` | Lưu ranh giới chia, năm fold, số dòng, ngưỡng lọc và tỷ lệ thực tế; không cần chứa hàng triệu ID trong JSON |
| `figures/rating_distribution.png`, `interaction_distribution.png` | Thể hiện phân bố rating và số tương tác |
| `results/data_profile.json` | Thống kê quy mô, giá trị thiếu/trùng và mức bao phủ |

- [ ] **Nghiệm thu tuần 1:** TV2–TV4 nạp được các tập chia; TV5 đọc được `movies.csv`; tổng số hàng sau xử lý khớp với số hàng ở train + validation + test.

## Tuần 2 — Baseline và quy tắc cho người dùng mới

### Việc cần làm

- [ ] Tính trung bình rating từ train và huấn luyện baseline có hiệu chỉnh user/item nếu phù hợp; có fallback cho ID chưa thấy.
- [ ] Dự đoán rating trên validation theo `row_id`; đo RMSE/MAE và thời gian train/suy luận.
- [ ] Đếm user/item mới ở validation, tách số liệu warm-start và cold-start; không xóa các trường hợp khó mà không báo cáo.
- [ ] Viết hàm nhận danh sách phim người dùng mới đã chọn/chấm điểm, loại các phim này khỏi danh sách gợi ý; bàn giao cho TV5.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `src/data/baseline.py`, `src/data/cold_start.py` | Dự đoán baseline và trả ứng viên cho user mới; xử lý hồ sơ rỗng/ít rating |
| `results/predictions_mean_validation.csv` | Đúng cột `row_id,model,split,userId,movieId,rating_true,rating_pred` |
| `results/metrics_mean_validation.csv`, `coverage_validation.csv` | Có RMSE/MAE, thời gian, số trường hợp warm/cold |
| `report/tv1_data.md` | Bản nháp mục nguồn dữ liệu, EDA, chia tập và baseline |

- [ ] **Nghiệm thu tuần 2:** RMSE/MAE tính lại từ file dự đoán khớp file metric; TV5 dùng được hàm cold-start trong demo baseline.

## Tuần 3 — Kiểm tra rò rỉ dữ liệu và năm lần kiểm chứng

### Việc cần làm

- [ ] Kiểm tra scaler, encoder, trung bình rating và ngưỡng lọc của từng thành viên chỉ học từ train tương ứng.
- [ ] Chạy baseline trên năm fold tiến theo thời gian; lưu metric từng fold và `mean ± std`.
- [ ] Soát file dự đoán validation của TV2–TV4: đúng `row_id`, không thiếu/trùng hàng so với tập đánh giá chung.
- [ ] Hỗ trợ TV5 kiểm tra danh sách ứng viên và trường hợp user mới; hoàn thiện phần viết dữ liệu trong báo cáo.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `src/data/audit.py` | Kiểm tra giao nhau giữa các tập, thứ tự thời gian, schema và ID |
| `results/data_audit.md` | Danh sách kiểm tra và kết quả, nêu rõ lỗi đã sửa nếu có |
| `results/metrics_mean_cv.csv` | Năm dòng/fold và thống kê tổng hợp rõ cách tính |
| `report/tv1_data.md` | Mô tả cuối cùng của quy trình dữ liệu để TV5 ghép báo cáo |

- [ ] **Nghiệm thu tuần 3:** nhóm xác nhận dataset và quy tắc đánh giá đã khóa; không sửa ranh giới chia sau khi xem kết quả test.

## Tuần 4 — Test cuối và kiểm tra demo

### Việc cần làm

- [ ] Sau khi nhóm khóa cấu hình, dự đoán baseline trên test **một lần** và tính RMSE/MAE.
- [ ] Báo cáo tỷ lệ user/item chưa thấy trong test và kết quả fallback; đối chiếu số dòng test với file dự đoán.
- [ ] Thử web với 0, 1 và nhiều phim đã chọn; kiểm tra phim đã chọn không xuất hiện lại trong Top-N.
- [ ] Xác nhận bảng số liệu của TV1 trong báo cáo và slide; cập nhật cách chạy bước dữ liệu trong `README.md`.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `results/predictions_mean_test.csv`, `metrics_mean_test.csv`, `coverage_test.csv` | Có đủ test, metric và tỷ lệ warm/cold; không dùng test để đổi tham số |
| `report/tv1_data.md`, `README.md` | Mục dữ liệu hoàn chỉnh, lệnh chạy lại và đường dẫn dữ liệu gốc |

- [ ] **Nghiệm thu tuần 4/hoàn thành TV1:** TV2–TV5 tái sử dụng được dữ liệu chia; mọi số liệu nguồn, số dòng và mốc thời gian trong báo cáo khớp với các file bàn giao.

### Mở rộng nếu phần bắt buộc đã xong

- [ ] Nếu nhóm làm Q-Learning: tạo trạng thái/lịch sử từ **train** trong `data/processed/rl_states.parquet` và mô tả quy tắc trong `report/tv1_data.md`.
