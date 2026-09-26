# TV5 — Apriori, web demo và tổng hợp báo cáo

**Thành viên:** ................................  
**Thời gian:** 4 tuần  
**Quy trình:** tham gia từ bước 1–2 để nhận dữ liệu; thực hiện 3. Apriori → 4. Gợi ý Top-N trên web → 5. Tổng hợp đánh giá.

Đánh dấu `[x]` sau khi **file đã có, nội dung kiểm tra được và các chủ mô hình xác nhận số liệu**. Mọi đường dẫn tính từ `project/`. TV5 ghép báo cáo, còn TV1–TV4 tự viết và duyệt phần chuyên môn của mình.

## Tuần 1 — Giao diện tích hợp và web mẫu

### Việc cần làm

- [ ] Thống nhất với TV1–TV4 hợp đồng hàm: nhận user hoặc hồ sơ phim đã chấm điểm cùng danh sách phim ứng viên, trả điểm và ID phim theo đúng thứ tự.
- [ ] Tạo web mẫu có tìm phim, chọn phim, nhập rating, chọn số lượng Top-N; chưa cần model thật.
- [ ] Xác định hai luồng: user/model đã có ID và người dùng mới cần cold-start. Ghi rõ giao diện sẽ hiển thị luồng nào.
- [ ] Kiểm tra cách nạp tên/thể loại phim từ `../Dataset/ml-25m/movies.csv` qua cấu hình; không sao chép dữ liệu vào mã nguồn.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `src/common/contracts.py` | Mô tả đầu vào, đầu ra và lỗi ID chưa thấy của hàm dự đoán |
| `src/app/main.py` | Web mẫu mở được, chọn/chấm điểm phim được trên dữ liệu nhỏ |
| `report/tv5_demo.md` | Sơ đồ luồng người dùng và quy tắc loại phim đã chọn |

- [ ] **Nghiệm thu tuần 1:** TV1–TV4 xác nhận giao diện suy luận; web mẫu chạy mà không cần model đã huấn luyện.

## Tuần 2 — Apriori và demo với baseline

### Việc cần làm

- [ ] Từ **train**, tạo một transaction cho mỗi user gồm các phim được chấm `>= 4`; ghi số transaction và quy tắc giới hạn phim nếu cần.
- [ ] Chạy Apriori với ngưỡng support/confidence đã ghi; tính lift; chọn 3–5 luật có thể giải thích, không xem luật là quan hệ nhân quả.
- [ ] Tích hợp baseline và hàm cold-start của TV1 vào web; tạo ứng viên, loại phim đã chọn, hiển thị Top-N với tên phim.
- [ ] Thử web với hồ sơ rỗng, 1 phim và nhiều phim; ghi ảnh hoặc mô tả lỗi cần sửa.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `src/association/apriori.py` | Tạo transaction và luật từ train, có cấu hình ngưỡng |
| `results/association_rules.csv`, `association_summary.csv` | Có antecedents/consequents, support, confidence, lift, số transaction |
| `src/app/recommender.py`, `main.py` | Demo baseline trả Top-N, không lặp phim đã chọn |
| `results/demo_smoke_checks.md`, `report/tv5_demo.md` | Ghi các tình huống thử và kết quả |

- [ ] **Nghiệm thu tuần 2:** người dùng chọn/chấm điểm phim và nhận Top-N bằng baseline; luật Apriori chỉ dựa vào train.

## Tuần 3 — So sánh validation và tích hợp model được chọn

### Việc cần làm

- [ ] Thu file metric/dự đoán validation của TV1–TV4; xác nhận cùng tập chia, số mẫu, đơn vị thời gian và phần cứng.
- [ ] Tạo bảng so sánh Linear/XGBoost/SVD/NCF với baseline; thêm Precision@10, Recall@10, NDCG@10 theo cùng quy tắc phim ứng viên và rating thích `>= 4`.
- [ ] Cùng nhóm chọn mô hình triển khai dựa trên validation, độ trễ và khả năng phục vụ user mới; ghi lý do chọn trước khi xem test.
- [ ] Nạp model được chọn vào web. Khi phải dùng fallback cho user mới, giao diện nêu rõ cách gợi ý đang dùng.
- [ ] Kiểm tra số phim Top-N, trùng lặp, phim đã chọn, lỗi ID chưa thấy và thời gian phản hồi.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `results/summary_validation.csv`, `topn_validation.csv` | Có nguồn từng metric, cùng giao thức đo, quy tắc ứng viên rõ ràng |
| `report/model_selection.md` | Lý do chọn model, tham số đã khóa, giới hạn user mới |
| `src/app/recommender.py`, `main.py` | Dùng được model được chọn và fallback phù hợp |
| `results/demo_smoke_checks.md` | Có kết quả thử ít nhất 5 tình huống đầu vào |

- [ ] **Nghiệm thu tuần 3:** cả nhóm ký nhận số liệu validation và mô hình triển khai; không đổi lựa chọn sau khi xem test.

## Tuần 4 — Đánh giá cuối, hoàn thiện web và báo cáo

### Việc cần làm

- [ ] Thu kết quả test một lần từ TV1–TV4; ghép bảng tổng hợp mà không đổi tham số/mô hình.
- [ ] Chạy đánh giá Top-N trên test theo quy tắc đã chốt; ghi rõ dữ liệu thiếu phản hồi với phim chưa xem.
- [ ] Chạy web từ môi trường sạch theo README; kiểm tra đầu vào rỗng, phim lạ, nhiều rating, fallback, Top-N và thời gian trả kết quả.
- [ ] Ghép nội dung TV1–TV4, chương mở đầu và phần demo; đối chiếu mọi bảng/hình với file kết quả nguồn. Nhờ từng người duyệt phần của mình.
- [ ] Chuẩn bị báo cáo Word, slide, ảnh demo và buổi chạy thử; cập nhật lệnh chạy web trong `README.md`.

### Sản phẩm phải có cuối tuần

| File cần tạo/cập nhật | Nội dung và điều kiện hoàn thành |
|---|---|
| `results/summary_test.csv`, `topn_test.csv` | Có metric cuối, số mẫu, quy tắc ứng viên và nguồn dữ liệu |
| `results/demo_smoke_checks.md`, `figures/demo_home.png`, `demo_recommendations.png` | Có bằng chứng web chạy và các tình huống lỗi được xử lý |
| `report/tv5_demo.md`, `report/bao_cao_cuoi.docx`, `report/slide_thuyet_trinh.pptx` | Có mô tả demo, phân công, kết quả và tài liệu trình bày đã duyệt |
| `README.md` | Có lệnh cài môi trường, chuẩn bị dữ liệu và khởi chạy web |

- [ ] **Nghiệm thu tuần 4/hoàn thành TV5:** web chạy theo README, báo đúng model/fallback, bảng tổng hợp khớp CSV của từng người và báo cáo được cả nhóm duyệt.

### Mở rộng nếu phần bắt buộc đã xong

- [ ] Nếu đủ thời gian, so sánh FP-Growth với Apriori trong `src/association/fp_growth.py` và `results/fp_growth_rules.csv`.
- [ ] Nếu nhóm làm Q-Learning, ghép `results/rl_baselines.csv` và `results/rl_training.csv`, tạo `figures/rl_cumulative_reward.png` và mô tả giả định mô phỏng trong `report/tv5_demo.md`.
