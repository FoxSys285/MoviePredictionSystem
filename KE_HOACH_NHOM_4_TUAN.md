# Kế hoạch đồ án Học máy: gợi ý phim với MovieLens 25M

**Thời gian:** 4 tuần, tính từ ngày nhóm bắt đầu.  
**Nhân sự:** 5 người; thay `TV1`–`TV5` bằng tên và thông tin liên hệ của từng thành viên.  
**Mục tiêu cuối:** báo cáo thực nghiệm có thể tái lập, bảng so sánh Linear/XGBoost/SVD/NCF và web demo gợi ý Top-N.

Nguồn dữ liệu: [MovieLens 25M – GroupLens](https://grouplens.org/datasets/movielens/25m/). Đọc [README của bộ dữ liệu](https://files.grouplens.org/datasets/movielens/ml-25m-README.html) trước khi tải, sử dụng hoặc chia sẻ dữ liệu.

## 1. Phạm vi và nguyên tắc chung

### Quy trình bắt buộc trong 4 tuần

1. **Thu thập dữ liệu:** lấy MovieLens 25M, kiểm tra các file, khám phá phân bố rating và làm sạch bản ghi không hợp lệ. TV1 bàn giao dữ liệu gốc đã xác minh cùng báo cáo EDA.
2. **Chia tập:** sắp xếp theo thời gian, tách **70% train, 15% validation, 15% test** và tạo năm lần kiểm chứng tiến theo thời gian trong train. TV1 bàn giao ID của từng tập và quy tắc lọc tương tác ít.
3. **Huấn luyện:** TV1 làm baseline; TV2 huấn luyện Linear, XGBoost, Logistic, Random Forest; TV3 huấn luyện SVD, KNN và các thí nghiệm giảm chiều/phân cụm; TV4 huấn luyện NCF; TV5 chạy Apriori. Mỗi người lưu model, cấu hình, seed và log.
4. **Dự đoán:** các mô hình tạo rating hoặc xác suất trên validation; sau khi khóa cấu hình, tạo dự đoán trên test. TV5 tích hợp mô hình được chọn vào web để xếp hạng và trả Top-N phim chưa xem.
5. **Đánh giá:** mỗi người tính chỉ số của mô hình mình; TV5 tổng hợp bảng so sánh độ chính xác, thời gian huấn luyện, độ trễ và kết quả web. Nhóm chọn mô hình triển khai từ validation và chỉ dùng test để báo cáo kết quả cuối.

**Luồng thực hiện:** tuần 1 hoàn thành bước 1–2; tuần 2–3 lặp bước 3–5 trên train/validation để cải thiện mô hình; tuần 4 khóa cấu hình rồi thực hiện bước 4–5 trên test một lần. Báo cáo, mã nguồn và slide là sản phẩm ghi lại kết quả của cả năm bước.

### Phần mở rộng nếu hoàn thành phần bắt buộc đúng hạn

- FP-Growth để đối chiếu với Apriori.
- Q-Learning trong môi trường mô phỏng, so với chiến lược ngẫu nhiên và phim phổ biến. **Không trình bày reward mô phỏng như hiệu quả thực tế trên người dùng.**

### Quy tắc đánh giá áp dụng cho cả nhóm

- Sắp xếp rating theo `timestamp`, chia một lần thành train/validation/test và lưu danh sách bản ghi của từng tập. Không xáo trộn ngẫu nhiên trước khi chia.
- Mọi thống kê dùng làm đặc trưng, chuẩn hóa và ngưỡng lọc người dùng/phim đều được học hoặc quyết định từ train. Không điền ô chưa có rating bằng 0.
- Trong train, tạo **5 lần kiểm chứng tiến theo thời gian** để báo cáo `mean ± std`. Các mô hình dự đoán rating dùng cùng năm lần chia. Validation dùng để chọn cấu hình, kiểm tra overfitting và quyết định mô hình triển khai; test chỉ được đánh giá một lần sau khi khóa lựa chọn.
- Cố định seed, ghi phiên bản thư viện/cấu hình máy, số dòng dữ liệu, tham số mô hình và thời gian chạy. Thời gian GPU/CPU chờ huấn luyện được ghi riêng, không tính là khối lượng lao động của một thành viên.
- Nếu toàn bộ 25M rating vượt quá tài nguyên, dùng **cùng một tập con cố định** cho các mô hình cần so sánh; ghi rõ cách lấy mẫu, quy mô và giới hạn của kết luận. Không dùng tập con khác nhau rồi so sánh trực tiếp.

## 2. Phân công tổng quan

Khối lượng dưới đây là **ước tính giờ làm việc chủ động**, để nhóm điều chỉnh sau tuần 1. Mỗi người còn tự viết phần phương pháp, kết quả và nhận xét của mình trong báo cáo.

| Thành viên | Đầu việc sở hữu | Hỗ trợ tích hợp | Ước tính |
|---|---|---|---:|
| **TV1** | Dữ liệu, EDA, chia tập, baseline, kiểm tra cold-start | Cung cấp dữ liệu chuẩn và xử lý hồ sơ người dùng mới cho web | 30–34 giờ |
| **TV2** | Linear, XGBoost, Logistic, Random Forest; tuning và đánh giá | Cung cấp hàm dự đoán thống nhất cho web | 30–34 giờ |
| **TV3** | SVD, KNN baseline, PCA/TruncatedSVD, K-Means, DBSCAN | Cung cấp mô hình gợi ý và vector đặc trưng | 30–34 giờ |
| **TV4** | NCF, Optuna, learning curve, đo hiệu năng | Đóng gói mô hình NCF và hàm suy luận | 30–34 giờ |
| **TV5** | Apriori, giao diện web, bảng so sánh và ghép báo cáo | Tích hợp các hàm dự đoán do TV1–TV4 bàn giao | 30–34 giờ |

**Trách nhiệm chung:** mỗi người kiểm tra chéo ít nhất một phần việc của người khác, viết phần mình phụ trách, đóng góp slide và trình bày được kết quả của mình. TV5 ghép tài liệu; TV1–TV4 chịu trách nhiệm xác nhận số liệu trong phần tương ứng.

## 3. Công việc chi tiết của từng thành viên

### TV1 — Dữ liệu, giao thức đánh giá và baseline

**Tuần 1**

- Tải và xác minh dữ liệu; đọc `ratings.csv`, `movies.csv`, kiểm tra kiểu dữ liệu, rating/timestamp thiếu hoặc trùng.
- Vẽ phân bố rating, số rating theo người dùng/phim, số phim theo thể loại và mức độ thưa của tương tác.
- Đề xuất ngưỡng lọc từ thống kê train, ví dụ số tương tác tối thiểu của người dùng/phim; nhóm chốt ngưỡng ở cuối tuần 1 và ghi tỷ lệ dữ liệu bị loại.
- Tạo một bản chia 70/15/15 theo thời gian và năm lần chia kiểm chứng trong train. Xuất ID hoặc chỉ mục bản ghi để mọi người dùng chính xác cùng dữ liệu.

**Tuần 2**

- Cài baseline điểm trung bình toàn cục và hiệu chỉnh theo người dùng/phim; xuất RMSE, MAE.
- Kiểm tra tỷ lệ người dùng/phim trong validation/test chưa có ở train và lập quy tắc đánh giá warm-start/cold-start.
- Chuẩn bị hàm chuyển các phim người dùng mới đã chấm điểm thành hồ sơ đầu vào cho web; nếu chưa đủ tương tác, trả gợi ý phổ biến hoặc theo phim tương tự.

**Tuần 3–4**

- Khóa dữ liệu và giao thức đánh giá; hỗ trợ các thành viên kiểm tra rò rỉ dữ liệu.
- Kiểm tra đầu vào web, tên phim và các phim đã xem phải bị loại khỏi Top-N.
- Viết mục dữ liệu, tiền xử lý, chia tập, baseline và giới hạn cold-start.

**Bàn giao:** tập chia/ID bản ghi, script tiền xử lý, biểu đồ EDA, baseline, mô tả quy tắc chia và thống kê cold-start.

### TV2 — Regression và classification

**Tuần 1**

- Thống nhất với TV1 các đặc trưng được phép dùng: ID mã hóa, thể loại/năm phim nếu cần, thống kê người dùng/phim chỉ tính trên train của từng lần chia.
- Tạo pipeline Linear Regression và XGBoost chạy được trên tập thử nhỏ; tránh tạo ma trận one-hot đặc quá lớn.

**Tuần 2**

- Huấn luyện hai mô hình regression. Tạo nhãn classification: `rating >= 4` là thích; các rating còn lại là không thích, và ghi rõ quy ước trong báo cáo.
- Huấn luyện Logistic Regression, Random Forest; lưu xác suất dự đoán để tính AUC-ROC.

**Tuần 3–4**

- Tìm tham số trên train bằng GridSearchCV/RandomizedSearchCV với cách chia tiến theo thời gian; chạy năm lần kiểm chứng cho mô hình dự đoán rating.
- Báo cáo RMSE/MAE, Accuracy/F1/AUC-ROC, thời gian huấn luyện và độ trễ dự đoán trên một số lượng mẫu cố định.
- Đóng gói hàm `predict(user, candidate_movies)` để TV5 tích hợp; viết mục mô hình và phân tích lỗi.

**Bàn giao:** mã huấn luyện/suy luận, cấu hình tham số, kết quả từng lần chạy và phần báo cáo regression/classification.

### TV3 — SVD, giảm chiều và phân cụm

**Tuần 1**

- Kiểm tra cách lưu ma trận user–item ở dạng thưa và cách chuẩn hóa rating trên train; thống nhất với TV1 danh sách user/item được giữ lại.
- Cài baseline KNN collaborative filtering hoặc item–item similarity để đối chiếu.

**Tuần 2**

- Huấn luyện SVD/matrix factorization để dự đoán rating; đánh giá RMSE/MAE và tạo hàm xếp hạng phim.
- Dùng PCA khi dữ liệu đã có vector đặc; với ma trận thưa lớn, dùng TruncatedSVD và ghi rõ lý do. Báo cáo reconstruction error, tỷ lệ phương sai giải thích khi phép đo phù hợp.

**Tuần 3–4**

- Dùng vector phim/người dùng học từ train để chạy K-Means và DBSCAN. Đo Silhouette, Davies–Bouldin trên cùng tập vector hoặc mẫu có quy tắc lấy mẫu cố định.
- Tìm tham số SVD trên train, chạy năm lần kiểm chứng, đo thời gian huấn luyện/suy luận; viết mục SVD, KNN, giảm chiều và phân cụm.
- Bàn giao hàm suy luận SVD và vector/item similarity cho TV5, TV1.

**Bàn giao:** mô hình SVD, KNN baseline, kết quả PCA/TruncatedSVD và clustering, biểu đồ cụm, file dự đoán và phần báo cáo.

### TV4 — Neural Collaborative Filtering (NCF)

**Tuần 1**

- Xây bảng ánh xạ `userId`, `movieId` thành chỉ số embedding từ train; quy định cách xử lý ID chưa thấy.
- Cài NCF gồm embedding user/item và MLP, đặt seed, tạo vòng lặp train/validation và lưu checkpoint.

**Tuần 2**

- Chạy thử trên tập nhỏ, kiểm tra loss giảm, đầu ra nằm trong thang rating hợp lệ và không dùng thông tin từ validation/test khi huấn luyện.
- Chạy bản đầy đủ theo tài nguyên khả dụng; lưu loss train/validation qua từng epoch và dùng early stopping.

**Tuần 3–4**

- Dùng Optuna thử số chiều embedding, số tầng/kích thước MLP, learning rate và regularization trong ngân sách thời gian đã thống nhất.
- Chạy năm lần kiểm chứng tiến theo thời gian theo cùng giao thức; báo cáo RMSE/MAE `mean ± std`, test một lần ở tuần 4, thời gian train và độ trễ suy luận.
- Vẽ learning curve, nhận xét overfitting, đóng gói model và hàm dự đoán cho TV5; viết mục NCF.

**Bàn giao:** mã NCF, checkpoint, cấu hình tốt nhất, log Optuna, learning curve, kết quả từng lần chạy và phần báo cáo.

### TV5 — Association, web demo và ghép kết quả

**Tuần 1**

- Tạo khung Streamlit hoặc FastAPI; giao diện cho chọn phim và chấm điểm các phim đã xem.
- Thống nhất với TV2–TV4 giao diện nhận dự đoán và định dạng file kết quả; chuẩn bị bảng tổng hợp tự đọc từ log.

**Tuần 2**

- Tạo transaction từ các phim được đánh giá `>= 4` trong train; chạy Apriori, chọn ngưỡng support/confidence hợp lý để số luật có thể phân tích.
- Báo cáo các luật tiêu biểu cùng support, confidence, lift; lưu ý luật phổ biến chưa chứng minh quan hệ nhân quả.
- Hoàn thiện web với baseline của TV1; tìm kiếm phim, nhập rating, loại phim đã chọn khỏi danh sách gợi ý.

**Tuần 3–4**

- Tích hợp mô hình được nhóm chọn bằng **kết quả validation**, bổ sung phương án cho người dùng mới và đo thời gian trả kết quả.
- Tạo bảng so sánh chung từ log; kiểm tra mọi kết quả có cùng dữ liệu, phần cứng và quy tắc đo trước khi đặt cạnh nhau.
- Ghép các phần báo cáo, thống nhất hình/bảng/tài liệu tham khảo, chạy thử demo và chuẩn bị bản nộp.

**Bàn giao:** kết quả Apriori, mã web chạy được, bảng so sánh, hướng dẫn chạy demo và bản báo cáo đã ghép.

## 4. Lịch 4 tuần và mốc nghiệm thu

Các bước trong bảng là năm bước đã nêu ở mục 1. Thí nghiệm trên validation có thể quay lại huấn luyện khi kết quả chưa đạt; tập test chỉ tham gia sau khi nhóm đã chốt cấu hình.

| Tuần | TV1 | TV2 | TV3 | TV4 | TV5 | Mốc cuối tuần |
|---|---|---|---|---|---|---|
| **1 — Bước 1–2** | Tải, EDA, chia tập | Pipeline Linear/XGBoost | Ma trận thưa, KNN | Ánh xạ ID, khung NCF | Khung web, định dạng log | Dữ liệu/giao thức chung được khóa; mọi pipeline chạy thử trên tập nhỏ |
| **2 — Bước 3–5 trên validation** | Baseline, cold-start | Regression + classification | SVD + giảm chiều | NCF bản đầu, learning curve | Apriori + web dùng baseline | Có kết quả validation sơ bộ và file dự đoán từ bốn mô hình chính |
| **3 — Bước 3–5 trên validation** | Kiểm tra rò rỉ, hỗ trợ tích hợp | Tuning + 5 lần chia | Tuning + 5 lần chia, clustering | Optuna + 5 lần chia | Tích hợp web, bảng so sánh validation | Chốt tham số và **chọn mô hình triển khai**, không dùng test để quyết định |
| **4 — Bước 4–5 trên test** | Kiểm thử dữ liệu/web, viết báo cáo | Test + viết báo cáo | Test + viết báo cáo | Test + viết báo cáo | Test chung, hoàn thiện web/báo cáo/slide | Chạy test một lần, demo hoạt động, nộp đầy đủ tài liệu |

**Họp ngắn hai lần mỗi tuần (15–20 phút):** đầu tuần chốt việc, giữa tuần xử lý vướng mắc. Cuối mỗi tuần, từng người cập nhật trạng thái `Hoàn thành / Đang làm / Bị chặn`, đường dẫn sản phẩm và số liệu đã có. Nếu một đầu việc trễ quá hai ngày, nhóm điều chỉnh phạm vi phần mở rộng trước khi ảnh hưởng đến mốc bắt buộc.

## 5. Quy ước bàn giao để các phần ghép được với nhau

| Sản phẩm | Người tạo | Quy ước tối thiểu |
|---|---|---|
| Dữ liệu chia tập | TV1 | Có danh sách ID/chỉ mục train, validation, test; ghi seed, mốc thời gian, số dòng và ngưỡng lọc |
| File dự đoán rating | TV1–TV4 | Các cột: `model`, `split`, `userId`, `movieId`, `rating_true`, `rating_pred` |
| Log thực nghiệm | Mỗi người | `model`, `task`, `fold`, `seed`, `metric`, `value`, `train_time_s`, `inference_ms`, `hardware`, `config` |
| Model dùng cho web | Chủ mô hình được chọn | Có file model, file ánh xạ ID, hướng dẫn nạp và hàm dự đoán cho danh sách phim ứng viên |
| Phần báo cáo cá nhân | Mỗi người | Mục tiêu, dữ liệu đầu vào, thuật toán, tham số, kết quả, nhận xét và giới hạn |

Không đưa toàn bộ dữ liệu MovieLens hoặc checkpoint lớn vào bản nộp mã nguồn nếu không được yêu cầu; cung cấp hướng dẫn tải và chạy lại.

## 6. Khung bảng kết quả trong báo cáo

### So sánh mô hình dự đoán rating

| Mô hình | Người phụ trách | RMSE CV (mean ± std) | MAE CV (mean ± std) | RMSE test | MAE test | Train time | Inference time | Nhận xét |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Mean rating | TV1 | … | … | … | … | … | … | … |
| Linear Regression | TV2 | … | … | … | … | … | … | … |
| XGBoost | TV2 | … | … | … | … | … | … | … |
| SVD | TV3 | … | … | … | … | … | … | … |
| NCF | TV4 | … | … | … | … | … | … | … |

`CV` là năm lần kiểm chứng theo thời gian **trong train**; kết quả test là một lần đánh giá cuối. Ghi rõ đơn vị thời gian và phần cứng. Ngoài RMSE/MAE, đo Precision@10, Recall@10 và NDCG@10 cho gợi ý Top-N trên các tương tác tương lai có rating `>= 4`; nêu rõ cách chọn phim ứng viên vì dữ liệu không chứa phản hồi cho mọi phim chưa được xem.

### Các thí nghiệm bổ sung

| Nhóm | Người phụ trách | Chỉ số phải báo cáo | Hình/bảng cần có |
|---|---|---|---|
| Classification | TV2 | Accuracy, F1, AUC-ROC | Bảng so Logistic/Random Forest, ma trận nhầm lẫn |
| SVD/PCA | TV3 | Reconstruction error, phương sai giải thích khi phù hợp | Biểu đồ theo số chiều |
| Clustering | TV3 | Silhouette, Davies–Bouldin | Bảng theo số cụm/tham số, mô tả cụm |
| Association | TV5 | Support, Confidence, Lift | Một số luật có ý nghĩa, giải thích ví dụ |
| NCF | TV4 | Loss train/validation, RMSE/MAE | Learning curve |
| RL mở rộng | Cả nhóm | Cumulative reward so với hai baseline | Biểu đồ reward, mô tả môi trường mô phỏng |

## 7. Tiêu chí hoàn thành

- Người khác trong nhóm có thể chạy lại phần tiền xử lý và nhận đúng số dòng của từng tập.
- Bốn mô hình chính có dự đoán trên cùng tập test, log cấu hình và bảng so sánh đầy đủ.
- Nhóm giải thích được vì sao chọn mô hình đưa lên web dựa trên độ chính xác **và** chi phí tính toán.
- Web cho chọn/chấm điểm phim, trả Top-N phim chưa chọn và xử lý được người dùng mới.
- Báo cáo ghi rõ phần đã làm, phần mở rộng chưa làm nếu có, giới hạn dữ liệu và phân công thực tế của từng thành viên.
