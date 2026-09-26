# Hướng dẫn thực hiện công việc cho nhóm 5 thành viên

Tài liệu này là hướng dẫn thao tác đi kèm [kế hoạch 4 tuần](./KE_HOACH_NHOM_4_TUAN.md). Thay `TV1`–`TV5` bằng tên thật của nhóm. [Khung thư mục `project`](../README.md) đã được tạo; từng thành viên bổ sung mã và kết quả vào phần mình phụ trách.

## 1. Cách làm việc chung trước khi chia việc

### Quy trình 5 bước của toàn nhóm

Mỗi người có thể làm song song phần chuẩn bị của mình, nhưng sản phẩm chính phải đi qua đúng thứ tự sau:

| Bước | Công việc và người phụ trách | Sản phẩm để chuyển sang bước tiếp theo |
|---|---|---|
| **1. Thu thập dữ liệu** | TV1 lấy và xác minh MovieLens 25M, đọc `ratings.csv`/`movies.csv`, kiểm tra dữ liệu, làm EDA; các thành viên còn lại thống nhất dữ liệu cần dùng | Dữ liệu nguồn đã xác minh, thống kê và biểu đồ EDA, quy tắc làm sạch |
| **2. Chia tập** | TV1 sắp xếp theo `timestamp`, chia 70/15/15, tạo năm lần chia tiến theo thời gian trong train; cả nhóm kiểm tra quy tắc lọc và rò rỉ dữ liệu | ID train/validation/test, các lần chia kiểm chứng, số dòng và mốc thời gian |
| **3. Huấn luyện** | TV1 làm baseline; TV2 làm regression/classification; TV3 làm SVD/KNN/PCA/clustering; TV4 làm NCF; TV5 làm Apriori | Model hoặc kết quả thí nghiệm, cấu hình tham số, seed và log huấn luyện |
| **4. Dự đoán** | TV1–TV4 xuất dự đoán cho validation, sau đó cho test khi cấu hình đã khóa; TV5 dùng model được chọn để trả Top-N trên web | CSV dự đoán có `row_id`, điểm dự đoán/xác suất, danh sách phim gợi ý không lặp phim đã xem |
| **5. Đánh giá** | Mỗi người tính chỉ số của phần mình; TV5 ghép bảng, cả nhóm nhận xét và chọn mô hình từ validation | Bảng metric, thời gian train/suy luận, biểu đồ, kết luận và demo |

**Trong tuần 2–3, nhóm lặp bước 3→4→5 trên train/validation** để cải thiện mô hình. Cuối tuần 3 khóa lựa chọn; tuần 4 chỉ chạy dự đoán và đánh giá trên test một lần. Không tính kết quả test rồi quay lại sửa tham số hoặc đổi mô hình.

### 1.1. Sản phẩm cần nộp sau 4 tuần

- Mã nguồn và hướng dẫn chạy lại từ dữ liệu gốc.
- Báo cáo có EDA, cách chia dữ liệu, mô hình, tham số, kết quả, so sánh và giới hạn.
- Bảng so sánh chung của Mean Rating, Linear, XGBoost, SVD và NCF; kết quả classification, clustering, association, giảm chiều ở các mục riêng.
- Web demo: chọn và chấm điểm phim đã xem, nhận danh sách Top-N phim chưa chọn.
- Slide và phần trình bày của cả 5 người.

### 1.2. Thư mục và tên file thống nhất

```text
project/
├── README.md                  # TV1 hướng dẫn chuẩn bị dữ liệu; TV5 bổ sung cách chạy web
├── requirements.txt           # Các thư viện và phiên bản thực tế đã dùng
├── data/
│   ├── raw/                   # Hướng dẫn dữ liệu; MovieLens gốc ở ../Dataset/ml-25m
│   └── processed/             # Tập chia và file ánh xạ ID
├── src/
│   ├── data/                  # TV1: tiền xử lý, chia tập, baseline
│   ├── classical/             # TV2: regression, classification
│   ├── factorization/         # TV3: SVD, KNN, PCA, clustering
│   ├── ncf/                   # TV4: NCF
│   ├── association/           # TV5: Apriori
│   └── app/                   # TV5: web demo
├── artifacts/                  # Model và ánh xạ ID cần cho demo
├── results/                    # CSV metric, dự đoán, bảng tổng hợp
├── figures/                    # Biểu đồ đưa vào báo cáo
└── report/                     # Phần viết của từng người và bản ghép
```

Nhóm có thể đổi tên thư mục, nhưng phải thống nhất **trước khi bắt đầu tuần 2**. Mỗi thành viên tự lưu mã, kết quả và mô tả cách chạy phần mình phụ trách.

### 1.3. Quy ước dữ liệu và đánh giá

1. `ratings.csv` có `userId`, `movieId`, `rating`, `timestamp`; `movies.csv` có thông tin tên/thể loại phim. Dùng ID bản ghi duy nhất (`row_id`) sau khi đọc dữ liệu để ghép dự đoán đúng hàng.
2. Sắp xếp theo thời gian, chia **70% train, 15% validation, 15% test**. Nếu nhiều bản ghi có cùng thời điểm ở ranh giới, chuyển cả nhóm bản ghi đó về một phía và ghi tỷ lệ thực tế.
3. Mọi scaler, encoder, thống kê người dùng/phim và ngưỡng lọc chỉ được học từ phần train tương ứng. Ô user–item chưa có rating là **chưa quan sát**, không phải rating `0`.
4. Tạo 5 lần kiểm chứng tiến theo thời gian **bên trong train**. Mỗi mô hình rating dùng cùng năm lần chia; có thể bắt đầu bằng `TimeSeriesSplit(n_splits=5)` trên thứ tự đã sắp xếp.
5. Tìm tham số trên train; dùng validation để kiểm tra cấu hình cuối và chọn mô hình triển khai. Chỉ tính metric test sau khi đã khóa tham số, quy tắc tiền xử lý và lựa chọn mô hình.
6. Nếu phải lấy mẫu vì giới hạn máy, TV1 tạo một tập con cố định và mọi mô hình trong bảng so sánh dùng đúng tập đó. Báo cáo nêu số rating ban đầu, số rating dùng và cách lấy mẫu.

**File dự đoán chuẩn:** `row_id,model,split,userId,movieId,rating_true,rating_pred`.  
**File log chuẩn:** `model,task,fold,seed,metric,value,train_time_s,inference_ms,hardware,config`.  
**Quy ước thời gian suy luận:** đo trên cùng số lượng cặp user–movie hoặc cùng số yêu cầu Top-N; ghi rõ số lượng và đơn vị `ms`.

### 1.4. Cách báo cáo tiến độ

Cuối mỗi tuần, mỗi người gửi cho nhóm bốn dòng sau và đường dẫn sản phẩm:

```text
Đã hoàn thành:
Kết quả/số liệu hiện có:
Vướng mắc cần hỗ trợ:
Việc sẽ làm tuần tới:
```

Không ghi "đã huấn luyện xong" nếu chưa có file model, file dự đoán, metric và cấu hình đủ để người khác kiểm tra.

---

## 2. TV1 — Dữ liệu, chia tập, baseline và cold-start

### Mục tiêu

TV1 tạo đầu vào tin cậy cho toàn nhóm. Mọi thành viên phải đọc cùng một bộ ID train/validation/test; TV1 chịu trách nhiệm phát hiện rò rỉ dữ liệu ở bước này.

**Bước phụ trách:** 1. Thu thập dữ liệu; 2. Chia tập; 3. Huấn luyện baseline; hỗ trợ bước 4–5.

### Hướng dẫn làm từng bước

1. **Tải và kiểm tra dữ liệu gốc (tuần 1).** Lấy MovieLens 25M từ [GroupLens](https://grouplens.org/datasets/movielens/25m/), đọc README và ghi nguồn/phiên bản. Kiểm tra các cột bắt buộc, kiểu dữ liệu, rating ngoài khoảng hợp lệ, timestamp thiếu và bản ghi trùng. Lưu số dòng gốc và số dòng sau mỗi bước xử lý.
2. **Khám phá dữ liệu (tuần 1).** Tính số người dùng, số phim, số rating; vẽ histogram rating, phân bố số tương tác/người dùng và số tương tác/phim. Tính độ thưa bằng công thức `1 - số cặp đã quan sát / (số user × số item)`. Viết 3–5 nhận xét ngắn có số liệu hỗ trợ.
3. **Chia theo thời gian (tuần 1).** Gán `row_id`, sắp xếp tăng theo `(timestamp, row_id)`, chia 70/15/15, lưu danh sách `row_id` của từng tập và mốc thời gian. Kiểm tra các tập không giao nhau và mỗi hàng gốc xuất hiện đúng một lần. Tạo thêm 5 lần chia tiến theo thời gian bên trong train.
4. **Xử lý người dùng/phim ít tương tác (tuần 1).** Tính số lượt tương tác chỉ trên train. Đề xuất ngưỡng cùng tỷ lệ dữ liệu bị loại, rồi chốt với nhóm. Giữ thông tin các hàng bị loại để báo cáo mức bao phủ; tạo tập đánh giá warm-start chung cho các mô hình và đếm riêng trường hợp không có user/item trong train. Không lặng lẽ bỏ các hàng test khó.
5. **Tạo dữ liệu dùng chung (tuần 1–2).** Xuất các file train/validation/test ở dạng tiết kiệm dung lượng như Parquet, ánh xạ ID và bảng `movies` đã chuẩn hóa. Viết script chạy lại các bước trên từ dữ liệu gốc. Gửi schema, số dòng và mốc thời gian cho TV2–TV5.
6. **Làm baseline (tuần 2).** Dự đoán bằng rating trung bình từ train. Có thể thêm hiệu chỉnh trung bình theo user/item với quy tắc fallback về trung bình chung. Tính RMSE/MAE trên validation và lưu dự đoán theo định dạng chuẩn.
7. **Chuẩn bị cho web (tuần 2–4).** Viết quy tắc cho người dùng mới: lấy các phim vừa chấm điểm, tạo danh sách ứng viên theo phim tương tự hoặc phim phổ biến, loại phim đã chọn. Thử với hồ sơ rỗng, 1 phim và nhiều phim; ghi rõ cách xử lý từng trường hợp.

### Bàn giao và cách tự kiểm tra

| Bàn giao | Điều kiện đạt |
|---|---|
| Script tiền xử lý và tập chia | Chạy lại cho cùng `row_id` và số dòng; không có giao giữa ba tập |
| Báo cáo EDA | Có biểu đồ, số liệu nguồn và mô tả ngưỡng lọc |
| 5 lần chia kiểm chứng | Trong mỗi lần, thời điểm train không muộn hơn thời điểm validation của lần đó |
| Baseline | Có dự đoán, RMSE/MAE và quy tắc fallback cho ID chưa thấy |
| Hỗ trợ web | Danh sách Top-N không chứa phim người dùng đã chọn |

**Lỗi cần tránh:** lọc trên toàn bộ dữ liệu trước khi chia; dùng trung bình rating từ test; tạo ma trận user–item đặc rất lớn; xóa test cold-start mà không báo cáo.

---

## 3. TV2 — Linear, XGBoost và classification

### Mục tiêu

TV2 cung cấp hai mô hình rating dựa trên đặc trưng và hai mô hình phân loại thích/không thích, cùng file kết quả có thể so sánh với SVD/NCF.

**Bước phụ trách:** 3. Huấn luyện; 4. Dự đoán; 5. Đánh giá cho regression/classification.

### Hướng dẫn làm từng bước

1. **Đọc giao thức của TV1 (tuần 1).** Nạp đúng ID chia tập, xác nhận số dòng. Thiết kế đặc trưng có thể tính tại thời điểm dự đoán: ID người dùng/phim mã hóa thưa, thể loại/năm phim, số lần tương tác, rating trung bình lịch sử. Với đặc trưng trung bình theo user/item, khi tạo hàng train phải loại rating của chính hàng đó hoặc chỉ dùng lịch sử trước thời điểm hàng đó; validation/test chỉ dùng thống kê từ train.
2. **Huấn luyện Linear Regression (tuần 1–2).** Bắt đầu bằng ít đặc trưng và tập nhỏ; dùng biểu diễn thưa nếu mã hóa ID. Ghi rõ biến đầu vào, cách chuẩn hóa và cách xử lý ID chưa thấy. So sánh với baseline trước khi tăng độ phức tạp.
3. **Huấn luyện XGBoost (tuần 2).** Dùng cùng nguồn dữ liệu và nhãn rating, có thể dùng bộ đặc trưng phù hợp cho cây. Đặt seed, giới hạn số cây/độ sâu để thử nhanh, rồi tìm tham số trên train. Không lấy mốc test để quyết định số vòng lặp.
4. **Tạo bài toán classification (tuần 2).** Định nghĩa `rating >= 4` là lớp thích, còn lại là không thích; ghi tỷ lệ hai lớp. Train Logistic Regression và Random Forest, xuất cả nhãn và xác suất lớp thích. Nếu mất cân bằng lớp, nêu cách xử lý và đo F1/AUC-ROC ngoài Accuracy.
5. **Tuning và kiểm chứng (tuần 3).** Dùng GridSearchCV/RandomizedSearchCV với cách chia tiến theo thời gian trong train. Sau khi chốt cấu hình, chạy năm lần chia chung của nhóm; lưu RMSE/MAE từng lần và `mean ± std`. Tính Accuracy/F1/AUC-ROC cho hai mô hình classification.
6. **Đóng gói và báo cáo (tuần 3–4).** Lưu model và các bộ biến đổi đặc trưng cần thiết. Cung cấp hàm hoặc mô-đun nhận `userId` và danh sách `movieId` ứng viên, trả điểm dự đoán theo đúng thứ tự. Đo thời gian train và suy luận trên cùng quy mô mẫu mà cả nhóm thống nhất. Viết nhận xét: khi nào mô hình vượt baseline, ở nhóm người dùng/phim nào sai nhiều.

### Bàn giao và cách tự kiểm tra

| Bàn giao | Điều kiện đạt |
|---|---|
| Linear và XGBoost | Có model, tham số, dự đoán validation/test, RMSE/MAE và thời gian |
| Logistic và Random Forest | Có Accuracy, F1, AUC-ROC; xác suất trong `[0, 1]` |
| Hàm suy luận | Số điểm trả về bằng số phim ứng viên; không đảo thứ tự phim |
| Mục báo cáo | Có định nghĩa nhãn, đặc trưng, bảng tham số và phân tích lỗi |

**Lỗi cần tránh:** dùng rating tương lai để tính đặc trưng lịch sử; tạo mean user/item của hàng train bao gồm chính nhãn cần dự đoán; báo cáo Accuracy mà không xem F1 khi lớp mất cân bằng.

---

## 4. TV3 — SVD, KNN, giảm chiều và phân cụm

### Mục tiêu

TV3 xây hướng gợi ý cộng tác, kiểm tra việc giảm chiều và phân nhóm người dùng/phim từ dữ liệu train.

**Bước phụ trách:** 3. Huấn luyện; 4. Dự đoán; 5. Đánh giá cho SVD/KNN/giảm chiều/phân cụm.

### Hướng dẫn làm từng bước

1. **Chuẩn bị ma trận thưa (tuần 1).** Ánh xạ user/item của train sang chỉ số liên tục. Chỉ lưu rating đã quan sát. Ghi số user, item, tương tác và kích thước bộ nhớ; không gọi `.toarray()` trên toàn bộ ma trận 25M.
2. **Baseline KNN (tuần 1–2).** Chọn item–item hoặc user–user similarity; giới hạn số láng giềng và chỉ tính trên các rating quan sát. Dự đoán rating hoặc xếp hạng phim; dùng fallback của TV1 khi không có láng giềng/ID.
3. **SVD cho rating (tuần 2).** Cài matrix factorization trên các rating quan sát bằng thư viện phù hợp hoặc tự cài. Tìm số chiều latent, regularization và số epoch trên train/validation. Xuất rating dự đoán và RMSE/MAE. **Phân biệt:** matrix factorization để dự đoán rating không đồng nghĩa với TruncatedSVD trên ma trận điền 0.
4. **Giảm chiều (tuần 2–3).** Với dữ liệu vector đặc, chạy PCA; với ma trận thưa lớn, ưu tiên TruncatedSVD. Thử vài số chiều, đo reconstruction error và phương sai giải thích khi phương pháp hỗ trợ. Không dùng test để chọn số chiều.
5. **Clustering (tuần 3).** Lấy embedding hoặc vector đặc trưng học từ train, chuẩn hóa nếu cần, rồi chạy K-Means và DBSCAN. Đo Silhouette và Davies–Bouldin trên một mẫu có quy tắc lấy mẫu cố định nếu dữ liệu quá lớn. Với DBSCAN, báo cáo cả tỷ lệ điểm nhiễu và số cụm tìm được.
6. **Kiểm chứng và bàn giao (tuần 3–4).** Chạy SVD trên năm lần chia chung, đo RMSE/MAE và thời gian. Lưu model, ánh xạ ID, hàm suy luận và file dự đoán. Viết kết luận về chất lượng so với KNN/mean rating, chi phí bộ nhớ và giới hạn với ID mới.

### Bàn giao và cách tự kiểm tra

| Bàn giao | Điều kiện đạt |
|---|---|
| SVD và KNN | Có dự đoán trên cùng tập đánh giá, RMSE/MAE và fallback |
| PCA/TruncatedSVD | Có biểu đồ theo số chiều và giải thích chỉ số được dùng |
| K-Means/DBSCAN | Có số cụm, chỉ số chất lượng và mô tả ý nghĩa một vài cụm |
| Model cho web | Có ánh xạ ID và hàm dự đoán; không gợi ý lại phim đã xem |

**Lỗi cần tránh:** coi rating thiếu là 0; đánh giá clustering trên tập lớn khiến máy hết bộ nhớ; dùng `TruncatedSVD` điền 0 như kết quả SVD dự đoán rating mà không nói rõ khác biệt.

---

## 5. TV4 — Neural Collaborative Filtering (NCF)

### Mục tiêu

TV4 huấn luyện NCF gồm embedding user/item và MLP để dự đoán rating; kết quả phải so sánh công bằng với Linear, XGBoost và SVD.

**Bước phụ trách:** 3. Huấn luyện; 4. Dự đoán; 5. Đánh giá cho NCF.

### Hướng dẫn làm từng bước

1. **Ánh xạ ID và tạo DataLoader (tuần 1).** Học `userId -> index` và `movieId -> index` từ train. Lưu ánh xạ cùng checkpoint. Tạo batch `(user_index, movie_index, rating)`; xác định trước cách xử lý ID chưa thấy để không truy cập embedding ngoài phạm vi.
2. **Cài mô hình tối thiểu (tuần 1–2).** Tạo embedding user/item, ghép vector và đưa qua MLP đến một điểm rating. Dùng MSE hoặc MAE làm loss nhất quán với mục tiêu. Nếu chuẩn hóa rating, lưu cách biến đổi và hoàn nguyên; kiểm tra điểm cuối cùng trong khoảng rating hợp lệ khi đánh giá.
3. **Chạy thử nhỏ (tuần 2).** Kiểm tra một batch truyền qua mô hình, loss không phải `NaN`, loss train giảm và checkpoint nạp lại cho cùng kết quả dự đoán. Chạy seed cố định; ghi phiên bản PyTorch/TensorFlow và thiết bị CPU/GPU thực tế.
4. **Huấn luyện và theo dõi overfitting (tuần 2–3).** Ghi train loss và validation loss mỗi epoch. Dùng early stopping hoặc lưu epoch tốt nhất theo validation. Vẽ learning curve; giải thích nếu train loss tiếp tục giảm nhưng validation loss tăng.
5. **Tối ưu bằng Optuna (tuần 3).** Thử số chiều embedding, kích thước/số tầng MLP, learning rate, weight decay và batch size. Chốt số trial/ngân sách máy trước khi chạy; có thể tìm cấu hình trên tập con cố định nhưng phải đánh giá cuối trên tập chung. Lưu mọi trial, kể cả trial kém.
6. **Kiểm chứng, test và bàn giao (tuần 3–4).** Huấn luyện lại từ đầu trên cùng năm lần chia tiến theo thời gian để lấy RMSE/MAE `mean ± std`. Sau khi nhóm khóa lựa chọn, chạy test một lần. Đo train time, inference time; lưu checkpoint, ánh xạ ID, cấu hình và hàm dự đoán cho TV5.

### Bàn giao và cách tự kiểm tra

| Bàn giao | Điều kiện đạt |
|---|---|
| Mã và checkpoint NCF | Nạp checkpoint rồi dự đoán lại được trên cùng đầu vào |
| Log huấn luyện | Có loss train/validation theo epoch, cấu hình và seed |
| Optuna | Có bảng trial và lý do chọn tham số cuối |
| Đánh giá | Có RMSE/MAE của 5 lần chia, test, thời gian và biểu đồ learning curve |

**Lỗi cần tránh:** đưa `userId`/`movieId` gốc trực tiếp vào embedding; tạo ánh xạ từ toàn bộ dữ liệu rồi vô tình dùng thông tin tương lai; dùng các tương tác chưa quan sát làm nhãn rating âm; so sánh thời gian CPU với GPU mà không ghi thiết bị.

---

## 6. TV5 — Apriori, web demo và tổng hợp báo cáo

### Mục tiêu

TV5 làm thí nghiệm luật kết hợp, ghép mô hình đã chọn thành demo có thể sử dụng và tổng hợp báo cáo mà không tự gánh phần viết của bốn người còn lại.

**Bước phụ trách:** 3. Huấn luyện/thực nghiệm Apriori; 4. Tích hợp dự đoán và Top-N; 5. Đánh giá, tổng hợp và trình bày.

### Hướng dẫn làm từng bước

1. **Thống nhất giao diện tích hợp (tuần 1).** Gửi cho TV1–TV4 quy ước đầu vào `userId` hoặc hồ sơ phim đã chấm điểm, danh sách `movieId` ứng viên; đầu ra là danh sách `(movieId, score)`. Thống nhất cách lưu model và lỗi ID chưa thấy. Dựng trang với ô tìm phim, danh sách phim đã chọn, mức rating và số lượng gợi ý N.
2. **Tạo transaction cho Apriori (tuần 2).** Từ **train**, mỗi user là một transaction gồm các phim được chấm `>= 4`. Cân nhắc giới hạn vào các phim có đủ lượt thích để giảm số cột và thời gian chạy; ghi rõ ngưỡng và số transaction còn lại. Chạy Apriori, tính support, confidence, lift; lọc luật quá hiếm hoặc quá hiển nhiên và giải thích 3–5 ví dụ. Các phim được đánh giá cách nhau nhiều năm không nhất thiết thuộc một phiên xem phim, nên chỉ diễn giải đây là đồng sở thích trong lịch sử.
3. **Hoàn thiện luồng web với baseline (tuần 2).** Cho phép người dùng chọn/chấm điểm ít nhất vài phim; hiển thị tên phim và thể loại. Tạo danh sách ứng viên, loại phim đã chọn, xếp hạng theo baseline/cold-start của TV1. Kiểm tra trường hợp không chọn phim và phim không có trong tập huấn luyện.
4. **Tích hợp mô hình được chọn (tuần 3).** Dựa trên bảng validation do cả nhóm xác nhận, chọn mô hình cân bằng RMSE/MAE, độ trễ và khả năng phục vụ người dùng mới. Với user/item có trong train, dùng model đã chọn. Với người dùng mới mà model cần embedding ID, dùng nhánh cold-start của TV1 hoặc suy ra vector mới nếu mô hình hỗ trợ; **giao diện phải ghi rõ đang dùng cách nào**, không gọi đó là dự đoán NCF/SVD nếu thật ra dùng fallback.
5. **Kiểm thử demo (tuần 3–4).** Chạy các trường hợp: không nhập phim; nhập 1 phim; nhập nhiều phim với điểm khác nhau; phim không có model; phim đã chọn không xuất hiện lại; Top-N có đúng N phim nếu đủ ứng viên; thời gian phản hồi được ghi lại. Ghi cách cài thư viện và lệnh chạy web trong README.
6. **Ghép và kiểm tra báo cáo (tuần 4).** Thu phần viết, bảng và hình từ TV1–TV4. Kiểm tra mọi con số trong bảng tổng hợp khớp với CSV gốc, cùng tập đánh giá và đơn vị thời gian. Thêm mục phân công thực tế, giới hạn và kết luận chọn model. Mỗi người duyệt lại phần của mình trước khi nộp.

### Bàn giao và cách tự kiểm tra

| Bàn giao | Điều kiện đạt |
|---|---|
| Apriori | Có transaction, ngưỡng, số luật và bảng Support/Confidence/Lift |
| Web demo | Chạy theo README; chọn/chấm điểm phim và nhận Top-N không lặp phim đã chọn |
| Bảng so sánh | Mỗi số liệu có file kết quả gốc, mô tả tập đo và đơn vị |
| Báo cáo ghép | Có đủ phần của 5 người, hình/bảng được đánh số và trích dẫn dữ liệu |

**Lỗi cần tránh:** tạo luật Apriori từ test; gợi ý lại phim đã chọn; lấy mô hình tốt nhất theo test; để web báo một mô hình nhưng thực tế dùng mô hình khác cho người dùng mới.

---

## 7. Phần Q-Learning mở rộng: chỉ bắt đầu khi phần bắt buộc đã đạt

Nếu tuần 3 các mô hình chính, bảng validation và demo cơ bản đã hoàn thành, nhóm có thể chia thêm như sau:

| Người | Việc nhỏ cần làm |
|---|---|
| TV1 | Tạo dữ liệu trạng thái/hồ sơ từ train, xác định phim đã xem và chưa xem |
| TV2 | Đề xuất trạng thái, hành động, phần thưởng và cách cập nhật Q |
| TV3 | Cài hai chiến lược đối chứng: gợi ý ngẫu nhiên, gợi ý phim phổ biến |
| TV4 | Huấn luyện Q-Learning trong môi trường mô phỏng với seed cố định |
| TV5 | Vẽ cumulative reward, ghi rõ giả định mô phỏng và ghép vào báo cáo |

Nếu chưa đủ thời gian, ghi Q-Learning là **hướng mở rộng chưa thực nghiệm**. Không lấy dữ liệu rating lịch sử làm bằng chứng trực tiếp rằng người dùng sẽ thích một phim chưa từng được gợi ý.

## 8. Danh sách kiểm tra trước khi nộp

- [ ] Nguồn MovieLens, phiên bản dữ liệu, quy tắc chia 70/15/15 và số dòng từng tập được ghi rõ.
- [ ] Có file 5 lần chia tiến theo thời gian; mọi mô hình rating dùng cùng giao thức.
- [ ] Test chỉ được đánh giá sau khi khóa tham số và chọn mô hình triển khai.
- [ ] Có baseline, Linear, XGBoost, SVD, NCF trong cùng bảng RMSE/MAE, train time và inference time.
- [ ] Có chỉ số classification, clustering, association và giảm chiều trong đúng mục báo cáo.
- [ ] Có learning curve NCF và giải thích hiện tượng overfitting nếu xuất hiện.
- [ ] Web hoạt động với người dùng mới và không gợi ý phim đã chọn; ghi rõ khi nào dùng fallback.
- [ ] Mỗi thành viên đã bàn giao mã, file kết quả, phần báo cáo và có thể trình bày công việc của mình.
