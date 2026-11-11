# Kết quả thực nghiệm và cách diễn giải

Phần này trả lời ba câu hỏi cho từng nhóm: bài toán nào được giải quyết, chỉ số nào dùng để quyết định, và kết quả hiện có cho thấy mô hình đáp ứng đến đâu. Ngưỡng trong tài liệu được chia thành **điều kiện hợp lệ của chỉ số** và **mốc chất lượng tham khảo**. Trừ các ranh giới được xác định bằng công thức (như R² = 0 hay Lift = 1), các mốc chất lượng là quy ước vận hành cho đề tài, không phải chuẩn đúng cho mọi bộ dữ liệu.

## 1. Phạm vi và giới hạn kết quả

Các kết quả dưới đây lấy từ các lần chạy đã lưu trong `artifacts/experiment-lab` (seed 42, một lần lặp cho mỗi nhóm): hồi quy `20261103T153707-a3440a34`, phân loại `20261103T153827-7731defa`, phân cụm `20261102T164848-6f37dc91`, luật kết hợp `20261102T164850-964b5255`, PCA `20261102T164854-e8cd8ef3`. Hồi quy và phân loại dùng mẫu 150 người dùng, gồm 24.319 rating: 17.023 train, 3.648 validation và 3.648 test. Phân cụm, luật kết hợp và PCA dùng mẫu 80 người dùng, gồm 11.748 rating: 8.223 train, 1.762 validation và 1.763 test. Các tập được chia theo thời gian; ngưỡng chọn trên validation được giữ nguyên khi đánh giá test.

Vì quy mô mẫu khác nhau và mỗi cấu hình mới được chạy một seed, các con số chỉ mô tả các lần chạy này. Không so trực tiếp trị số giữa các nhóm, không xem chênh lệch rất nhỏ trong một lần chạy là khác biệt có ý nghĩa thống kê, và không suy rộng kết quả mẫu ra toàn bộ MovieLens 25M. Khi báo cáo kết quả mới, cần ghi kèm seed, số người dùng/rating, cấu hình và phiên bản dữ liệu.

## 2. Bảng chọn nhóm theo yêu cầu bài toán

| Yêu cầu cần giải quyết | Nhóm phù hợp | Chỉ số ưu tiên | Quy tắc chọn trong nhóm |
|---|---|---|---|
| Ước lượng một người dùng sẽ chấm phim bao nhiêu sao | Hồi quy | RMSE, kèm MAE và R² | Chọn RMSE validation thấp nhất; dùng test để xác nhận. Xem MAE để diễn giải sai số theo đơn vị sao. |
| Dự đoán người dùng có thích phim hay không | Phân loại | F1 cùng Precision/Recall | Chọn F1 validation cao nhất nếu cần cân bằng; ưu tiên Precision khi cần ít gợi ý sai, hoặc Recall khi cần tìm đủ phim thích. |
| Khám phá nhóm phim hoặc người dùng có hồ sơ tương tự | Phân cụm | Silhouette, đối chiếu CHI và DBI | Cần có ít nhất hai cụm hợp lệ. Silhouette cao, CHI cao và DBI thấp là tín hiệu đồng thuận; kiểm tra cả tỷ lệ điểm bị DBSCAN loại thành nhiễu. |
| Tìm phim thường được thích cùng nhau | Luật kết hợp | Lift, kèm Support và Confidence | Lift phải lớn hơn 1 để biểu thị liên kết dương; cần Support đủ để luật không chỉ dựa vào vài giao dịch. |
| Nén đặc trưng nhưng vẫn dự đoán rating tốt | Giảm chiều PCA | RMSE của Ridge sau PCA, kèm Explained Variance Ratio và Reconstruction Error | Chọn số chiều trên validation theo mức nén và sai số dự đoán; đối chiếu test với Ridge không PCA. |

## 3. Kết quả và ngưỡng diễn giải theo nhóm

### 3.1. Hồi quy: dự đoán rating dạng số

Rating thật nằm trên thang 0,5–5 sao. MAE và RMSE càng thấp càng tốt; MSE cùng thứ tự với RMSE vì RMSE là căn bậc hai của MSE. R² > 0 nghĩa là mô hình tốt hơn dự đoán hằng bằng trung bình rating của chính tập đang chấm; R² = 0 tương đương baseline đó; R² < 0 kém hơn baseline. R² có thể âm, còn MAE/MSE/RMSE không âm. Nếu rating thật không có phương sai, R² không xác định.

Để diễn giải MAE trên thang có bước rating 0,5 sao, có thể dùng mốc dự án sau: MAE ≤ 0,50 là tốt; 0,50–0,75 là mức chấp nhận được cho thử nghiệm ban đầu; MAE > 0,75 là còn yếu. Đây là mốc thực hành để người đọc hình dung mức sai số, không thay thế việc so với baseline và không phải ngưỡng phổ quát. Với mọi mô hình, phải đánh giá trên cùng tập và cùng cách chia.

| Thuật toán | MAE (Val/Test) | MSE (Val/Test) | RMSE (Val/Test) | R² (Val/Test) |
|---|---:|---:|---:|---:|
| Linear Regression | 0,884 / 0,974 | 1,247 / 1,465 | 1,116 / 1,210 | −0,152 / 0,050 |
| XGBoost Regressor | 0,861 / 0,958 | 1,204 / 1,456 | 1,097 / 1,207 | −0,112 / 0,055 |
| SVD/MF có bias | **0,851 / 0,933** | 1,246 / 1,468 | 1,116 / 1,212 |  −0,151 / 0,047 |
| NCF hồi quy | 0,852 / 0,933 | 1,250 / 1,469 | 1,118 / 1,212 | −0,155 / 0,047 |

**Nhận xét và lựa chọn:** XGBoost có RMSE thấp nhất trên validation (1,097) và test (1,207), nên là ứng viên đầu nếu mục tiêu chính là hạn chế sai số lớn. SVD có MAE test thấp nhất (0,933), tức sai lệch tuyệt đối trung bình thấp hơn một chút. Chênh lệch RMSE giữa XGBoost và SVD trên test chỉ khoảng 0,005 sao; với một seed chưa đủ căn cứ kết luận XGBoost vượt trội ổn định. Theo quy tắc chọn bằng RMSE validation, XGBoost là lựa chọn của lần chạy này; theo MAE, SVD là lựa chọn. Cần nêu rõ chỉ số chính thay vì gộp các chỉ số thành một thứ hạng duy nhất.

MAE test của cả bốn mô hình đều lớn hơn 0,75 sao nên chưa đạt mốc chất lượng tham khảo của đề tài. (R^2) validation âm ở cả bốn mô hình, còn (R^2) test chỉ khoảng 0,05. Kết quả cho thấy lợi thế dự đoán còn yếu và có khác biệt rõ giữa validation và test theo thời gian; nên cải thiện đặc trưng/độ bao phủ dữ liệu và chạy nhiều seed trước khi tuyên bố mô hình tốt.

### 3.2. Phân loại: dự đoán thích/không thích

Rating từ 4 sao được gán nhãn “thích”; rating thấp hơn được gán “không thích”. Accuracy, Precision, Recall và F1 nằm trong [0, 1], càng cao càng tốt, nhưng không có một ngưỡng tuyệt đối dùng được cho mọi ứng dụng. Accuracy cần so với dự đoán lớp đa số; Precision/Recall phải được đặt theo chi phí của gợi ý sai và phim thích bị bỏ sót. Trong đề tài, ngưỡng xác suất được chọn theo F1 validation rồi giữ nguyên cho test.

| Thuật toán | Accuracy (Val/Test) | Precision (Val/Test) | Recall (Val/Test) | F1 (Val/Test) |
|---|---:|---:|---:|---:|
| Logistic Regression | **0,573 / 0,560** | **0,573 / 0,544** | 0,999 / 0,992 | **0,728 / 0,703** |
| Random Forest | 0,573 / 0,559 | 0,572 / 0,543 | **0,999 / 0,992** | 0,728 / 0,702 |

**Mốc vận hành tham khảo:** nếu hệ thống cần đề xuất phim đáng tin, đặt mục tiêu Precision ≥ 0,70; nếu ưu tiên tìm đủ phim người dùng thích, đặt Recall ≥ 0,70. Có thể dùng F1 ≥ 0,70 làm mức sàn cân bằng cho thử nghiệm ban đầu. Đây là yêu cầu dự án cần thống nhất theo mục đích sử dụng, không phải định luật thống kê. Không kết luận mô hình tốt chỉ vì một chỉ số cao.

**Nhận xét và lựa chọn:** Tỷ lệ lớp thích trên test xấp xỉ 52,4%, nên baseline dự đoán lớp đa số đạt Accuracy khoảng 0,524; baseline dự đoán mọi mẫu là thích đạt F1 khoảng 0,688. Logistic đạt Accuracy 0,560 và F1 0,703, tức có cải thiện nhưng còn khiêm tốn so với các baseline đơn giản. Logistic nhỉnh hơn về Accuracy, Precision và F1 test, nhưng chênh lệch F1 với Random Forest chỉ khoảng 0,0004. Logistic là lựa chọn theo F1 trong lần chạy này, song hai mô hình gần như ngang nhau. Recall trên test xấp xỉ 0,992 nhưng Precision chỉ khoảng 0,544: mô hình tìm được gần hết phim thích nhưng trong các phim dự đoán là thích vẫn có nhiều phim không đạt nhãn thích. Vì vậy, kết quả chỉ phù hợp khi ưu tiên độ bao phủ; chưa đáp ứng mục tiêu gợi ý có độ tin cậy cao nếu yêu cầu Precision khoảng 0,70 trở lên.

### 3.3. Phân cụm: khám phá nhóm phim tương tự

Các chỉ số ở đây đo cấu trúc hình học, không đo độ chính xác dự đoán nhãn. Silhouette thuộc [−1, 1]: giá trị gần 1 biểu thị cụm tách biệt, gần 0 biểu thị chồng lấn, âm thường báo điểm nằm gần cụm khác hơn cụm được gán. Các mốc diễn giải kinh nghiệm thường dùng là ≤0,25: chưa thấy cấu trúc đáng kể; >0,25–0,50: cấu trúc yếu; >0,50–0,70: tương đối hợp lý; >0,70: cấu trúc mạnh. Đây không phải ranh giới cứng; cần cân nhắc biểu diễn và hình dạng cụm. CHI càng cao và DBI càng thấp thường tốt hơn, nhưng không có ngưỡng CHI/DBI phổ quát: chỉ so sánh chúng trên cùng dữ liệu, đặc trưng, thang đo và tập đánh giá. Các hướng dẫn này phù hợp với cách diễn giải Silhouette của Kaufman và Rousseeuw; thư viện scikit-learn cũng nêu Silhouette chỉ xác định khi có từ 2 đến n−1 nhãn cụm. ([Kaufman & Rousseeuw, 1990](https://doi.org/10.1002/9780470316801); [scikit-learn: `silhouette_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html))

| Thuật toán | Silhouette (Val/Test) | CHI (Val/Test) | DBI (Val/Test) |
|---|---:|---:|---:|
| K-Means | 0,041 / 0,089 | 2,816 / 5,459 | 2,826 / 2,214 |
| DBSCAN | N/A / N/A | N/A / N/A | N/A / N/A |

**Nhận xét và lựa chọn:** Silhouette K-Means thấp hơn nhiều so với mốc 0,25, nên không có bằng chứng về các cụm tách biệt đáng kể theo bộ đặc trưng hiện tại. CHI/DBI không có ngưỡng tuyệt đối để đổi các giá trị này thành “đạt”. DBSCAN không tạo đủ cụm hợp lệ trên các điểm không nhiễu để tính chỉ số; N/A không phải 0 và không chứng minh rằng mọi cấu hình DBSCAN đều kém. K-Means là phương án duy nhất có điểm số đọc được trong lần chạy này, nhưng chưa thể gọi là nghiệm phân cụm tốt. Cần điều chỉnh K/eps/min_samples trên validation và báo thêm tỷ lệ đối tượng bị gán nhiễu, vì chỉ chấm các điểm không nhiễu có thể làm kết quả trông tốt hơn thực tế.

### 3.4. Luật kết hợp: tìm phim thường được thích cùng nhau

Support và Confidence thuộc [0, 1]. Lift không âm khi Support của vế phải khác 0: Lift > 1 biểu thị liên kết dương; Lift = 1 tương đương độc lập; Lift < 1 biểu thị liên kết âm. Không có một ngưỡng Support/Confidence chung: Support cần đủ lớn để luật dựa trên nhiều giao dịch; Confidence cần so với tần suất nền của phim ở vế phải; Lift cần được đọc cùng Support để tránh đánh giá quá cao luật hiếm.

Trong cấu hình khai phá này, chỉ giữ luật có Support train ≥ 0,08 và Confidence train ≥ 0,30. Đây là điều kiện tạo luật trên train; không áp dụng như điều kiện loại bỏ trên validation/test. Trên các tập giữ lại, các chỉ số là trung bình của cùng một tập luật cố định nên một giá trị trung bình thấp hơn ngưỡng train là tín hiệu cần xem xét khả năng tổng quát hóa, không phải lỗi công thức.

| Thuật toán | Support (Val/Test) | Confidence (Val/Test) | Lift (Val/Test) |
|---|---:|---:|---:|
| Apriori | 0,044 / 0,054 | 0,248 / 0,372 | 1,328 / 2,613 |
| FP-Growth | 0,044 / 0,054 | 0,248 / 0,372 | 1,328 / 2,613 |

**Nhận xét và lựa chọn:** Lift trung bình lớn hơn 1 ở cả hai tập, cho thấy liên kết dương trong mẫu này. Tuy nhiên, Confidence validation (0,248) thấp hơn mức Confidence train tối thiểu 0,30, còn Lift test tăng mạnh so với validation; cần kiểm tra từng luật thay vì chỉ dựa vào trung bình để xác nhận tính ổn định. Apriori và FP-Growth cho cùng các luật và cùng chỉ số chất lượng trong lần chạy này, nên không có thuật toán thắng về Support/Confidence/Lift. Muốn chọn giữa hai cách khai phá cần so thời gian chạy/bộ nhớ trên cùng cấu hình; các chỉ số chất lượng hiện có không trả lời câu hỏi đó.

### 3.5. PCA: nén đặc trưng nhưng giữ khả năng dự đoán rating

Explained Variance Ratio (EVR) thuộc [0, 1] và tăng theo số thành phần; EVR cao nghĩa là giữ lại nhiều phương sai của đặc trưng, nhưng không bảo đảm giữ được thông tin hữu ích nhất để dự đoán rating. Reconstruction Error không âm và càng thấp càng ít sai lệch khi khôi phục đặc trưng. Downstream Model Performance ở đây là RMSE của cùng Ridge, nên càng thấp càng tốt và phải so với Ridge không PCA trên cùng tập. Không có một ngưỡng EVR hay Reconstruction Error chung để tuyên bố PCA “tốt”. EVR ≥ 0,90 có thể dùng làm mục tiêu nếu yêu cầu là giữ phần lớn phương sai tổng thể, nhưng không bắt buộc nếu mục tiêu chính là tiết kiệm chiều mà vẫn dự đoán tốt.

| Cấu hình | EVR | Reconstruction Error (Val/Test) | Ridge RMSE (Val/Test) |
|---|---:|---:|---:|
| Không PCA, 44 chiều | 1,000 | 0,000 / 0,000 | 1,060 / 1,065 |
| PCA, 1 chiều | 0,093 | 0,840 / 0,936 | 0,977 / 0,897 |
| PCA, 2 chiều | 0,158 | 0,803 / 0,881 | 0,970 / 0,893 |
| PCA, 4 chiều | 0,260 | 0,736 / 0,813 | 0,971 / **0,881** |
| PCA, 8 chiều | 0,414 | 0,657 / 0,735 | 0,970 / 0,887 |
| PCA, 16 chiều | 0,642 | 0,574 / 0,664 | **0,970** / 0,883 |
| PCA, 44 chiều | 1,000 | 0,000 / 0,000 | 1,060 / 1,065 |

**Quy tắc chọn đề xuất cho mục tiêu dự đoán:** dùng Validation để tìm RMSE tốt nhất trong các cấu hình PCA; chọn số chiều nhỏ nhất có RMSE không cao hơn quá 1% so với mức tốt nhất; cuối cùng xác nhận rằng RMSE test không xấu hơn baseline không PCA quá 1%. Đây là ngưỡng vận hành được đề xuất khi diễn giải run hiện tại; vì quy tắc chưa được khóa trước lần chạy này, kết luận chọn PCA 1 chiều chỉ mang tính thăm dò. Ở lần chạy xác nhận tiếp theo, cần chốt ngưỡng trước khi xem kết quả và chỉ dùng Test để đánh giá cuối. Theo quy tắc đó, PCA 1 chiều là cấu hình nén gọn nhất trong 1% quanh RMSE validation thấp nhất; RMSE test 0,897 cũng thấp hơn baseline không PCA 1,065. Nếu mục tiêu thay vào đó là giữ ít nhất 90% phương sai, PCA 1 chiều không đạt và cấu hình 44 chiều gần như không giảm số chiều. Hai yêu cầu “nén tối đa” và “giữ phương sai tổng thể” cần được nêu riêng.

**Nhận xét và lựa chọn:** Trong lần chạy này PCA cải thiện RMSE của Ridge ở các cấu hình 1–16 chiều so với không PCA. PCA 16 chiều có RMSE validation thấp nhất về trị số chính xác, nhưng các cấu hình 2, 8 và 16 chiều gần như hòa nhau sau khi làm tròn; PCA 4 chiều có RMSE test thấp nhất. Test không được dùng để chọn cấu hình. Theo quy tắc dự án ở trên, chọn PCA 1 chiều để ưu tiên nén tối đa mà vẫn giữ chất lượng dự đoán gần mức tốt nhất validation. EVR chỉ 0,093 và Reconstruction Error còn cao, nên cấu hình này không phù hợp nếu yêu cầu chính là tái tạo đầy đủ cấu trúc đặc trưng. Chỉ có một seed; kết quả cần được lặp lại trước khi kết luận lợi ích nén là ổn định.

## 4. Kết luận so sánh

- **Dự đoán rating:** XGBoost được chọn nếu lấy RMSE validation làm tiêu chí chính; SVD đạt MAE test thấp nhất. Chưa có bằng chứng về một mô hình vượt trội rõ ràng, và mức sai số tuyệt đối vẫn yếu theo mốc tham khảo của thang rating.
- **Dự đoán thích/không thích:** Logistic đạt F1 test cao nhất nhưng gần như hòa Random Forest. Recall cao, Precision thấp; phù hợp mục tiêu bao phủ hơn là mục tiêu gợi ý có độ tin cậy cao.
- **Phân cụm:** K-Means là phương án duy nhất có chỉ số hợp lệ ở lần chạy này, nhưng Silhouette cho thấy cấu trúc cụm rất yếu. DBSCAN cần hiệu chỉnh tham số rồi chạy lại.
- **Luật kết hợp:** Apriori và FP-Growth hòa về chất lượng chỉ số; Lift lớn hơn 1 cho thấy liên kết dương, còn chênh lệch validation/test yêu cầu kiểm tra độ ổn định của từng luật.
- **Giảm chiều:** PCA 1 chiều là lựa chọn theo quy tắc nén và dự đoán đã nêu; không đạt mục tiêu giữ 90% phương sai. Nếu yêu cầu là bảo toàn cấu trúc dữ liệu tổng thể, cấu hình hiện tại chưa chứng minh được có thể giảm chiều đáng kể.

“Tốt nhất” luôn gắn với yêu cầu và chỉ số chính của bài toán. Khi so sánh, dùng validation để chọn, giữ nguyên lựa chọn để báo cáo test, đọc các chỉ số bổ trợ và baseline, đồng thời không coi N/A là 0. Không cộng hoặc xếp hạng trực tiếp RMSE, F1, Silhouette, Lift và EVR vì chúng đo các mục tiêu khác nhau.
