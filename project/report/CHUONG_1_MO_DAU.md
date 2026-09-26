# CHƯƠNG 1. MỞ ĐẦU

## 1.1. Đặt vấn đề

Một người xem có thể thích phim khoa học viễn tưởng nhưng không thích mọi phim thuộc thể loại này. Sở thích còn phụ thuộc vào nhịp kể, diễn viên, thời điểm xem và nhiều yếu tố khó mô tả bằng vài từ khóa. Vì vậy, chỉ lọc phim theo thể loại hoặc xếp theo mức độ phổ biến chưa đủ để tạo danh sách phù hợp với từng người. Lịch sử đánh giá phim cung cấp một cách tiếp cận khác: từ những phim người dùng đã chấm điểm, hệ thống ước lượng mức độ yêu thích của họ đối với các phim chưa xem.

Bài toán trở nên khó hơn khi số điểm đánh giá ít hơn rất nhiều so với số cặp người dùng–phim có thể có. Một ô trống trong bảng đánh giá chỉ cho biết người dùng chưa chấm điểm phim đó; không thể kết luận họ không thích phim. Mô hình cũng thiếu thông tin khi gặp người dùng hoặc phim mới. Những trường hợp này cần được nhận diện trong lúc đánh giá và có cách xử lý khi đưa hệ thống vào giao diện sử dụng.

Nhóm chọn MovieLens 25M để nghiên cứu bài toán trên. Theo GroupLens, bộ dữ liệu gồm 25.000.095 lượt đánh giá của 162.541 người dùng và thông tin về 62.423 phim [5]. Mỗi lượt đánh giá có dấu thời gian, nhờ đó nhóm có thể huấn luyện trên các tương tác xảy ra trước và kiểm tra dự đoán ở giai đoạn sau. Câu hỏi chính của đồ án là: **trên cùng dữ liệu và cùng cách đánh giá, mô hình nào dự đoán điểm phim đủ tốt, có chi phí tính toán phù hợp và có thể dùng trong một web demo gợi ý phim?**

Để trả lời, nhóm so sánh mô hình điểm trung bình, hồi quy, phân rã ma trận và Neural Collaborative Filtering (NCF). Kết quả dự đoán điểm được dùng làm cơ sở xếp hạng phim, nhưng chất lượng danh sách gợi ý cũng cần được xem xét riêng: sai số điểm thấp chưa bảo đảm những phim đứng đầu danh sách là lựa chọn hữu ích nhất.

## 1.2. Tình hình nghiên cứu và tính mới của đồ án

### 1.2.1. Nghiên cứu liên quan

Harper và Konstan trình bày quá trình hình thành các bộ dữ liệu MovieLens và lưu ý rằng dữ liệu đánh giá phản ánh hành vi của những người đã sử dụng hệ thống, chứ không phải sở thích của toàn bộ người xem phim [1]. Đây là lý do nhóm phải nêu rõ dữ liệu nào được quan sát, cách chia tập và giới hạn khi suy rộng kết quả.

Trong lọc cộng tác, Koren, Bell và Volinsky mô tả cách biểu diễn người dùng và phim bằng các nhân tố tiềm ẩn để ước lượng điểm đánh giá chưa có [2]. Nhóm sử dụng hướng phân rã ma trận này cho mô hình SVD. Một hướng khác là học từ các đặc trưng được xây dựng trước, chẳng hạn thống kê đánh giá của người dùng, của phim hoặc thông tin thể loại. Linear Regression cho một mốc so sánh đơn giản; XGBoost, theo công trình của Chen và Guestrin, sử dụng tập hợp các cây để học những quan hệ phức tạp hơn giữa đặc trưng và đầu ra [3].

He và cộng sự đề xuất NCF, dùng embedding và mạng nơ-ron để học tương tác giữa người dùng với sản phẩm [4]. Nghiên cứu gốc tập trung vào **phản hồi ngầm**, tức hành vi tương tác thay cho điểm chấm cụ thể. Trong đồ án này, NCF được điều chỉnh để dự đoán **rating tường minh** từ 0,5 đến 5 sao. Do khác bài toán, kết quả của công trình gốc không được dùng làm bằng chứng rằng NCF sẽ tốt hơn SVD hoặc XGBoost trên MovieLens 25M; nhóm cần đo lại bằng cùng một giao thức thực nghiệm.

### 1.2.2. Tính mới của đồ án

Đồ án sử dụng các phương pháp đã được công bố. Đóng góp trong phạm vi môn học nằm ở cách đặt chúng vào một phép so sánh có thể kiểm tra lại: cùng ranh giới train/validation/test theo thời gian, cùng các lần kiểm chứng trong train và cùng quy tắc ghi kết quả. Nhóm sẽ công bố cả số lượng dự đoán hợp lệ và thời gian chạy, vì một mô hình có sai số thấp trên một nhóm nhỏ người dùng không thể được so sánh trực tiếp với mô hình dự đoán cho toàn bộ tập.

Khảo sát dữ liệu ban đầu của nhóm cho thấy **88,48% lượt đánh giá trong validation thuộc về người dùng chưa xuất hiện trong train**. Tỷ lệ này xuất phát từ cách chia toàn cục theo thời gian đã chọn và ảnh hưởng trực tiếp đến mô hình học embedding theo ID. Vì vậy, nhóm tách kết quả trên nhóm người dùng/phim đã có đủ lịch sử khỏi kết quả trên toàn tập, đồng thời chuẩn bị quy tắc gợi ý ban đầu cho người dùng mới. Mô hình đưa vào web demo sẽ được chọn từ kết quả validation và chi phí tính toán thực tế.

## 1.3. Mục tiêu nghiên cứu

### 1.3.1. Mục tiêu tổng quát

Xây dựng quy trình dự đoán điểm đánh giá và gợi ý phim từ MovieLens 25M; so sánh các mô hình theo độ chính xác, khả năng xử lý trường hợp ít dữ liệu và chi phí tính toán; sau đó triển khai phương án phù hợp trong một web demo.

### 1.3.2. Mục tiêu nghiên cứu cụ thể

1. Kiểm tra dữ liệu gốc, mô tả phân bố rating và mức độ thưa của tương tác; lưu dữ liệu đã xử lý cùng quy tắc tạo lại.
2. Chia dữ liệu theo thời gian với tỷ lệ mục tiêu 70% train, 15% validation và 15% test; tạo năm lần kiểm chứng tiến theo thời gian bên trong train. Mọi đặc trưng thống kê và ngưỡng lọc phải được xác định từ phần train tương ứng.
3. Huấn luyện baseline điểm trung bình, Linear Regression, XGBoost, SVD và NCF cho bài toán dự đoán rating; ghi lại tham số, seed, thời gian huấn luyện và thời gian suy luận.
4. So sánh RMSE, MAE trên cùng tập đánh giá; phân tích riêng nhóm đã có lịch sử và nhóm người dùng/phim chưa được mô hình quan sát. Với gợi ý Top-N, xác định rõ tập phim ứng viên trước khi tính chỉ số xếp hạng.
5. Xây dựng web demo cho phép chọn hoặc chấm điểm phim đã xem, nhận danh sách Top-N và loại các phim vừa chọn khỏi danh sách. Demo cần có phương án gợi ý khi người dùng chưa có ID trong tập huấn luyện.

Các thí nghiệm phân loại, phân cụm, giảm chiều và khai phá luật kết hợp giúp nhóm tìm hiểu dữ liệu từ những góc nhìn khác. Q-Learning là hướng mở rộng nếu phần so sánh và demo chính hoàn thành đúng tiến độ.

## 1.4. Đối tượng và quy mô nghiên cứu

### 1.4.1. Phạm vi

Dữ liệu sử dụng là MovieLens 25M do GroupLens công bố. Phần chính của đồ án lấy các cột `userId`, `movieId`, `rating`, `timestamp` trong `ratings.csv`; tên và thể loại phim trong `movies.csv` phục vụ mô tả dữ liệu, tạo đặc trưng khi phù hợp và hiển thị trên web. Rating nhận các giá trị từ 0,5 đến 5 sao, cách nhau 0,5 sao [5]. Nhóm đã kiểm tra và chia 25.000.095 lượt đánh giá thành train, validation và test theo thứ tự thời gian. Nếu một mô hình vượt quá tài nguyên máy, nhóm sẽ ghi rõ quy tắc lấy mẫu và chỉ so sánh trực tiếp các mô hình chạy trên cùng tập bản ghi.

Phạm vi đánh giá là **ngoại tuyến**: mô hình được kiểm tra bằng những đánh giá đã có trong dữ liệu, chưa được thử nghiệm với người dùng thật. MovieLens 25M không cung cấp thông tin nhân khẩu học và những người dùng được chọn vào bộ dữ liệu đều đã đánh giá ít nhất 20 phim [5]. Vì thế, web demo cho người dùng hoàn toàn mới là một tình huống mô phỏng; kết quả ngoại tuyến không đủ để khẳng định mức hài lòng của người dùng khi sử dụng thực tế.

### 1.4.2. Đối tượng

Đơn vị phân tích chính là một lượt đánh giá: một người dùng chấm điểm cho một phim tại một thời điểm. Từ các lượt đánh giá đã quan sát, nhóm dự đoán điểm cho cặp người dùng–phim ở giai đoạn sau và xếp hạng các phim ứng viên để tạo Top-N. Mức độ tương tác của từng người dùng và từng phim cũng là đối tượng phân tích, vì nó quyết định mô hình có đủ lịch sử để đưa ra dự đoán cá nhân hóa hay không.

Ở các thí nghiệm phụ, nhóm quy ước rating từ 4 sao trở lên là tín hiệu yêu thích cho bài toán phân loại và tạo tập phim được đánh giá tích cực để khai phá luật kết hợp. Quy ước này chỉ phục vụ thực nghiệm; ô chưa có rating không được tự động gán thành phản hồi tiêu cực.

## 1.5. Nội dung và kỹ thuật nghiên cứu

### 1.5.1. Nội dung nghiên cứu

Đồ án đi theo năm bước, mỗi bước tạo đầu vào rõ ràng cho bước sau:

1. **Thu thập dữ liệu:** lấy MovieLens 25M, kiểm tra cấu trúc file, giá trị thiếu hoặc không hợp lệ, bản ghi trùng và phân bố tương tác.
2. **Chia tập:** sắp xếp rating theo thời gian, chia train/validation/test theo tỷ lệ mục tiêu 70/15/15 và tạo năm lần kiểm chứng tiến theo thời gian trong train. Giữ lại thông tin về người dùng/phim ít tương tác để đánh giá mức bao phủ.
3. **Huấn luyện:** xây dựng baseline và bốn mô hình dự đoán rating chính; tìm tham số trên dữ liệu được phép dùng ở từng giai đoạn. Các bài toán phân loại, phân cụm, giảm chiều và luật kết hợp được thực hiện như thí nghiệm bổ sung.
4. **Dự đoán:** xuất điểm dự đoán gắn với ID bản ghi để đối chiếu đúng rating thật; dùng điểm hoặc quy tắc dự phòng để xếp hạng phim chưa được người dùng chọn trong web demo.
5. **Đánh giá:** đối chiếu độ chính xác, mức bao phủ, thời gian huấn luyện và độ trễ suy luận; chọn mô hình từ validation. Tập test chỉ dùng cho lần đánh giá cuối sau khi nhóm đã chốt cấu hình.

### 1.5.2. Kỹ thuật nghiên cứu

Nhóm dùng Python để xử lý và thực nghiệm. Dữ liệu dung lượng lớn được đọc, kiểm tra và lưu ở định dạng Parquet; mã tiền xử lý hiện dùng DuckDB. Các mô hình dự đoán rating gồm Linear Regression, XGBoost, SVD và NCF; NCF dự kiến cài bằng PyTorch. Ở phần phân tích bổ sung, Logistic Regression và Random Forest phục vụ phân loại, K-Means và DBSCAN để khảo sát nhóm, PCA hoặc TruncatedSVD để giảm chiều, còn Apriori để khai phá luật kết hợp. Nhóm dự kiến dùng scikit-learn, mlxtend và xây dựng web demo bằng Streamlit. Phiên bản thư viện và cấu hình chạy cần được lưu cùng kết quả thực nghiệm.

Việc chia tập dựa trên `timestamp` và giữ các rating cùng thời điểm ở cùng một phía của ranh giới. Năm lần kiểm chứng trong train dùng phần dữ liệu quá khứ ngày càng lớn để dự đoán một giai đoạn tiếp theo. Nhóm dự kiến dùng tìm kiếm tham số của scikit-learn cho mô hình phù hợp và Optuna cho NCF; quá trình tìm kiếm không sử dụng tập test. Với các mô hình học theo ID người dùng/phim, kết quả được báo cáo trên cùng tập con đủ lịch sử và trên toàn bộ validation khi đã áp dụng quy tắc dự phòng.

RMSE và MAE là thước đo chính của dự đoán rating. Với Top-N, nhóm dự kiến dùng Precision@K, Recall@K và NDCG@K, kèm mô tả cách chọn phim ứng viên và cách xác định một gợi ý là phù hợp. Các bài toán phụ sử dụng Accuracy, F1, AUC-ROC cho phân loại; Silhouette và Davies–Bouldin cho phân cụm; Support, Confidence và Lift cho luật kết hợp. Đường loss trên train và validation được dùng để nhận diện dấu hiệu quá khớp của NCF. Thời gian huấn luyện và suy luận được ghi cùng điều kiện phần cứng để việc so sánh có ý nghĩa.

## Tài liệu tham khảo

[1] F. M. Harper và J. A. Konstan, “The MovieLens Datasets: History and Context,” *ACM Transactions on Interactive Intelligent Systems*, tập 5, số 4, 2015. DOI: [10.1145/2827872](https://doi.org/10.1145/2827872).

[2] Y. Koren, R. Bell và C. Volinsky, “Matrix Factorization Techniques for Recommender Systems,” *Computer*, tập 42, số 8, tr. 30–37, 2009. DOI: [10.1109/MC.2009.263](https://doi.org/10.1109/MC.2009.263).

[3] T. Chen và C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” *Proceedings of KDD*, 2016. DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).

[4] X. He và cộng sự, “Neural Collaborative Filtering,” *Proceedings of WWW*, 2017. [Bản công bố của tác giả](https://arxiv.org/abs/1708.05031).

[5] GroupLens Research, “[MovieLens 25M Dataset README](https://files.grouplens.org/datasets/movielens/ml-25m-README.html),” 2019.
