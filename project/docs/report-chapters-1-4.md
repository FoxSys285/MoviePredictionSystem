# CHƯƠNG 1: TỔNG QUAN

## 1.1. BỐI CẢNH VÀ LÝ DO CHỌN ĐỀ TÀI

Các dịch vụ xem phim có danh mục lớn, trong khi người dùng thường chỉ có thời gian cân nhắc một phần nhỏ số phim. Hệ gợi ý hỗ trợ sắp xếp nội dung theo lịch sử và tín hiệu tương tác, giúp người dùng tập trung vào các lựa chọn có khả năng phù hợp. MovieLens là một nền tảng nghiên cứu và bộ dữ liệu được sử dụng rộng rãi để khảo sát hành vi đánh giá, đề xuất và đánh giá hệ gợi ý [1], [2].

MovieLens 25M có rating tường minh, mã người dùng/phim và timestamp. Ma trận tương tác rất thưa: một người dùng chỉ đánh giá một phần nhỏ danh mục. Do đó, các mô hình phải học từ những rating quan sát được mà không xem phim chưa đánh giá là phim bị chấm 0. Rating và độ phổ biến cũng thay đổi theo thời gian; cách chia ngẫu nhiên có thể làm cho dữ liệu tương lai xuất hiện trong cả Train lẫn Test. Thiết kế theo thời gian giúp đánh giá gần hơn tình huống hệ thống dự đoán từ lịch sử đã có [2], [15].

Một thuật toán có thể đạt sai số thấp nhưng tốn nhiều thời gian huấn luyện hoặc khó phục vụ suy luận; mô hình đơn giản có thể phù hợp hơn nếu chênh lệch chất lượng nhỏ. Bởi vậy, đồ án không chỉ xây dựng một mô hình dự đoán rating mà còn tổ chức các nhóm thuật toán theo đúng đầu ra của từng bài toán, đánh giá bằng metric phù hợp và đo chi phí tính toán [3], [16].

## 1.2. MỤC TIÊU ĐỀ TÀI

Mục tiêu tổng quát là xây dựng và rà soát một hệ thống học máy dùng MovieLens 25M để dự đoán rating, phân tích sở thích và minh họa gợi ý phim cá nhân hóa trên giao diện web. Kết quả của các mô hình dự đoán rating sẽ được so sánh về độ chính xác, độ ổn định và chi phí trước khi chọn mô hình cho demo.

Đồ án đặt ra các mục tiêu cụ thể: chuẩn bị dữ liệu có dấu vết và chia theo thời gian; cài đặt các nhóm hồi quy, phân loại, phân cụm, luật kết hợp, giảm chiều và học tăng cường; lưu cấu hình cùng kết quả mỗi lần chạy; so sánh với baseline thích hợp; và tạo giao diện để người dùng nhập một số phim đã xem rồi nhận danh sách Top-N.

## 1.3. CÂU HỎI NGHIÊN CỨU

Các câu hỏi chính gồm: trong bốn mô hình Linear Regression, XGBoost, SVD-CF và NCF, mô hình nào đạt cân bằng tốt giữa RMSE/MAE và thời gian; Logistic Regression và Random Forest phân loại mức độ yêu thích ra sao khi nhãn có thể mất cân bằng; K-Means và DBSCAN tạo được các nhóm sở thích có độ bao phủ và khả năng diễn giải như thế nào; luật Apriori có Support, Confidence và Lift hữu ích hay không; PCA/TruncatedSVD giữ lại bao nhiêu thông tin và có cải thiện tác vụ downstream so với baseline; Q-Learning hoặc DQN có vượt Random/Popularity trong cùng môi trường offline hay không?

Các câu hỏi này được trả lời theo từng mục tiêu riêng. RMSE, F1, Silhouette và Lift đo những khía cạnh khác nhau, vì vậy đồ án không cộng chúng thành một điểm tổng hợp duy nhất. Việc chọn mô hình rating cho demo dựa trên kết quả cùng protocol, Validation, độ ổn định qua fold và chi phí đo được [16].

## 1.4. ĐỐI TƯỢNG VÀ PHẠM VI NGHIÊN CỨU

Đối tượng nghiên cứu là các lượt đánh giá phim có rating từ 0,5 đến 5 sao, liên kết với userId, movieId và thời điểm đánh giá. Metadata thể loại được dùng để tạo hồ sơ genre và phân cụm. Bộ dữ liệu không có thông tin nhân khẩu học, vì vậy đồ án không suy luận tuổi, giới tính hoặc đặc điểm cá nhân từ mã người dùng [1], [2].

Phạm vi thuật toán gồm Linear Regression/OLS hoặc Ridge, XGBoost Regressor, SVD Collaborative Filtering và NCF cho dự đoán rating; Logistic Regression và Random Forest cho phân loại; K-Means và DBSCAN cho phân cụm; Apriori cho luật kết hợp; PCA và TruncatedSVD cho giảm chiều; cùng Q-Learning cho gợi ý tuần tự. DQN được trình bày như biến thể hiện có trong Experiment Studio, tách biệt với Q-Learning dạng bảng.

Phạm vi triển khai là web demo chạy cục bộ bằng FastAPI và React/TypeScript. Đồ án dùng đánh giá offline trên dữ liệu MovieLens; kết quả mô phỏng không khẳng định tác động lên người dùng thật. Dữ liệu rating kết thúc năm 2019, nên các kết luận mô tả phạm vi lịch sử này, không đại diện cho mức phổ biến hoặc danh mục phim hiện tại.

## 1.5. PHƯƠNG PHÁP THỰC HIỆN

Đồ án kết hợp khảo cứu bài báo và tài liệu kỹ thuật với việc kiểm tra mã nguồn, dữ liệu và artifact trong repository. Trước hết, dữ liệu được kiểm tra schema, rating, ID và trùng lặp; sau đó lịch sử được chia thành Train, Validation và Test theo thời gian. Mọi thống kê và phép biến đổi học từ dữ liệu chỉ được fit trên phần Train hoặc fold tương ứng.

Các thuật toán được huấn luyện theo nhóm, chọn cấu hình trên Validation và đánh giá cuối trên Test sau khi khóa lựa chọn. Regression và Classification được xem xét qua cross-validation thời gian; NCF có learning curve và early stopping; các nhóm còn lại dùng chỉ số phù hợp với đầu ra của chúng. Thời gian fit, latency, kích thước model và độ bao phủ được ghi cùng metric chất lượng. Các kết quả và giới hạn được diễn giải dựa trên artifact thay vì thay thế bằng số liệu minh họa [16], [17].

## 1.6. ĐÓNG GÓP CỦA ĐỒ ÁN

Đóng góp chính của đồ án là tổ chức nhiều phương pháp học máy trong một ứng dụng thực nghiệm có thể cấu hình và lưu lại lịch sử; đặt các mô hình dự đoán rating cổ điển cạnh NCF để so sánh trên cùng bài toán; và làm rõ hai cách dùng thường bị gọi chung là SVD: phân rã nhân tố để dự đoán rating và giảm chiều bằng TruncatedSVD. Việc đánh giá đồng thời chất lượng, độ ổn định, độ phủ và chi phí giúp kết luận lựa chọn mô hình có căn cứ hơn.

Đồ án cũng tích hợp các thuật toán phân loại, phân cụm, khai phá luật, giảm chiều và học tăng cường vào một phòng thực nghiệm, đồng thời cung cấp web demo để xem gợi ý Top-N. Đây là đóng góp ở phạm vi ứng dụng học phần; báo cáo không tuyên bố đề xuất thuật toán mới hoặc chứng minh hiệu quả trực tuyến.

## 1.7. BỐ CỤC BÁO CÁO

Chương 1 giới thiệu bối cảnh, mục tiêu, câu hỏi nghiên cứu, phạm vi và cách thực hiện. Chương 2 trình bày cơ sở lý thuyết của các nhóm thuật toán và metric liên quan. Chương 3 mô tả MovieLens 25M, cách xác định bài toán, cấu hình thuật toán và thiết kế đánh giá; chương này cũng ghi rõ những điểm khác nhau giữa các pipeline đang có trong repository. Chương 4 được bố trí theo từng nhóm thuật toán với bảng và vị trí biểu đồ để bổ sung sau khi chạy protocol đã thống nhất. Tài liệu tham khảo liệt kê các nguồn nghiên cứu và tài liệu kỹ thuật được trích trong báo cáo.

# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

## 2.1. HỆ GỢI Ý VÀ DỮ LIỆU TƯƠNG TÁC

### 2.1.1. Rating tường minh và ma trận người dùng–phim

Hệ gợi ý cộng tác khai thác lịch sử tương tác để ước lượng mức phù hợp giữa người dùng và sản phẩm. Trong đề tài này, rating là phản hồi tường minh: một người dùng chủ động chấm một bộ phim trên thang 0,5–5 sao. Ký hiệu ma trận rating là R, tập người dùng là U, tập phim là I và tập các cặp đã có rating là Ω. Phần lớn cặp người dùng–phim không xuất hiện trong Ω, vì vậy ma trận R rất thưa. Độ thưa được tính bằng 1 − |Ω|/(|U|×|I|).

Ô chưa có rating không đồng nghĩa với rating 0 hoặc với việc người dùng không thích phim. Khi huấn luyện mô hình dự đoán, hàm mất mát chỉ được tính trên các rating quan sát. Việc điền số 0 vào toàn bộ ô khuyết sẽ biến trạng thái “chưa biết” thành một nhãn giả và làm sai mục tiêu học [2], [3].

### 2.1.2. Cold-start và thứ tự thời gian

Cold-start xảy ra khi người dùng hoặc phim chưa có đủ lịch sử để mô hình ước lượng sở thích. Mô hình nhân tố tiềm ẩn cần tương tác để học embedding; nếu một ID chưa xuất hiện trong phần huấn luyện, mô hình không thể suy ra nhân tố riêng chỉ từ ID đó. Một hệ thống thực tế cần đo tỷ lệ dự đoán được cho các trường hợp này và quy định cách dự phòng, chẳng hạn dùng trung bình, độ phổ biến hoặc thông tin thể loại.

Rating phim thay đổi theo thời điểm: mức phổ biến của phim và sở thích của người dùng không cố định. Vì vậy, tập đánh giá nên nằm sau tập huấn luyện theo thời gian. Thống kê, bộ chuẩn hóa, giảm chiều và cấu hình mô hình phải được fit trong phần dữ liệu được phép dùng; nếu chúng nhìn thấy rating tương lai, kết quả sẽ lạc quan hơn khả năng dự đoán trong triển khai [15], [16].

## 2.2. NHÓM HỒI QUY DỰ ĐOÁN RATING

### 2.2.1. Hồi quy tuyến tính và Ridge Regression

Hồi quy tuyến tính ước lượng rating bằng tổng có trọng số của các đặc trưng đầu vào. Với mẫu n, vector đặc trưng xₙ và rating rₙ, dự đoán có dạng r̂ₙ = β₀ + xₙᵀβ. Nghiệm bình phương tối thiểu chọn hệ số làm nhỏ tổng bình phương sai số. Ridge bổ sung một khoản phạt L2 để hạn chế hệ số quá lớn khi đặc trưng có tương quan hoặc số chiều cao:

**r̂ₙ = β₀ + xₙᵀβ;  J(β) = Σₙ(rₙ − r̂ₙ)² + α‖β‖².**

Trong đề tài, các đặc trưng có thể gồm thống kê chỉ tính từ Train, mức độ hoạt động của người dùng/phim, thể loại phim và sở thích thể loại của người dùng. Mã ứng dụng chính có Ridge với tham số alpha; bộ chạy Experiment Studio có hồi quy tuyến tính dạng OLS theo batch. Khi ghi bảng kết quả, cần gọi đúng biến thể đã chạy: OLS và Ridge là hai cấu hình khác nhau [4].

Ưu điểm của mô hình tuyến tính là thời gian huấn luyện và suy luận thường thấp, hệ số có thể xem xét để hiểu ảnh hưởng của đặc trưng. Mô hình không tự biểu diễn mọi tương tác phi tuyến; năng lực dự đoán phụ thuộc vào cách xây đặc trưng và mức độ phù hợp của quan hệ tuyến tính.

### 2.2.2. XGBoost Regressor

XGBoost xây một tập cây hồi quy theo cơ chế boosting. Mỗi cây mới bổ sung một phần hiệu chỉnh vào dự đoán hiện có nhằm giảm hàm mất mát. Mục tiêu gồm sai số dự đoán và khoản điều chuẩn độ phức tạp của cây:

**J = Σₙ ℓ(rₙ, r̂ₙ) + Σₜ Ω(fₜ),  r̂ₙ = Σₜ fₜ(xₙ).**

Độ sâu cây, số cây và learning rate ảnh hưởng đến năng lực mô hình cũng như chi phí tính toán. Cây sâu có thể nắm bắt tương tác phức tạp nhưng cũng dễ khớp quá mức nếu dữ liệu hoặc điều chuẩn không phù hợp. XGBoost có các cơ chế xử lý đặc trưng thưa và tối ưu bộ nhớ, song vẫn cần dùng cùng tập dữ liệu, cách chia và quy trình chọn cấu hình với các mô hình đối chiếu [5].

### 2.2.3. SVD Collaborative Filtering

SVD-CF trong bài toán này là mô hình nhân tố tiềm ẩn cho rating tường minh, thường được gọi là Funk SVD. Với người dùng u và phim i, mô hình biểu diễn sở thích bằng vector pᵤ và đặc điểm phim bằng vector qᵢ. Dự đoán thường thêm trung bình chung μ cùng độ lệch riêng của người dùng và phim:

**r̂ᵤᵢ = μ + bᵤ + bᵢ + pᵤᵀqᵢ.**

Các nhân tố và bias được học bằng cách giảm sai số trên những cặp (u,i) thuộc Ω, đồng thời phạt độ lớn của tham số để giảm overfitting. SGD cập nhật vector theo sai số của từng rating đã quan sát. Tăng số nhân tố có thể biểu diễn nhiều kiểu sở thích hơn nhưng làm tăng số tham số và nhu cầu dữ liệu cho mỗi người dùng/phim [3].

Tên gọi SVD trong nhóm dự đoán rating không có nghĩa là phân rã ma trận user–item đã điền giá trị 0 bằng phép phân rã đại số đầy đủ. Đây là một mô hình học các nhân tố từ những phần tử quan sát. Trường hợp user hoặc phim chưa từng xuất hiện cần được xử lý bằng bias, trung bình hoặc cơ chế dự phòng đã mô tả.

### 2.2.4. Neural Collaborative Filtering

Neural Collaborative Filtering (NCF) thay phép tương tác tuyến tính đơn giản giữa vector user và item bằng một mạng nơ-ron có thể học quan hệ phi tuyến. Mỗi người dùng và phim được ánh xạ thành embedding; mô hình kết hợp các embedding rồi đưa qua MLP để tạo điểm dự đoán [6].

Trong phiên bản của dự án, mạng ghép embedding người dùng, embedding phim và tích từng phần tử của hai vector. Đầu ra MLP dự đoán phần dư sau bias toàn cục, bias user và bias item. Mô hình được tối ưu bằng MSE trên rating quan sát; AdamW, dropout và early stopping được dùng để kiểm soát học. Đường cong Train/Validation theo epoch giúp quan sát khi sai số Train tiếp tục giảm nhưng sai số Validation không còn cải thiện.

Bài báo NCF ban đầu trình bày bài toán lọc cộng tác với phản hồi ẩn. Cài đặt của đồ án điều chỉnh mục tiêu sang rating tường minh, do đó cần mô tả rõ hàm mất mát và cách chia dữ liệu của phiên bản đang chạy, không suy rộng kết quả của bài báo sang dữ liệu này. NCF là một phương án cần so sánh; số tham số và thời gian huấn luyện cao hơn không bảo đảm RMSE thấp hơn.

## 2.3. NHÓM PHÂN LOẠI MỨC ĐỘ YÊU THÍCH

### 2.3.1. Logistic Regression

Bài toán phân loại chuyển rating thành hai nhãn: y=1 nếu rating ≥3,5 (“Thích”) và y=0 nếu rating <3,5 (“Không thích”). Logistic Regression tính một tổ hợp tuyến tính z=β₀+xᵀβ rồi dùng hàm sigmoid để ước lượng xác suất thuộc lớp Thích:

**P(y=1|x) = 1/(1+e⁻ᶻ).**

Ngưỡng 0,5 là một lựa chọn mặc định, không phải quy tắc bắt buộc. Nếu mục tiêu ưu tiên F1 của lớp Thích, ngưỡng có thể được chọn trên Validation rồi khóa trước khi đánh giá Test. Hệ số mô hình diễn giải được qua log-odds; tuy vậy, ranh giới quyết định là tuyến tính theo không gian đặc trưng [7].

### 2.3.2. Random Forest Classifier

Random Forest là tổ hợp nhiều cây quyết định. Mỗi cây được huấn luyện trên mẫu bootstrap; tại các nút, thuật toán xem xét một tập con đặc trưng ngẫu nhiên. Dự đoán phân lớp được tổng hợp từ các cây, còn xác suất có thể lấy từ tỷ lệ phiếu của lớp trong rừng. Việc tạo nhiều cây với mẫu và tập đặc trưng khác nhau giúp giảm phụ thuộc vào một cây đơn lẻ [8].

Số cây, độ sâu và số mẫu tối thiểu tại lá điều khiển độ phức tạp. Rừng sâu có thể học tương tác phi tuyến nhưng tốn bộ nhớ và suy luận hơn mô hình tuyến tính. Với nhãn lệch, Accuracy có thể cao dù mô hình bỏ sót lớp thiểu số; vì vậy cần xem F1, Precision, Recall, AUC-ROC, ma trận nhầm lẫn và tỷ lệ nhãn của từng tập.

## 2.4. NHÓM PHÂN CỤM SỞ THÍCH NGƯỜI DÙNG

### 2.4.1. Biểu diễn user–genre và K-Means

Để phân cụm người dùng, mỗi user được biểu diễn bằng vector sở thích theo thể loại, tính từ dữ liệu Train. Một thành phần có thể là rating trung bình hoặc mức rating tương đối của user đối với thể loại; số lượt xem/đánh giá theo genre có thể được dùng làm đặc trưng riêng. Việc tách mức chấm chung khỏi sở thích thể loại giúp tránh nhầm người thường chấm cao với người thật sự có thiên hướng về một genre.

K-Means tìm K tâm cụm và gán mỗi điểm vào tâm gần nhất nhằm giảm tổng bình phương khoảng cách trong cụm:

**J = Σₖ Σₓ∈Cₖ ‖x − μₖ‖².**

Thuật toán cần chọn K trước và nhạy với khởi tạo cũng như thang đo đặc trưng. Vì vậy, cần chuẩn hóa đầu vào trên Train, thử nhiều K với seed cố định, rồi báo kích thước từng cụm và độ ổn định khi đổi seed [9].

### 2.4.2. DBSCAN

DBSCAN xác định cụm dựa trên mật độ lân cận. Hai tham số chính là eps, bán kính vùng lân cận, và min_samples, số điểm tối thiểu để một vùng được xem là đủ dày. Điểm core có đủ số hàng xóm trong eps; điểm biên nằm gần điểm core nhưng không đủ hàng xóm; điểm còn lại được xem là nhiễu (noise). DBSCAN không yêu cầu khai báo trước số cụm và có thể tìm cụm hình dạng không lồi [10].

Chọn eps quá nhỏ làm nhiều user thành noise; eps quá lớn có thể nhập các nhóm khác nhau vào một cụm. Silhouette hoặc Davies–Bouldin cần được đọc cùng tỷ lệ noise và độ bao phủ, vì chỉ số trên phần điểm được gán có thể trông tốt khi phần lớn user bị loại. DBSCAN chuẩn không có phương thức predict phổ quát; nếu hệ thống cần gán user mới, phải ghi rõ quy tắc mở rộng, chẳng hạn gán theo điểm core lân cận, và đo riêng thời gian của quy tắc đó.

## 2.5. NHÓM KHAI PHÁ LUẬT KẾT HỢP APRIORI

Mỗi người dùng được xem là một transaction gồm các phim họ đã đánh giá tích cực. Trong cấu hình của dự án, rating từ 4 sao trở lên có thể được dùng làm tiêu chí phim được thích. Một luật X→Y cho biết các transaction có tập phim X thường đồng thời chứa tập phim Y; đây là quan hệ đồng xuất hiện, không chứng minh X gây ra Y.

Với tập transaction D, support của tập item X là tỷ lệ transaction chứa X. Confidence của luật X→Y là tỷ lệ transaction chứa Y trong số transaction đã chứa X. Lift so sánh confidence với tần suất nền của Y:

**support(X) = count(X)/|D|.**

**confidence(X→Y) = support(X∪Y)/support(X).**

**lift(X→Y) = confidence(X→Y)/support(Y).**

Lift lớn hơn 1 cho thấy Y xuất hiện cùng X thường xuyên hơn mức kỳ vọng nếu hai tập độc lập; lift không phải xác suất dự đoán hay độ chính xác. Apriori khai thác tính chất: nếu một itemset là phổ biến thì mọi tập con của nó cũng phải phổ biến. Thuộc tính này cho phép loại sớm ứng viên chứa tập con không đạt min_support [11].

## 2.6. NHÓM GIẢM CHIỀU DỮ LIỆU

### 2.6.1. Principal Component Analysis

PCA biến đổi các đặc trưng đã được center sang hệ trục trực giao mới. Các thành phần chính được sắp theo lượng phương sai chúng giải thích; giữ k thành phần đầu tạo biểu diễn ngắn hơn. Tỷ lệ phương sai giải thích cộng dồn đến k thành phần là tổng eigenvalue của k thành phần chia tổng eigenvalue. PCA có thể hỗ trợ trực quan hóa, giảm nhiễu hoặc giảm số chiều, nhưng giữ phương sai cao không tự động đồng nghĩa dự đoán rating tốt hơn [12].

Với X đã center và ma trận thành phần Vₖ, biểu diễn giảm chiều là Z=XVₖ; phép tái tạo xấp xỉ là X̂=ZVₖᵀ. Reconstruction error đo mức sai khác giữa X và X̂ trên dữ liệu chưa dùng để fit.

### 2.6.2. TruncatedSVD và phân biệt với SVD-CF

TruncatedSVD chiếu dữ liệu lên một số vector kỳ dị đầu mà không bắt buộc center ma trận đầu vào. Tính chất này phù hợp hơn với ma trận thưa, nhưng khi áp dụng trên vector user–genre, cần diễn giải explained variance theo cách tính thực tế của thuật toán và không đồng nhất với độ chính xác của bộ gợi ý.

TruncatedSVD và SVD-CF giải hai mục tiêu khác nhau. TruncatedSVD nén biểu diễn đặc trưng để đo phương sai và khả năng tái tạo; SVD-CF học latent factors trực tiếp từ rating người dùng–phim đã quan sát để dự đoán điểm. Kết quả của hai phương pháp phải nằm ở các bảng và metric tương ứng, không gộp dưới một nhãn “SVD” chung.

## 2.7. HỌC TĂNG CƯỜNG CHO GỢI Ý TUẦN TỰ

### 2.7.1. MDP và Q-Learning

Một bài toán học tăng cường được mô tả bằng trạng thái s, hành động a, phần thưởng r và xác suất chuyển trạng thái. Chính sách π chọn hành động theo trạng thái để tối đa hóa tổng phần thưởng chiết khấu. Q-Learning học giá trị hành động Q(s,a) từ trải nghiệm:

**Q(s,a) ← Q(s,a) + α[r + γ maxₐ′Q(s′,a′) − Q(s,a)].**

Ở đây α là learning rate, γ là hệ số chiết khấu. Epsilon-greedy chọn hành động ngẫu nhiên với xác suất ε để thăm dò, còn lại khai thác giá trị Q cao nhất. Trong gợi ý tuần tự, trạng thái có thể gồm hồ sơ thể loại, các phim đã chọn và tín hiệu phản hồi gần nhất; hành động là chọn một phim ứng viên; reward được tính từ phản hồi của user. Watkins và Dayan chứng minh điều kiện hội tụ cho Q-Learning dạng bảng trong thiết lập rời rạc, với điều kiện các cặp trạng thái–hành động được thăm đủ [13].

### 2.7.2. DQN và giới hạn đánh giá offline

Deep Q-Network thay bảng Q bằng mạng nơ-ron để xấp xỉ giá trị hành động khi không gian trạng thái lớn. Mạng mục tiêu, replay buffer và lấy mẫu minibatch giúp ổn định huấn luyện trong các thiết lập DQN kinh điển [14]. Đây là phần mở rộng so với Q-Learning dạng bảng, không nên gọi hai mô hình là một triển khai nếu cấu trúc và mã chạy khác nhau.

Đánh giá từ lịch sử rating là offline: dữ liệu chỉ ghi nhận một số phim mà user đã xem hoặc chấm. Với phim chưa có phản hồi, không thể suy ra user sẽ không thích phim. Vì vậy, báo cáo reward cần kèm tỷ lệ action có phản hồi quan sát được và so sánh trên cùng candidate pool, cohort, seed và số bước. Reward mô phỏng không thay cho thử nghiệm trực tuyến hay đánh giá phản thực không thiên lệch nếu không có xác suất hiển thị (propensity).

# CHƯƠNG 3: DỮ LIỆU VÀ THIẾT KẾ THỰC NGHIỆM

## 3.1. MÔ TẢ BỘ DỮ LIỆU

### 3.1.1. Nguồn dữ liệu và cấu trúc tệp

Đề tài sử dụng MovieLens 25M do GroupLens Research thuộc University of Minnesota công bố. Bộ dữ liệu được phát hành tháng 12/2019 và gồm dữ liệu rating cùng metadata phim, nhãn cộng đồng và liên kết ngoài [1], [2]. Trong repository hiện có bản CSV đã làm sạch tại thư mục dataset_cleaned; bản rating gốc không bị ghi đè.

**Bảng 3.1: Các tệp trong MovieLens 25M và vai trò trong đồ án**

| Tệp | Trường chính | Vai trò |
|---|---|---|
| ratings.csv | userId, movieId, rating, timestamp | Huấn luyện, validation, test và xây lịch sử tương tác |
| movies.csv | movieId, title, genres | Danh mục, đặc trưng thể loại và hiển thị kết quả |
| tags.csv | userId, movieId, tag, timestamp | Nhãn cộng đồng; không dùng làm đầu vào bốn mô hình rating chính |
| links.csv | movieId, imdbId, tmdbId | Đối chiếu phim với IMDb/TMDB và liên kết ngoài |

### 3.1.2. Quy mô và đặc điểm dữ liệu

Thống kê toàn bộ CSV trong workspace cho thấy dữ liệu có hơn 25 triệu rating, rất ít so với toàn bộ cặp có thể có giữa user và phim. Thang điểm rời rạc từ 0,5 đến 5 sao với bước 0,5. Dataset không cung cấp nhân khẩu học, ảnh poster, nội dung phim, diễn viên hoặc đạo diễn; các trường này không được tự suy diễn để đưa vào mô hình.

**Bảng 3.2: Thống kê MovieLens 25M trong bản dữ liệu hiện có**

| Chỉ số | Giá trị |
|---|---:|
| Số rating | 25.000.095 |
| Số user có rating | 162.541 |
| Số phim trong movies.csv | 62.423 |
| Số phim có ít nhất một rating | 59.047 |
| Số phim chưa có rating | 3.376 |
| Số tag | 1.093.360 |
| Số thể loại đã xác định | 19 |
| Rating trung bình | 3,53385/5 |
| Mật độ ma trận user–item | 0,2464% |
| Khoảng thời gian rating | 09/01/1995–21/11/2019 |

### 3.1.3. Làm sạch, giá trị thiếu và biểu diễn thưa

Bản dataset_cleaned kiểm tra schema, khóa ID, rating hợp lệ, tham chiếu movieId và các trường văn bản. Báo cáo kiểm tra dữ liệu đi kèm ghi lại thay đổi và dấu vân tay file. Các cặp user–movie không có rating được giữ ở trạng thái chưa biết; không điền bằng 0. Việc thiếu tương tác là đặc tính của dữ liệu thưa, không phải một giá trị rating cần chuẩn hóa.

Mã dự án không tạo ma trận dense có kích thước 162.541×62.423. Các mô hình rating học trên những dòng (userId, movieId, rating) đã quan sát; các thao tác cần sparse structure dùng biểu diễn thưa hoặc xử lý theo batch. Riêng phân cụm và giảm chiều dùng hồ sơ user–genre có số chiều nhỏ hơn, được tính từ phần Train.

Rating gốc được giữ trên thang 0,5–5. Một số mô hình chuẩn hóa hoặc điều chuẩn đặc trưng bên trong pipeline, nhưng repository không thực hiện chuẩn hóa toàn cục nhãn rating rồi khẳng định dữ liệu gốc đã được chuẩn hóa. Nếu một thí nghiệm chuẩn hóa target, phép biến đổi chỉ được fit trên Train và phải hoàn nguyên dự đoán trước khi tính RMSE/MAE theo đơn vị sao.

### 3.1.4. Lọc tương tác ít và chia dữ liệu theo thời gian

Giao thức chính của Experiment Studio chia lịch sử theo thời gian trong từng user, xấp xỉ 70% Train, 15% Validation và 15% Test. Các rating có timestamp bằng nhau không bị tách qua hai tập. User cần có đủ lịch sử để tạo ba đoạn hợp lệ; cấu hình chuẩn đặt tối thiểu 10 rating và các trường hợp bị loại được ghi trong provenance. Chuẩn bị dữ liệu hiện có được lưu trong Parquet và manifest ghi số dòng mỗi tập [19].

**Bảng 3.3: Số dòng của bản chia dữ liệu đang có trong workspace**

| Tập | Số rating | Vai trò |
|---|---:|---|
| Train | 17.500.066 | Fit mô hình, đặc trưng, scaler và reducer |
| Validation | 3.750.014 | Chọn cấu hình, ngưỡng phân loại và checkpoint NCF/RL |
| Test | 3.750.015 | Đánh giá cuối sau khi đã khóa lựa chọn |

Các con số trên là manifest của lần chuẩn bị hiện có, không phải kết quả dự đoán. Thứ tự thời gian được xác định trong lịch sử mỗi user; đây không phải một phép chia ngẫu nhiên từng dòng và cũng không tương đương với việc mô phỏng một mốc tương lai toàn cục duy nhất.

## 3.2. XÁC ĐỊNH BÀI TOÁN CỦA TỪNG NHÓM

**Bảng 3.4: Đầu vào, đầu ra và mục tiêu đánh giá theo nhóm thuật toán**

| Nhóm | Đầu vào | Đầu ra | Bài toán |
|---|---|---|---|
| Hồi quy | user, phim và đặc trưng chỉ tính từ Train | Rating liên tục 0,5–5 | Dự đoán rating cho cặp user–phim chưa quan sát |
| Phân loại | Đặc trưng của cặp user–phim | Xác suất/nhãn Thích hoặc Không thích | Phân lớp rating theo ngưỡng 3,5 sao |
| Phân cụm | Hồ sơ sở thích và tần suất user theo genre | ID cụm hoặc noise | Tìm nhóm user có hồ sơ thể loại gần nhau |
| Luật kết hợp | Transaction user gồm phim được thích | Luật X→Y | Tìm phim thường được thích cùng nhau |
| Giảm chiều | Vector user–genre đã scale | Tọa độ k chiều và vector tái tạo | Đo mức nén, phương sai giữ lại và sai số tái tạo |
| Học tăng cường | Trạng thái lịch sử, danh sách ứng viên và phản hồi | Chuỗi hành động gợi ý | Tối đa hóa reward tuần tự trong môi trường mô phỏng |

Các nhóm trả lời những câu hỏi khác nhau. RMSE không thể xếp hạng một phân cụm hay luật kết hợp; Silhouette không đánh giá chất lượng dự đoán rating. Kết quả phải được trình bày theo mục tiêu riêng, sau đó mới đối chiếu chi phí hoặc khả năng tích hợp vào demo [16].

## 3.3. CẤU HÌNH VÀ CÁCH ÁP DỤNG CÁC THUẬT TOÁN

### 3.3.1. Cấu hình chung và kiểm soát rò rỉ

Seed mặc định là 42. Dữ liệu được đọc theo chunk khi chuẩn bị; các tập phục vụ mô hình được lưu thành Parquet. Mọi thống kê user/item, mã hóa, scaler và reducer cần fit riêng trong Train hoặc fold tương ứng rồi mới biến đổi phần giữ lại. Khi đánh giá Test, mô hình đã được chọn bằng Validation; không quay lại thay đổi cấu hình theo điểm Test.

Trong mô hình hồi quy dạng đặc trưng, mã dự án tính thống kê và hồ sơ sở thích từ dữ liệu fit. Tại thời điểm tạo hàng huấn luyện, đóng góp rating của chính hàng đó được loại khỏi một số thống kê/hồ sơ để tránh nhãn rò rỉ vào đặc trưng. Với NCF, rating chưa quan sát không đưa vào loss như nhãn âm.

### 3.3.2. Hồi quy và NCF

Ứng dụng chính dùng Ridge alpha=10, XGBoost 160 cây/depth 6/learning rate 0,05, SVD-CF 32 nhân tố/15 epochs/learning rate 0,005/regularization 0,02 và NCF 16 nhân tố/hidden size 32/tối đa 12 epochs/batch 2.048/learning rate 0,003/dropout 0,1/regularization 0,01/patience 3. Experiment Studio có bộ giá trị khởi đầu riêng: Linear Regression OLS, XGBoost 150 cây/depth 6/learning rate 0,05, SVD 32 nhân tố và NCF 32 nhân tố/hidden size 32/learning rate 0,003/regularization 0,02/dropout 0,1/batch 2.048/tối đa 15 epochs. Không gộp hai cấu hình này; mỗi bảng kết quả phải lấy cấu hình đã lưu cùng run.

Script report_temporal_experiments.py định nghĩa một không gian tìm kiếm riêng: XGBoost 100 cây với depth 3 hoặc 6; SVD 8 hoặc 16 factors; Logistic Regression C=0,1 hoặc 1; Random Forest 100 cây với depth 6 hoặc 12 và min_samples_leaf=3. OLS là mốc cố định, không có tham số cần tìm. Optuna TPE chạy 8 trial NCF; factors ∈{8,16,32}, hidden size ∈{16,32}, learning rate 0,001–0,01 theo log, regularization 0,0001–0,02 theo log, dropout ∈{0,1; 0,2}, batch size 1.024. Đây là cấu hình của script riêng, không phải cấu hình mặc định của Experiment Studio.

### 3.3.3. Phân loại

Nhãn dương là rating ≥3,5. Logistic Regression và Random Forest dùng cùng đặc trưng, cùng fold thời gian và cùng tập Validation/Test. Experiment Studio bắt đầu Logistic với C=1/max_iter=600 và Random Forest với 100 cây/depth 8/min_samples_leaf=5. Script báo cáo tìm C và độ sâu theo lưới đã nêu ở mục 3.3.2. Tỷ lệ nhãn của từng tập được ghi lại; ngưỡng phân lớp được chọn trên Validation nếu mục tiêu ưu tiên F1 lớp Thích. AUC-ROC được tính từ xác suất, không tính từ nhãn đã ngưỡng hóa.

### 3.3.4. Phân cụm

Hồ sơ user–genre chỉ dùng rating Train. K-Means thử K thuộc {2, 3, 4, 6, 8, 10}. Experiment Studio cho DBSCAN dải eps khởi đầu {1,0; 1,5; 2,0} và min_samples {5, 10}; chế độ tự động có thể lấy eps từ các phân vị khoảng cách láng giềng. Bộ chọn yêu cầu tối thiểu hai cụm hợp lệ và mặc định độ bao phủ ít nhất 80%. Khi so cụm, ghi số cụm, kích thước cụm, tỷ lệ noise và độ bao phủ; Silhouette/DBI chỉ được so trong cùng cách chuẩn hóa và cùng tập được chấm. Nếu đo gán user mới cho DBSCAN, phải ghi rõ quy tắc dựa trên core neighbor và thời gian của bước gán.

### 3.3.5. Apriori

Mỗi user Train tạo một transaction chứa phim rating từ 4 sao trở lên. Transaction rỗng vẫn thuộc mẫu số support. Cấu hình khởi đầu của Experiment Studio giới hạn 80 phim, min_support=0,08, min_confidence=0,30 và độ dài luật tối đa 2; có thể thay đổi trước khi chạy. Các tham số thực tế, số luật sinh được và thời gian khai phá phải được lưu. Bảng kết quả xếp luật theo Lift, Confidence và Support; các luật được tính trên tập Train và được chấm lại trên tập giữ lại bằng cùng danh sách luật.

### 3.3.6. PCA và TruncatedSVD

PCA được fit trên đặc trưng đã center; TruncatedSVD không center ma trận đầu vào. Experiment Studio mặc định thử k={2,4,8}; script báo cáo mở rộng danh sách đến 12 và 16. Scaler và reducer chỉ được fit trên Train/fold. Báo explained variance trên Train và reconstruction MSE trên Validation/Test. Nếu đánh giá giá trị ứng dụng, dùng vector đã tái tạo làm đầu vào cho một tác vụ rating cố định và so RMSE/MAE với Mean Rating hoặc Item-KNN trên đúng split.

Trong Experiment Studio hiện tại, PCA/TruncatedSVD còn là reducer dùng chung để tạo biến thể đầu vào cho một số nhóm. Khi dùng chế độ so sánh, biến thể raw và reduced cần giữ cùng seed, dữ liệu, cohort và ngân sách. Bảng chẩn đoán giảm chiều độc lập không được gọi là bảng RMSE của SVD-CF.

### 3.3.7. Q-Learning và môi trường gợi ý

Môi trường Gymnasium nhận lịch sử Train và catalog gồm 30 phim phổ biến nhất trên Train. Trạng thái có user, tập phim đã chọn, genre của hành động trước và tín hiệu thích/không thích gần nhất. Hành động chọn một phim chưa được chọn trong catalog. Với script Q-Learning, reward quan sát là (rating−2,75)/2,25, nằm trong [-1,1]; feedback thiếu nhận 0 theo quy ước và cờ observed=false để không xem đó là dislike. Gymnasium cung cấp API reset/step chuẩn để định nghĩa môi trường và vòng tương tác [18].

Script report_temporal_experiments.py dùng Q-Learning dạng bảng với alpha=0,2, gamma=0,9, epsilon=0,2, 5.000 episode, horizon 5; đánh giá mỗi policy trên 100 episode. Các policy Random và Popularity dùng cùng catalog, cohort, episode và giới hạn số hành động. Báo cáo cumulative reward cùng average reward/action, tỷ lệ feedback quan sát và độ bao phủ; không dùng riêng cumulative reward nếu số action thực tế khác nhau. Experiment Studio mới dùng DQN với cấu hình riêng và không được ghi nhãn thành Q-Learning dạng bảng.

### 3.3.8. Ghi log, tái lập và nền tảng demo

Mỗi run cần lưu seed, dấu vân tay dữ liệu/cấu hình, phiên bản thư viện, candidate đã thử, cấu hình được chọn, metric từng fold, thời gian fit/suy luận và kích thước model. Repository hiện xuất kết quả dạng JSON, CSV và Excel trong lịch sử run; chưa thấy cấu hình tích hợp MLflow. Nếu nhóm chọn MLflow làm nơi ghi log chính, cần bổ sung và kiểm tra connector trước khi ghi đó là chức năng đã triển khai.

Web demo hiện dùng backend FastAPI và frontend React/TypeScript. Người dùng có thể ghi nhận rating, xem lịch sử và yêu cầu gợi ý Top-N. Mô hình dùng để triển khai cuối sẽ được chọn theo bảng so sánh ở Chương 4; khi chưa có mô hình đã huấn luyện, hệ thống có nhãn cho gợi ý phổ biến/nội dung thay vì trình bày điểm dự đoán giả. Đây là demo chạy cục bộ, không phải triển khai dịch vụ công khai.

## 3.4. PHƯƠNG PHÁP ĐÁNH GIÁ

### 3.4.1. Hồi quy rating

Với n rating thật rⱼ và dự đoán r̂ⱼ, RMSE và MAE được tính như sau:

**RMSE = √[(1/n)Σⱼ(rⱼ−r̂ⱼ)²];  MAE = (1/n)Σⱼ|rⱼ−r̂ⱼ|.**

RMSE phạt sai số lớn mạnh hơn do bình phương sai số; MAE biểu diễn độ lệch tuyệt đối trung bình theo đơn vị sao. Báo cáo cả hai, kèm baseline Mean Rating và Item-KNN nếu chúng được chạy trên cùng split. Với k-fold, tính mean và sample standard deviation của từng metric; đồng thời báo Validation và Test cuối.

### 3.4.2. Phân loại

Accuracy là tỷ lệ dự đoán đúng; Precision đo tỷ lệ dự đoán Thích thực sự là Thích; Recall đo tỷ lệ user–phim Thích được tìm thấy; F1 là trung bình điều hòa của Precision và Recall. AUC-ROC đo khả năng xếp hạng mẫu dương cao hơn mẫu âm qua các ngưỡng. Báo thêm confusion matrix [[TN, FP], [FN, TP]], ngưỡng đã khóa và tỷ lệ hai lớp. Khi một tập chỉ có một lớp, AUC không xác định và cần ghi N/A thay vì tự gán giá trị [17].

### 3.4.3. Phân cụm và luật kết hợp

Silhouette gần 1 thể hiện điểm gần cụm được gán và xa cụm khác; giá trị quanh 0 cho thấy vùng chồng lấn, giá trị âm có thể chỉ ra gán cụm không phù hợp. Davies–Bouldin so sánh độ phân tán trong cụm với độ tách biệt giữa cụm; giá trị thấp hơn thường thuận lợi hơn. Cả hai phụ thuộc vào biểu diễn, metric khoảng cách, số cụm và cách xử lý noise.

Support đo độ phổ biến của itemset trong tất cả transaction; Confidence đo tỷ lệ consequent trong transaction có antecedent; Lift so với tần suất nền của consequent. Khi so luật trên tập Validation/Test, mẫu số transaction phải được định nghĩa nhất quán và kết quả cần kèm độ phủ antecedent.

### 3.4.4. Giảm chiều

Explained variance ratio được fit trên Train; reconstruction error được tính trên phần giữ lại. Báo số thành phần, tỷ lệ phương sai giữ lại, MSE tái tạo và thời gian fit/transform. Nếu khẳng định giảm chiều giúp dự đoán rating, cần trình bày thêm RMSE/MAE downstream trên cùng split với baseline Mean Rating và Item-KNN. Giữ được nhiều phương sai không thay thế cho phép đo hiệu quả tác vụ.

### 3.4.5. Học tăng cường và chi phí tính toán

Q-Learning/DQN được so với Random và Popularity trên cùng điều kiện. Báo cumulative reward, average reward trên mỗi action thực tế, tỷ lệ action có phản hồi, số user đủ điều kiện và learning curve theo episode hoặc checkpoint. Độ phủ thấp phải được nêu vì reward được tính trên lịch sử quan sát, không phải phản hồi phản thực.

Training time đo thời gian fit; inference latency đo thời gian dự đoán theo batch cố định và báo cả đơn vị trên mỗi rating khi phù hợp. Ghi thiết bị, số lõi, RAM, phiên bản phần mềm, kích thước model và trạng thái cache. Không so thời gian của hai run đo trên máy hoặc ngân sách khác nhau như một benchmark tuyệt đối.

### 3.4.6. Cross-validation và lựa chọn mô hình

Thiết kế của đề tài yêu cầu 5 fold theo thời gian trên phần Train. Các timestamp bằng nhau được giữ cùng fold; mỗi fold chỉ dùng quá khứ để fit và chấm đoạn thời gian kế tiếp. Script báo cáo hiện có GridSearchCV với splitter thời gian; RandomizedSearchCV chỉ nên ghi nếu nhóm bổ sung và lưu cấu hình chạy. NCF dùng Optuna. Sau lựa chọn, refit trên Train+Validation, khóa mô hình rồi mới đánh giá Test. Mean ± std của fold là chỉ báo độ ổn định, không thay thế metric Test cuối [16], [17].

### 3.4.7. Đối chiếu giao thức với trạng thái mã nguồn

Repository có nhiều đường chạy phục vụ các mục tiêu khác nhau. Cần chọn đúng một protocol cho mỗi bảng và không trộn các run khác schema hoặc cách chia khác nhau.

**Bảng 3.5: Các điểm cần phân biệt khi báo cáo trạng thái triển khai**

| Nội dung | Trạng thái thấy trong repository | Cách ghi trong báo cáo |
|---|---|---|
| Dữ liệu và chia tập | Artifact chuẩn bị hiện có dùng dữ liệu đầy đủ, split thời gian trong từng user 70/15/15 | Ghi đúng tên protocol và manifest của run dùng để sinh kết quả |
| Số fold | Experiment Studio dùng 5 fold thời gian trong từng user; report_temporal_experiments.py định nghĩa 5 fold toàn cục | Chạy lại Studio để tạo kết quả 5 fold; không gán k=5 cho các run lịch sử đã chạy 3 fold |
| Tuning | Report runner có GridSearchCV/Optuna; Studio dùng candidate sweep riêng | Chỉ ghi thư viện tuning tương ứng với run thực tế; chưa thấy RandomizedSearchCV/MLflow được nối trong giao diện hiện tại |
| Cold-start | Bộ chuẩn bị chính yêu cầu user có tối thiểu 10 rating; report runner riêng dùng ngưỡng user/item 5/3 | Nêu ngưỡng và phạm vi fit ở từng protocol; không trộn hai ngưỡng |
| Học tăng cường | Report runner có Q-Learning dạng bảng; Experiment Studio hiện tại có DQN và baseline | Ghi Q-Learning nếu chạy bảng Q; ghi DQN nếu chạy mạng xấp xỉ |
| SVD/PCA | Studio dùng reducer PCA/TruncatedSVD dùng chung; report runner có chẩn đoán giảm chiều riêng | Phân biệt reducer với SVD-CF dự đoán rating |
| Kết quả lịch sử | Tài liệu v3 ghi mẫu 150 user, 24.319 rating và chia IID; protocol mới dùng thời gian | Không chép số liệu v3 vào bảng temporal của Chương 4 |

Mã report runner đặt giới hạn mẫu 300 user và dùng chia toàn cục theo timestamp, còn artifact chuẩn bị chính trong workspace dùng per-user temporal 70/15/15. Thư mục report-temporal của run được mô tả trong tài liệu không có trong bản workspace hiện tại. Vì vậy, trước khi điền Chương 4, nhóm cần chạy và giữ artifact của đúng protocol đã chọn. Đây là điều kiện để đối chiếu công bằng, không phải kết quả đã hoàn tất.

# CHƯƠNG 4: THỰC NGHIỆM VÀ SO SÁNH

Chương này được thiết kế để nhóm bổ sung sau khi chạy protocol đã thống nhất ở Chương 3. Các ô ghi “điền sau” là chỗ chờ số liệu từ artifact; báo cáo hiện không đưa vào kết quả dự đoán hay số đo tự tạo.

## 4.1. NHÓM THUẬT TOÁN HỒI QUY

### 4.1.1. Kết quả dự đoán rating

Điền mean ± std trên 5 fold và metric holdout theo cùng đơn vị sao. Nếu một mô hình không chạy được, ghi nguyên nhân; không thay bằng số của protocol khác.

**Bảng 4.1: Chất lượng dự đoán rating của các mô hình**

| Mô hình | RMSE CV (mean ± std) | MAE CV (mean ± std) | RMSE Validation | RMSE Test | MAE Test |
|---|---:|---:|---:|---:|---:|
| Linear Regression/OLS | điền sau | điền sau | điền sau | điền sau | điền sau |
| Ridge Regression (nếu chạy) | điền sau | điền sau | điền sau | điền sau | điền sau |
| XGBoost Regressor | điền sau | điền sau | điền sau | điền sau | điền sau |
| SVD Collaborative Filtering | điền sau | điền sau | điền sau | điền sau | điền sau |
| Neural Collaborative Filtering | điền sau | điền sau | điền sau | điền sau | điền sau |
| Mean Rating baseline | — | — | điền sau | điền sau | điền sau |
| Item-KNN baseline | — | — | điền sau | điền sau | điền sau |

### 4.1.2. Thời gian và learning curve NCF

Đo thời gian fit cuối trên Train+Validation, latency trên một batch Test cố định và kích thước artifact. Tách thời gian tuning/CV khỏi thời gian fit cuối. Chèn biểu đồ Train/Validation loss của NCF theo epoch; trục, đơn vị loss và epoch dừng cần ghi rõ.

**Bảng 4.2: Chi phí tính toán của các mô hình rating**

| Mô hình | Training time (s) | Inference (µs/rating) | Kích thước model (KiB) | Thiết bị |
|---|---:|---:|---:|---|
| Linear Regression/OLS | điền sau | điền sau | điền sau | điền sau |
| XGBoost Regressor | điền sau | điền sau | điền sau | điền sau |
| SVD Collaborative Filtering | điền sau | điền sau | điền sau | điền sau |
| Neural Collaborative Filtering | điền sau | điền sau | điền sau | điền sau |

Vị trí chèn biểu đồ gồm Train loss và Validation loss của NCF theo epoch.

**Hình 4.1: Learning curve của Neural Collaborative Filtering**

### 4.1.3. Nhận xét nhóm hồi quy

Nhận xét sau khi có số liệu: mô hình có RMSE/MAE thấp nhất; độ lệch giữa Validation, Test và các fold; ảnh hưởng của cold-start; chi phí huấn luyện/suy luận; và mô hình đáp ứng yêu cầu triển khai. Kết luận chọn mô hình phải dựa trên Validation/CV và ngân sách, không chọn lại theo Test.

## 4.2. NHÓM THUẬT TOÁN PHÂN LOẠI

Báo cáo tỷ lệ nhãn Thích/Không thích trên từng split và nêu lớp được dùng để tính F1. AUC-ROC cần tính từ xác suất; ghi ngưỡng được chọn trên Validation.

**Bảng 4.3: So sánh Logistic Regression và Random Forest**

| Mô hình | Accuracy CV (mean ± std) | F1 Thích CV (mean ± std) | Accuracy Test | F1 Test | AUC-ROC Test |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | điền sau | điền sau | điền sau | điền sau | điền sau |
| Random Forest Classifier | điền sau | điền sau | điền sau | điền sau | điền sau |

**Bảng 4.4: Ngưỡng và chi phí phân loại**

| Mô hình | Ngưỡng | Precision Test | Recall Test | Fit (s) | Inference (µs/rating) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | điền sau | điền sau | điền sau | điền sau | điền sau |
| Random Forest Classifier | điền sau | điền sau | điền sau | điền sau | điền sau |

**Bảng 4.5: Ma trận nhầm lẫn trên Test**

| Mô hình | Ma trận [[TN,FP],[FN,TP]] | Nhận xét lỗi chính |
|---|---|---|
| Logistic Regression | điền sau | điền sau |
| Random Forest Classifier | điền sau | điền sau |

Nhận xét sau khi có số liệu: mức mất cân bằng nhãn, lớp nào bị bỏ sót, và liệu chênh lệch Accuracy có đi cùng F1/AUC hay không.

## 4.3. NHÓM THUẬT TOÁN PHÂN CỤM

Ghi biểu diễn user–genre, scaler, số cụm, tỷ lệ noise và tiêu chí chọn tham số. DBSCAN phải nêu rõ cách gán cụm cho user mới; nếu chỉ tính chỉ số nội bộ trên điểm được gán, cần báo kèm coverage.

**Bảng 4.6: Chỉ số nội bộ của K-Means và DBSCAN**

| Thuật toán | Tham số được chọn | Split | Silhouette | Davies–Bouldin | Số cụm |
|---|---|---|---:|---:|---:|
| K-Means | điền sau | Validation | điền sau | điền sau | điền sau |
| K-Means | điền sau | Test | điền sau | điền sau | điền sau |
| DBSCAN | điền sau | Validation | điền sau | điền sau | điền sau |
| DBSCAN | điền sau | Test | điền sau | điền sau | điền sau |

**Bảng 4.7: Noise, độ bao phủ và chất lượng gợi ý theo cụm**

| Thuật toán | Split | Noise (%) | Coverage (%) | Recall@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| K-Means | Validation | — | điền sau | điền sau | điền sau |
| K-Means | Test | — | điền sau | điền sau | điền sau |
| DBSCAN | Validation | điền sau | điền sau | điền sau | điền sau |
| DBSCAN | Test | điền sau | điền sau | điền sau | điền sau |

**Bảng 4.8: Chi phí fit và gán user mới**

| Thuật toán | Fit (s) | Gán user mới (ms/user) | Quy tắc fallback |
|---|---:|---:|---|
| K-Means | điền sau | điền sau | điền sau |
| DBSCAN | điền sau | điền sau | điền sau |

## 4.4. NHÓM KHAI PHÁ LUẬT KẾT HỢP

Ghi rõ ngưỡng rating phim được xem là thích, số transaction, min_support, min_confidence và giới hạn itemset. Nếu không có luật đạt ngưỡng, báo không có luật; không chèn ví dụ minh họa vào bảng kết quả.

**Bảng 4.9: Cấu hình Apriori**

| Transactions Train | Ngưỡng rating thích | min_support | min_confidence | Độ dài itemset tối đa |
|---:|---:|---:|---:|---:|
| điền sau | điền sau | điền sau | điền sau | điền sau |

**Bảng 4.10: Quy mô khai phá luật**

| Số luật đạt ngưỡng | Thời gian khai phá (s) | Truy vấn luật (ms/user) | Độ phủ antecedent Validation | Độ phủ antecedent Test |
|---:|---:|---:|---:|---:|
| điền sau | điền sau | điền sau/N/A | điền sau | điền sau |

**Bảng 4.11: Các luật kết hợp được chọn để phân tích**

| STT | Luật X→Y | Support | Confidence | Lift | Độ phủ vế trái |
|---:|---|---:|---:|---:|---:|
| 1 | điền sau | điền sau | điền sau | điền sau | điền sau |
| 2 | điền sau | điền sau | điền sau | điền sau | điền sau |
| 3 | điền sau | điền sau | điền sau | điền sau | điền sau |
| 4 | điền sau | điền sau | điền sau | điền sau | điền sau |
| 5 | điền sau | điền sau | điền sau | điền sau | điền sau |

Nhận xét từng luật theo Lift và Support; không diễn giải luật đồng xuất hiện thành quan hệ nhân quả. Nếu dùng luật để gợi ý, đo thêm tỷ lệ user có antecedent và candidate coverage.

## 4.5. NHÓM GIẢM CHIỀU DỮ LIỆU

Tách kết quả PCA khỏi TruncatedSVD. Báo phương sai trên Train và reconstruction error trên Validation/Test; sau đó, nếu có đánh giá downstream, so cùng baseline trên cùng dữ liệu.

**Bảng 4.12: Explained variance và reconstruction error**

| Phương pháp | Số thành phần | Explained variance Train (%) | Reconstruction MSE Validation | Reconstruction MSE Test | Fit (s) | Transform (ms/user) |
|---|---:|---:|---:|---:|---:|---:|
| PCA | điền sau | điền sau | điền sau | điền sau | điền sau | điền sau |
| TruncatedSVD | điền sau | điền sau | điền sau | điền sau | điền sau | điền sau |

**Bảng 4.13: Đánh giá downstream rating sau giảm chiều**

| Biểu diễn/mô hình | RMSE Validation | RMSE Test | MAE Test | Latency | So với baseline |
|---|---:|---:|---:|---:|---|
| Mean Rating | điền sau | điền sau | điền sau | điền sau | Baseline |
| Item-KNN | điền sau | điền sau | điền sau | điền sau | Baseline |
| PCA + mô hình rating đã khóa | điền sau | điền sau | điền sau | điền sau | điền sau |
| TruncatedSVD + mô hình rating đã khóa | điền sau | điền sau | điền sau | điền sau | điền sau |

Diễn giải liệu số chiều giảm có làm tăng tốc hoặc giảm kích thước model hay không, và phần cải thiện/giảm sút metric tác vụ có đủ để biện minh chi phí chuyển đổi không.

## 4.6. NHÓM HỌC TĂNG CƯỜNG

Bảng phải ghi một cách tường minh reward, số episode, horizon, catalog hành động, cách xử lý feedback thiếu và checkpoint được chọn bằng Validation. So sánh chính dùng cùng episode và candidate pool.

**Bảng 4.14: Reward và độ phủ phản hồi trên Test**

| Policy | Cumulative reward | Average reward/action | Feedback quan sát (%) | User coverage (%) | Train (s) | Latency (ms/action) |
|---|---:|---:|---:|---:|---:|---:|
| Q-Learning hoặc DQN (ghi đúng mô hình đã chạy) | điền sau | điền sau | điền sau | điền sau | điền sau | điền sau |
| Random Recommendation | điền sau | điền sau | điền sau | điền sau | — | điền sau |
| Popularity Recommendation | điền sau | điền sau | điền sau | điền sau | — | điền sau |

Vị trí chèn biểu đồ reward theo episode hoặc checkpoint; cần thể hiện cùng đường baseline nếu có.

**Hình 4.2: Reward trong quá trình huấn luyện và kiểm tra policy**

Nhận xét kết quả cùng coverage và giới hạn offline. Reward cao nhưng ít phản hồi quan sát không đủ để khẳng định mô hình sẽ cải thiện trải nghiệm người dùng thật.

## 4.7. TỔNG HỢP SO SÁNH MÔ HÌNH DỰ ĐOÁN RATING

Đây là bảng quyết định chính cho mô hình rating được dùng trong demo. Điền kết quả từ cùng run và cùng protocol; mục tiêu là cân bằng sai số, độ ổn định qua fold, thời gian và bộ nhớ.

**Bảng 4.15: Chất lượng Linear, XGBoost, SVD-CF và NCF**

| Mô hình | RMSE Test | MAE Test | RMSE CV (mean ± std) |
|---|---:|---:|---:|
| Linear Regression/OLS | điền sau | điền sau | điền sau |
| XGBoost | điền sau | điền sau | điền sau |
| SVD-CF | điền sau | điền sau | điền sau |
| NCF | điền sau | điền sau | điền sau |

**Bảng 4.16: Chi phí và nhận xét triển khai mô hình rating**

| Mô hình | Fit (s) | Latency (µs/rating) | Model (KiB) | Nhận xét triển khai |
|---|---:|---:|---:|---|
| Linear Regression/OLS | điền sau | điền sau | điền sau | điền sau |
| XGBoost | điền sau | điền sau | điền sau | điền sau |
| SVD-CF | điền sau | điền sau | điền sau | điền sau |
| NCF | điền sau | điền sau | điền sau | điền sau |

### 4.7.1. Lựa chọn mô hình cho web demo

Mô hình được chọn: điền sau. Căn cứ: RMSE/MAE Validation, mean ± std cross-validation, khả năng xử lý user/phim mới, latency, thời gian fit và kích thước model. Ghi rõ đánh đổi được chấp nhận và phương án fallback.

### 4.7.2. Tích hợp và kiểm tra demo

Demo sử dụng FastAPI và React/TypeScript. Người dùng chọn hoặc chấm các phim đã xem; backend lập hồ sơ, loại phim đã có trong lịch sử khỏi tập ứng viên rồi trả về Top-N. Ghi mô hình được nạp, thời gian phản hồi, số gợi ý, hành vi khi profile trống và cách hệ thống xử lý user/phim ngoài tập Train. Phần này chỉ được hoàn tất sau khi mô hình được chọn từ Bảng 4.12.

# TÀI LIỆU THAM KHẢO

[1] GroupLens Research, “MovieLens 25M Dataset,” University of Minnesota, phát hành tháng 12/2019. https://grouplens.org/datasets/movielens/25m/.

[2] F. M. Harper and J. A. Konstan, “The MovieLens Datasets: History and Context,” ACM Transactions on Interactive Intelligent Systems, vol. 5, no. 4, Article 19, 2015. DOI: 10.1145/2827872.

[3] Y. Koren, R. Bell, and C. Volinsky, “Matrix Factorization Techniques for Recommender Systems,” Computer, vol. 42, no. 8, pp. 30–37, 2009. DOI: 10.1109/MC.2009.263.

[4] A. E. Hoerl and R. W. Kennard, “Ridge Regression: Biased Estimation for Nonorthogonal Problems,” Technometrics, vol. 12, no. 1, pp. 55–67, 1970. DOI: 10.1080/00401706.1970.10488634.

[5] T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785–794, 2016. DOI: 10.1145/2939672.2939785.

[6] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, “Neural Collaborative Filtering,” in Proceedings of the 26th International Conference on World Wide Web, pp. 173–182, 2017. DOI: 10.1145/3038912.3052569.

[7] D. R. Cox, “The Regression Analysis of Binary Sequences,” Journal of the Royal Statistical Society: Series B (Methodological), vol. 20, no. 2, pp. 215–232, 1958. DOI: 10.1111/j.2517-6161.1958.tb00292.x.

[8] L. Breiman, “Random Forests,” Machine Learning, vol. 45, pp. 5–32, 2001. DOI: 10.1023/A:1010933404324.

[9] S. P. Lloyd, “Least Squares Quantization in PCM,” IEEE Transactions on Information Theory, vol. 28, no. 2, pp. 129–137, 1982. DOI: 10.1109/TIT.1982.1056489.

[10] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, “A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise,” in Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, pp. 226–231, 1996.

[11] R. Agrawal and R. Srikant, “Fast Algorithms for Mining Association Rules in Large Databases,” in Proceedings of the 20th International Conference on Very Large Data Bases, pp. 487–499, 1994.

[12] H. Hotelling, “Analysis of a Complex of Statistical Variables into Principal Components,” Journal of Educational Psychology, vol. 24, no. 6, pp. 417–441, 1933. DOI: 10.1037/h0071325.

[13] C. J. C. H. Watkins and P. Dayan, “Q-Learning,” Machine Learning, vol. 8, pp. 279–292, 1992. DOI: 10.1023/A:1022676722315.

[14] V. Mnih et al., “Human-Level Control through Deep Reinforcement Learning,” Nature, vol. 518, pp. 529–533, 2015. DOI: 10.1038/nature14236.

[15] Y. Koren, “Collaborative Filtering with Temporal Dynamics,” in Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 447–456, 2009. DOI: 10.1145/1557019.1557072.

[16] A. Gunawardana and G. Shani, “A Survey of Accuracy Evaluation Metrics of Recommendation Tasks,” Journal of Machine Learning Research, vol. 10, pp. 2935–2962, 2009.

[17] Scikit-learn Developers, “Metrics and scoring: quantifying the quality of predictions,” documentation. https://scikit-learn.org/stable/modules/model_evaluation.html.

[18] Gymnasium Documentation, “Gymnasium: An API standard for reinforcement learning environments,” Farama Foundation. https://gymnasium.farama.org/.

[19] Nhóm 05, “MoviePredictionSystem: mã nguồn, cấu hình và tài liệu thực nghiệm,” repository nội bộ của đồ án, phiên bản được rà soát ngày 20/11/2026.
