"""Complete the requested Vietnamese report without changing training code or saved runs.

Reuses the locally supplied report builder for its evidence tables and figures,
and replaces Chapter 2 with an experiment-independent theoretical treatment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

import build_student_report as base


THEORY = r"""
## 2.1. NHÓM THUẬT TOÁN HỒI QUY
Hồi quy học ánh xạ từ đầu vào x sang đầu ra số thực y. Với dự đoán đánh giá, đầu ra là mức độ yêu thích trên thang điểm, còn đầu vào có thể là đặc trưng mô tả hoặc chỉ số người dùng–đối tượng. Cần phân biệt tối ưu sai số điểm với tối ưu thứ tự gợi ý: hai mô hình có RMSE tương đương vẫn có thể tạo danh sách Top-N khác nhau. Phần này trình bày mô hình và nguyên tắc đánh giá chung, chưa xét cấu hình hay kết quả của hệ thống.
### 2.1.1. Linear Regression
**Lý thuyết.** Hồi quy tuyến tính giả thiết kỳ vọng có điều kiện của đầu ra được biểu diễn bằng tổng có trọng số của các đặc trưng. Từ “tuyến tính” chỉ tính tuyến tính theo hệ số; vẫn có thể đưa x² hoặc tích của hai biến vào vector đặc trưng. Ordinary Least Squares (OLS) tìm hệ số làm nhỏ nhất tổng bình phương phần dư. Tài liệu LinearRegression của scikit-learn xác định đây là bài toán bình phương tối thiểu, có tùy chọn ước lượng hệ số chặn [6], [18].
= ŷ = β₀ + Σⱼ βⱼxⱼ;     J(β) = Σᵢ(yᵢ − ŷᵢ)².
**Cách hoạt động.** Sau khi xác định vector đầu vào, thuật toán giải bài toán tối ưu để ước lượng β. Với ma trận đủ hạng, nghiệm lý thuyết có thể viết qua phương trình chuẩn; trong tính toán thường dùng phân rã số hoặc bộ giải bình phương tối thiểu để tránh nghịch đảo trực tiếp. Khi các cột phụ thuộc tuyến tính, có thể tồn tại nhiều bộ hệ số cho cùng dự đoán, vì vậy cần xem xét hạng ma trận và độ ổn định số.
**Các tham số quan trọng.** fit_intercept quyết định có học β₀; cách chuẩn hóa và cách tạo biến tương tác ảnh hưởng trực tiếp đến mô hình. OLS cơ bản không có learning rate hoặc số epoch bắt buộc. Nếu thêm phạt λ‖β‖² thì bài toán trở thành Ridge; đây là biến thể khác OLS. Khi đọc kết quả cần xác định đúng hàm mục tiêu thay vì gọi mọi mô hình tuyến tính là Ridge [6].
**Đánh giá hiệu quả.** RMSE và MAE đo sai số trên dữ liệu giữ lại, còn R² so mức giảm tổng bình phương sai số với dự đoán trung bình. Kiểm tra phần dư theo mức dự đoán giúp nhận ra xu hướng cong hoặc độ phân tán thay đổi. Các giả định độc lập, phương sai không đổi và phân phối sai số đặc biệt quan trọng khi suy luận khoảng tin cậy cho hệ số; chất lượng dự đoán vẫn phải kiểm chứng ngoài mẫu.
**Ưu điểm, nhược điểm và ứng dụng.** Mỗi lần dự đoán chỉ cần tích vô hướng nên mô hình phù hợp làm mốc so sánh và phục vụ nhanh. Hệ số dễ đọc khi thang đo và mã hóa rõ ràng, nhưng không tự mang ý nghĩa nhân quả. Trong hệ gợi ý, mô hình có thể kết hợp mức chấm điểm của người dùng, chất lượng trung bình của đối tượng và nội dung. Nếu thiếu đặc trưng tương tác, hai người có sở thích khác nhau vẫn dễ nhận dự đoán gần giống nhau.
### 2.1.2. XGBoost Regressor
**Lý thuyết.** XGBoost là hệ thống gradient boosting cây: dự đoán được xây bằng cách cộng dần các cây hồi quy. Chen và Guestrin mô tả hàm mục tiêu gồm tổn thất dự đoán và phạt độ phức tạp cây, đồng thời sử dụng thông tin đạo hàm để lựa chọn phép chia [5]. Nhờ cấu trúc cây, mô hình có thể học quan hệ phi tuyến và tương tác mà không cần khai báo trước toàn bộ tích đặc trưng.
= ŷᵢ⁽ᵗ⁾ = ŷᵢ⁽ᵗ⁻¹⁾ + ηfₜ(xᵢ);     Obj = Σᵢℓ(yᵢ, ŷᵢ) + ΣₜΩ(fₜ).
**Cách hoạt động.** Ở vòng t, mô hình tính gradient và Hessian của loss theo dự đoán hiện tại, xây cây cải thiện mục tiêu rồi cộng đầu ra cây với hệ số η. Khác với Random Forest huấn luyện các cây tương đối độc lập, cây boosting phụ thuộc vào các vòng trước. Với squared error, gradient phản ánh hướng điều chỉnh phần dư; việc chia lá nhằm gom các mẫu có nhu cầu điều chỉnh gần nhau.
**Các tham số quan trọng.** n_estimators và learning_rate tạo sự đánh đổi giữa số vòng với độ lớn cập nhật. max_depth, min_child_weight và gamma hạn chế phép chia; subsample và colsample_bytree tạo lấy mẫu; reg_alpha và reg_lambda phạt trọng số lá. objective phải phù hợp bài toán hồi quy. tree_method, số luồng và thiết bị ảnh hưởng chi phí thực thi, cần ghi cùng cấu hình đánh giá [19].
**Đánh giá hiệu quả.** Dùng RMSE/MAE ngoài mẫu và theo dõi chênh lệch Train–Validation khi tăng số cây. Early stopping sử dụng tập kiểm tra nội bộ để chọn số vòng, còn Test cần giữ riêng. Đánh giá đồng thời thời gian, bộ nhớ và kích thước mô hình vì một cải thiện sai số nhỏ có thể đòi hỏi nhiều cây hơn. Feature importance mô tả cách mô hình sử dụng biến, không chứng minh biến gây ra thay đổi đầu ra.
**Ưu điểm, nhược điểm và ứng dụng.** Cây boosting phù hợp dữ liệu bảng có nhiều thống kê và quan hệ tương tác. Ngưỡng chia ít phụ thuộc vào đổi thang tuyến tính của từng biến, nhưng vẫn phụ thuộc biểu diễn đầu vào và dữ liệu thiếu. Mô hình không tự hiểu ý nghĩa của mã user hoặc movie dạng số. Với gợi ý cá nhân hóa, nếu chỉ có ID mà thiếu biểu diễn hữu ích, ưu thế phi tuyến không bảo đảm vượt mô hình nhân tố.
### 2.1.3. SVD Collaborative Filtering
**Lý thuyết.** Mô hình nhân tố tiềm ẩn biểu diễn người dùng u bằng pᵤ và đối tượng i bằng qᵢ trong không gian k chiều. Theo Koren, Bell và Volinsky, dự đoán có thể kết hợp tích vô hướng với trung bình toàn cục và độ lệch riêng của người dùng/đối tượng [3]. Trong ngữ cảnh recommender, tên SVD thường dùng cho mô hình học từ rating quan sát; không nên đồng nhất với phân rã đại số trên ma trận đã điền đủ ô.
= r̂ᵤᵢ = μ + bᵤ + bᵢ + pᵤᵀqᵢ.
= J = Σ₍ᵤ,ᵢ₎∈Ω(rᵤᵢ − r̂ᵤᵢ)² + λ(‖P‖²F + ‖Q‖²F + ‖b‖²).
**Cách hoạt động.** Khởi tạo factor và bias, duyệt các cặp đã quan sát, tính sai số rồi cập nhật tham số bằng SGD hoặc tối ưu luân phiên. Tập Ω chỉ chứa ô có phản hồi. Đưa toàn bộ ô trống vào loss như rating bằng 0 sẽ thay đổi giả thiết học: mô hình bị buộc giải thích rất nhiều phản hồi tiêu cực chưa từng được quan sát. Đây là khác biệt nền tảng giữa học rating tường minh và phân rã ma trận số thông thường.
**Các tham số quan trọng.** Số nhân tố k quyết định dung lượng biểu diễn và bộ nhớ xấp xỉ O((U+I)k). Learning rate, số lượt quét và hệ số regularization kiểm soát tối ưu, mức khớp và tổng quát hóa. k nhỏ có thể bỏ sót khác biệt sở thích; k quá lớn với ít tương tác làm tăng nguy cơ ghi nhớ. Cần xét số quan sát của từng user/item thay vì chỉ tổng số rating.
**Đánh giá hiệu quả.** Đánh giá RMSE/MAE trên cặp giữ lại, kèm độ phủ với ID đã gặp/chưa gặp. Nếu sử dụng factor để xếp hạng, cần thêm Recall@K hoặc NDCG@K theo cùng tập ứng viên. Sai số trên user đã có lịch sử không đại diện cho người dùng hoàn toàn mới. So sánh với trung bình hoặc bias baseline giúp kiểm tra phần tương tác tiềm ẩn đóng góp bao nhiêu.
**Ưu điểm, nhược điểm và ứng dụng.** Mô hình phù hợp ma trận thưa và dự đoán mỗi cặp với O(k) phép tính. Dù các chiều tiềm ẩn không nhất thiết tương ứng thể loại có tên, chúng có thể gom những mẫu đồng sở thích. Hạn chế chính là cold-start, thiên lệch do dữ liệu chỉ gồm lựa chọn đã quan sát và khả năng khó diễn giải từng chiều. Có thể bổ sung metadata hoặc fallback để phục vụ ID chưa được học.
### 2.1.4. Neural Collaborative Filtering
**Lý thuyết.** NCF thay hàm tương tác cố định bằng một mạng học từ embedding user và item. He và cộng sự đề xuất khung NCF cho phản hồi ngầm, trong đó MLP mô hình hóa tương tác phi tuyến [4]. Cần phân biệt bài toán nhị phân của công trình gốc với biến thể dự đoán số sao: biến thể hồi quy cần đầu ra và loss phù hợp dữ liệu tường minh.
= pᵤ = Embedding(u); qᵢ = Embedding(i); r̂ᵤᵢ = gθ(pᵤ, qᵢ).
**Cách hoạt động.** Một minibatch chứa cặp user–item được ánh xạ sang embedding, ghép hoặc kết hợp bằng tích từng phần tử, truyền qua các lớp ẩn và tạo đầu ra. Lan truyền ngược cập nhật đồng thời bảng embedding và trọng số mạng. Với rating, có thể dùng MSE; với implicit feedback, lựa chọn nhãn âm và hàm loss là một phần của thiết kế, không thể tự động xem mọi ô thiếu là phản hồi âm.
**Các tham số quan trọng.** Kích thước embedding và số lớp/độ rộng MLP quyết định dung lượng; batch size, learning rate và optimizer ảnh hưởng đường học. Dropout, weight decay và early stopping là các cơ chế kiểm soát khớp quá mức. Khi thay số lớp, cần giữ ngân sách tính toán và cách đánh giá rõ ràng để tránh so độ sâu mạng đồng thời với nhiều thay đổi khác.
**Đánh giá hiệu quả.** Báo RMSE/MAE ngoài mẫu cho hồi quy, đồng thời vẽ Train loss và Validation loss cùng đại lượng. Nếu Train tiếp tục giảm nhưng Validation tăng qua nhiều epoch, đó là dấu hiệu cần kiểm tra overfitting; phải nêu tập dùng tính đường Train và khôi phục checkpoint tốt nhất. Đánh giá nhiều seed và thời gian fit giúp xác định cải thiện có ổn định, có đáng với chi phí hay không.
**Ưu điểm, nhược điểm và ứng dụng.** NCF cho phép học hàm tương tác mềm dẻo và mở rộng với nhiều nguồn đặc trưng. Mạng lớn hơn không tự bảo đảm chất lượng tốt hơn: ít lịch sử cho mỗi ID, tối ưu khó và thay đổi phân phối có thể làm kết quả kém ổn định. Trong phục vụ, embedding của người dùng mới phải được xây từ hồ sơ hoặc thay bằng fallback. Cần phân biệt chi phí một cặp với chi phí chấm điểm toàn bộ danh mục.
## 2.2. NHÓM THUẬT TOÁN PHÂN LOẠI
### 2.2.1. Logistic Regression
**Lý thuyết.** Logistic Regression mô hình hóa log-odds của lớp dương là một hàm tuyến tính. Sigmoid chuyển điểm số thành giá trị trong (0,1); vì vậy mô hình trả xác suất trước khi tạo nhãn. Hàm mất mát là negative log-likelihood Bernoulli, thường kết hợp regularization [6], [20]. Tên gọi “Regression” có nguồn gốc từ cách mô hình hóa nhưng nhiệm vụ ở đây là phân loại.
= p(y=1|x) = 1/(1 + exp(−β₀ − βᵀx)); log[p/(1−p)] = β₀ + βᵀx.
= LogLoss = −(1/n)Σᵢ[yᵢlog(pᵢ) + (1−yᵢ)log(1−pᵢ)].
**Cách hoạt động.** Sau khi mã hóa và chuẩn hóa đầu vào khi cần, bộ giải tối ưu log-loss để học hệ số. Ở bước suy luận, nhãn dương được gán nếu p≥τ. Thay τ không huấn luyện lại hệ số nhưng thay Precision/Recall. Tăng Recall bằng hạ ngưỡng thường kéo theo thêm false positive; vì vậy lựa chọn ngưỡng cần gắn với chi phí bỏ sót và chi phí gợi ý sai.
**Các tham số quan trọng.** C là nghịch đảo cường độ regularization trong scikit-learn; C nhỏ thường phạt mạnh hơn. Solver và loại penalty phải tương thích. max_iter/tol kiểm soát dừng tối ưu; class_weight thay trọng số đóng góp của từng lớp. Thay trọng số hoặc lấy mẫu lại có thể làm thay đổi ý nghĩa xác suất, do đó cần kiểm tra calibration khi xác suất được dùng cho ra quyết định [20].
**Đánh giá hiệu quả.** Accuracy, Precision, Recall, F1 và AUC-ROC cần được đọc cùng nhau. AUC đo khả năng sắp thứ tự điểm xác suất trên nhiều ngưỡng; F1 phụ thuộc ngưỡng và lớp được coi là dương. Khi dữ liệu lệch lớp, một mô hình luôn đoán lớp đa số vẫn có thể đạt Accuracy cao. Cần so với baseline này và đọc ma trận nhầm lẫn, thay vì kết luận chỉ từ một tỷ lệ đúng.
**Ưu điểm, nhược điểm và ứng dụng.** Mô hình gọn, suy luận nhanh và diễn giải hệ số theo log-odds nếu các biến được mã hóa rõ. Ranh giới tuyến tính có thể chưa mô tả tốt các kết hợp sở thích phức tạp; cần biến tương tác hoặc mô hình khác. Trong gợi ý, Logistic phù hợp dự đoán xác suất vượt ngưỡng yêu thích, còn nó không trả lời trực tiếp người dùng sẽ chấm bao nhiêu sao.
### 2.2.2. Random Forest
**Lý thuyết.** Random Forest kết hợp nhiều cây quyết định và đưa ngẫu nhiên vào dữ liệu lẫn tập đặc trưng xét tại mỗi nút. Breiman xây dựng nền tảng của phương pháp dựa trên tập hợp các bộ dự đoán cây [7]. Với phân loại, xác suất lớp có thể được tổng hợp từ xác suất của từng cây; trung bình hóa giúp giảm biến thiên so với một cây đơn [21].
= p̂(y=1|x) = (1/B)Σᵦ₌₁ᴮ p̂ᵦ(y=1|x).
**Cách hoạt động.** Mỗi cây được xây trên mẫu bootstrap nếu bật lấy mẫu hoàn lại. Tại mỗi nút chỉ xét một tập con biến, chọn phép chia cải thiện độ thuần nhất rồi tiếp tục đến điều kiện dừng. Các cây có thể huấn luyện song song. Lấy ngẫu nhiên feature giúp những biến mạnh không khiến mọi cây trở thành các bản sao quá giống nhau.
**Các tham số quan trọng.** n_estimators ảnh hưởng độ ổn định và chi phí tổng hợp; max_depth, min_samples_split và min_samples_leaf giới hạn kích thước cây. max_features điều khiển số biến xét khi chia; bootstrap và max_samples quyết định phạm vi mẫu mỗi cây. class_weight điều chỉnh lớp; n_jobs quyết định mức song song. Tăng số cây không sửa được một cách xây feature hoặc chia dữ liệu bị rò rỉ [21].
**Đánh giá hiệu quả.** Sử dụng các chỉ số phân loại ngoài mẫu và đo cả dung lượng lẫn latency. Out-of-bag hữu ích trong bối cảnh mẫu độc lập nhưng không thay thế phép chia thời gian khi mục tiêu là dự đoán tương lai. Độ quan trọng theo giảm impurity có thể ưu ái biến nhiều giá trị; permutation importance ngoài mẫu là cách kiểm tra bổ sung và vẫn cần đọc trong bối cảnh các biến tương quan.
**Ưu điểm, nhược điểm và ứng dụng.** Rừng học được phi tuyến, ngưỡng và tương tác với ít yêu cầu chuẩn hóa. Đổi lại, nhiều cây sâu có thể tốn RAM và chi phí dự đoán, khó giải thích hơn Logistic. Trong phân loại mức độ yêu thích, mô hình phù hợp khi thống kê lịch sử và metadata có quan hệ phi tuyến; hiệu quả phụ thuộc việc các thống kê có được tạo từ đúng phần quá khứ hay không.
## 2.3. NHÓM THUẬT TOÁN PHÂN CỤM
Phân cụm tổ chức các đối tượng không có nhãn thành những nhóm theo biểu diễn và độ đo tương tự đã chọn. Cụm không phải “nhãn đúng” tự nhiên của người dùng: đổi từ rating tuyệt đối sang rating tương đối hoặc tần suất có thể tạo phân hoạch khác. Vì vậy, cần xác định câu hỏi mô tả trước khi tối ưu chỉ số hình học.
### 2.3.1. K-Means
**Lý thuyết.** K-Means tìm K tâm và phân hoạch làm nhỏ nhất tổng khoảng cách Euclid bình phương trong cụm. Thuật toán Lloyd luân phiên bước gán điểm và cập nhật tâm [8]. Nghiệm tìm được phụ thuộc khởi tạo và thường là tối ưu cục bộ; K là lựa chọn mô hình, không được thuật toán tự xác định.
= J = Σₖ₌₁ᴷ Σₓ∈Cₖ ‖x − μₖ‖².
**Cách hoạt động.** Khởi tạo tâm, gán mỗi điểm về tâm gần nhất, tính lại tâm bằng trung bình các điểm rồi lặp đến điều kiện dừng. K-means++ hỗ trợ chọn tâm ban đầu trải rộng hơn; nhiều lần khởi tạo giúp giảm phụ thuộc một nghiệm. Dự đoán cho điểm mới là tìm tâm gần nhất sau cùng phép biến đổi đầu vào [22].
**Các tham số quan trọng.** n_clusters chọn độ chi tiết; init và n_init ảnh hưởng khởi tạo; max_iter và tol ảnh hưởng dừng; random_state hỗ trợ tái lập. Chuẩn hóa cần gắn với ý nghĩa biến: nếu một chiều có độ lớn cao hơn nhiều, nó có thể chi phối khoảng cách. Với hồ sơ thể loại, còn cần cân nhắc mức độ thiếu thông tin của người có ít tương tác.
**Đánh giá hiệu quả.** Inertia giảm khi tăng K nên không thể chỉ chọn K bằng giá trị nhỏ nhất. Silhouette, Davies–Bouldin, kích thước cụm, ổn định qua khởi tạo và khả năng diễn giải nên được xem cùng nhau. Một nghiệm gom người dùng theo “chấm điểm dễ hay khó” có thể tách hình học rõ mà chưa phản ánh sở thích nội dung; cần đối chiếu hồ sơ trung tâm của từng cụm.
**Ưu điểm, nhược điểm và ứng dụng.** K-Means đơn giản, dễ gán người dùng mới và có chi phí gần O(nKdT). Phương pháp phù hợp phân đoạn hồ sơ có dạng tương đối lồi, nhưng nhạy với ngoại lệ và cụm khác mật độ/kích thước. Trong hệ gợi ý, có thể dùng cụm để sinh ứng viên hoặc mô tả nhóm sở thích; cần kiểm chứng ranking riêng trước khi xem đó là cải thiện gợi ý.
### 2.3.2. DBSCAN
**Lý thuyết.** DBSCAN định nghĩa cụm bằng kết nối mật độ, phân biệt điểm lõi, điểm biên và nhiễu. Điểm lõi có ít nhất min_samples điểm trong lân cận bán kính ε theo quy ước đếm của implementation. Ester và cộng sự đề xuất phương pháp để nhận ra cụm có hình dạng tùy ý và xử lý nhiễu [9].
= Nε(x) = {z : d(x,z) ≤ ε};     điểm lõi nếu |Nε(x)| ≥ min_samples.
**Cách hoạt động.** Bắt đầu từ điểm chưa xét, kiểm tra lân cận; nếu đủ mật độ thì mở rộng cụm qua các điểm lõi liên thông. Điểm không đủ mật độ có thể trở thành biên khi nằm gần lõi, hoặc giữ nhãn noise. Một chuỗi cầu nối đủ mật độ có thể nhập nhiều vùng thành một cụm; vì vậy “không cần cho trước số cụm” không đồng nghĩa luôn tách được các nhóm có ích.
**Các tham số quan trọng.** eps và min_samples phải được lựa chọn đồng thời với metric và chuẩn hóa. ε quá nhỏ thường tạo nhiều noise; quá lớn có thể nối gần như toàn bộ dữ liệu. leaf_size và thuật toán tìm láng giềng ảnh hưởng tốc độ, còn chiều cao làm khoảng cách khó phân biệt. Tài liệu scikit-learn lưu ý một số cấu hình DBSCAN có chi phí bộ nhớ bậc hai trong trường hợp xấu [23].
**Đánh giá hiệu quả.** Tính Silhouette/DBI trên các điểm được gán cụm và công bố đồng thời số cụm, tỷ lệ noise và cỡ mẫu. Nếu chỉ có một cụm hợp lệ, các chỉ số yêu cầu từ hai cụm không xác định; không gán giá trị 0 để biểu diễn lỗi này. Một cấu hình có chỉ số cao trên phần rất nhỏ dữ liệu cần bị xem xét về độ phủ trước khi chọn.
**Ưu điểm, nhược điểm và ứng dụng.** DBSCAN hữu ích khi cần giữ nhiễu riêng và tìm nhóm không hình cầu. Nhược điểm là khó dùng một ε cho nhiều mật độ và không có predict chuẩn như K-Means. Nếu ứng dụng gán điểm mới theo lõi gần nhất trong ε, phải gọi đó là quy tắc mở rộng, nêu trường hợp không có lõi phù hợp và đo thời gian đúng quy tắc đó [23].
## 2.4. KHAI PHÁ LUẬT KẾT HỢP
### 2.4.1. Apriori
**Lý thuyết.** Luật X→Y mô tả sự đồng xuất hiện giữa hai tập mục không giao nhau trong cơ sở giao dịch. Apriori khai thác tính chất phản đơn điệu của support: nếu một tập không phổ biến, mọi tập chứa nó cũng không thể đạt ngưỡng. Agrawal và Srikant dùng nguyên lý này để cắt ứng viên trước khi đếm trên dữ liệu [10].
**Cách hoạt động.** Trước hết đếm tập một mục để tạo L₁. Từ Lₖ₋₁, nối các tập phù hợp thành ứng viên Cₖ, loại ứng viên có tập con không phổ biến, đếm support rồi giữ Lₖ. Sau đó tách các itemset phổ biến thành tiền đề và kết luận để sinh luật đạt ngưỡng confidence. Quy trình kết thúc khi không còn ứng viên; không đòi hỏi thứ tự thời gian giữa các mục trong cùng giao dịch.
**Các tham số quan trọng.** min_support quyết định mức phổ biến tối thiểu; min_confidence lọc độ tin cậy có điều kiện; max_len giới hạn chiều dài itemset. Ngưỡng xác định mục “được yêu thích”, giới hạn danh mục và cách giữ/bỏ giao dịch rỗng làm thay đổi bài toán trước khi Apriori chạy. Khi so hai cấu hình, cần giữ rõ mẫu số support và phạm vi danh mục.
= support(X→Y) = |{T : X∪Y⊆T}|/N.
= confidence(X→Y) = support(X∪Y)/support(X).
= lift(X→Y) = confidence(X→Y)/support(Y).
**Đánh giá hiệu quả.** Support cho biết độ phủ, confidence cho biết tỷ lệ Y xuất hiện khi có X, còn lift điều chỉnh theo mức phổ biến nền của Y. Minh họa do báo cáo xây dựng: trong 1.000 giao dịch, X xuất hiện 200 lần, Y 400 lần, cả hai 160 lần; support=0,16, confidence=0,80 và lift=2. Nếu Y đã xuất hiện 800 lần thì cùng confidence=0,80 chỉ cho lift=1. Ví dụ cho thấy confidence cao có thể đến từ kết luận vốn phổ biến [24].
**Ưu điểm, nhược điểm và ứng dụng.** Luật dễ diễn giải và phù hợp gợi ý “người thích tập phim này thường thích tập phim kia”. Nhưng số ứng viên tăng nhanh khi ngưỡng thấp hoặc giao dịch dài; luật còn thiên về phim phổ biến. Lift lớn trên rất ít quan sát dễ thiếu ổn định. Muốn dùng làm recommender cần đo trên giai đoạn tương lai, loại phim đã xem và đánh giá precision/recall/coverage của danh sách, thay vì chỉ báo các luật đẹp trên tập khai phá.
## 2.5. NHÓM GIẢM CHIỀU DỮ LIỆU
### 2.5.1. Principal Component Analysis
**Lý thuyết.** PCA tìm các trục trực giao có phương sai giảm dần trên dữ liệu đã trừ trung bình. Giữ k trục đầu tạo phép xấp xỉ hạng thấp, tối ưu sai số tái tạo bình phương trong lớp phép chiếu tuyến tính trực giao. Jolliffe và Cadima trình bày PCA như một công cụ mô tả và giảm chiều, đồng thời nhấn mạnh vai trò của thang đo và diễn giải thành phần [11].
= Xc = X − 1μᵀ;     Z = XcVₖ;     X̂ = ZVₖᵀ + 1μᵀ.
**Cách hoạt động.** Ước lượng trung bình trên tập fit, center dữ liệu, tính các thành phần bằng SVD hoặc bài toán trị riêng, rồi chiếu mọi tập bằng đúng trung bình và trục đã học. PCA có center nhưng không mặc nhiên chuẩn hóa phương sai từng feature. Nếu các biến có đơn vị khác nhau, quyết định dùng StandardScaler phải dựa trên ý nghĩa của phương sai cần giữ [25].
**Các tham số quan trọng.** n_components chọn số trục hoặc mức phương sai theo bộ giải được hỗ trợ; svd_solver ảnh hưởng thuật toán số và chi phí; whiten thay đổi thang các thành phần. Với randomized solver còn có random_state và mức lặp. Fit PCA bằng cả Test dù không sử dụng nhãn vẫn cho mô hình biết phân phối tương lai; bộ giảm chiều phải thuộc pipeline của từng fold.
**Đánh giá hiệu quả.** Explained variance ratio đo phần phương sai được giữ, reconstruction error đo mức mất thông tin tái tạo. Hai chỉ số này không thay thế sai số của mô hình sau giảm chiều. Một hướng có phương sai nhỏ vẫn có thể chứa tín hiệu dự đoán; ngược lại hướng phương sai lớn có thể chủ yếu biểu diễn mức phổ biến. Do đó cần cùng lúc đánh giá chất lượng tái tạo, RMSE/MAE và tổng chi phí của chuỗi xử lý.
**Ưu điểm, nhược điểm và ứng dụng.** PCA giúp nén các đặc trưng tương quan, trực quan hóa và đôi khi giảm nhiễu. Hạn chế là biến mới trộn nhiều feature nên khó giải thích, phụ thuộc thang đo và chỉ mô tả cấu trúc tuyến tính. Với cây chia theo từng trục, phép xoay PCA có thể làm một quan hệ dễ chia trước đó trở nên khó biểu diễn; vì thế giảm số chiều không bảo đảm tăng chất lượng boosting.
### 2.5.2. TruncatedSVD
**Lý thuyết.** TruncatedSVD xấp xỉ X bằng k bộ singular vector/giá trị lớn nhất mà không trừ trung bình. Tài liệu chính thức phân biệt điểm này với PCA và nêu khả năng xử lý ma trận sparse hiệu quả [12]. Các singular vector thể hiện hướng năng lượng lớn của ma trận gốc; một hướng mạnh có thể phản ánh mức phổ biến hoặc tần suất hơn là khác biệt sở thích đã center.
= X ≈ UₖΣₖVₖᵀ;     Z = XVₖ;     X̂ = ZVₖᵀ.
**Cách hoạt động.** Bộ giải randomized hoặc ARPACK tìm không gian con kích thước nhỏ, sau đó chiếu dữ liệu vào không gian ấy. Không center giúp tránh việc một ma trận nhiều số 0 trở thành ma trận đặc chỉ vì trừ trung bình. Tuy nhiên, đầu ra Z thường là dense; chi phí lưu n×k vẫn phải được dự trù, đặc biệt khi n là số lượt tương tác.
**Các tham số quan trọng.** n_components điều khiển hạng xấp xỉ; algorithm chọn bộ giải; n_iter, n_oversamples và random_state ảnh hưởng phép xấp xỉ randomized [12]. Nên kiểm tra việc tăng số vòng còn cải thiện kết quả hay chỉ tăng chi phí. Nếu dữ liệu đã được center ở bước trước, lợi thế “không center” của chính TruncatedSVD không còn đồng nghĩa toàn bộ pipeline giữ được sparse.
**Đánh giá hiệu quả.** Reconstruction error phải được tính trên cùng ma trận và cùng quy ước chuẩn hóa. Explained variance của TruncatedSVD không được diễn giải như PCA khi hai đầu vào khác nhau về centering. Đối với ma trận user–item, lỗi trên mọi ô có thể bị các ô 0 áp đảo; cần nêu rõ mask và ý nghĩa của 0. Với feature matrix phục vụ hồi quy, đo thêm sai số rating và thời gian transform/predict.
**Ưu điểm, nhược điểm và ứng dụng.** Phương pháp phù hợp nén đặc trưng thưa hoặc biểu diễn văn bản, có thể dùng tạo latent representation. Nó không tự trở thành mô hình collaborative filtering học có mặt nạ: SVD-CF tối ưu rating quan sát, còn TruncatedSVD phân rã ma trận số được đưa vào. Do vậy, một kết quả reconstruction thấp của TruncatedSVD không chứng minh RMSE rating của SVD-CF cũng thấp.
### 2.5.3. Nguyên tắc so sánh mô hình sau giảm chiều
Một phép so sánh downstream có ý nghĩa giữ nguyên tập dữ liệu, split, tập ứng viên cấu hình và cách đo, chỉ thay bước biểu diễn: feature gốc, PCA hoặc TruncatedSVD. Mỗi fold cần fit scaler/reducer độc lập trên phần Train tương ứng; Validation dùng chọn k và cấu hình predictor, Test chỉ dùng sau khi khóa lựa chọn. Cần báo cả chi phí reducer lẫn predictor, vì giảm số chiều nhưng thêm phép biến đổi có thể không giảm latency toàn chuỗi.
= Reconstruction MSE = ‖X − X̂‖²F/(nd); EVR(k) = Σⱼ₌₁ᵏλⱼ / Σⱼ₌₁ᵈλⱼ (PCA).
## 2.6. Q-LEARNING
### 2.6.1. Cơ sở lý thuyết và cách hoạt động
Học tăng cường tối ưu hành động theo tổng lợi ích nhiều bước. Một Markov Decision Process gồm trạng thái S, hành động A, chuyển trạng thái, reward và hệ số chiết khấu γ. Qπ(s,a) là kỳ vọng tổng reward chiết khấu nếu chọn a ở s rồi theo chính sách π. Q-Learning là phương pháp model-free, off-policy: học hướng về hành động tốt nhất ở bước sau dù dữ liệu có thể sinh bởi chính sách đang khám phá [13].
= Q(s,a) ← Q(s,a) + α[r + γ maxₐ′Q(s′,a′) − Q(s,a)].
Quy trình gồm quan sát trạng thái, chọn hành động, nhận reward/trạng thái mới, tính sai lệch temporal difference rồi cập nhật Q. Với trạng thái kết thúc, target không có phần bootstrap tương lai. Chính sách ε-greedy chọn ngẫu nhiên với xác suất ε và khai thác hành động Q cao nhất ở phần còn lại. Trong gợi ý, phải loại hành động không hợp lệ như phim đã được đề xuất nếu môi trường yêu cầu không lặp.
### 2.6.2. Các tham số và điều kiện học
Learning rate α quyết định mức cập nhật, γ điều chỉnh trọng số lợi ích tương lai, ε và lịch giảm ε điều khiển khám phá. Số episode, horizon, định nghĩa reward và độ hạt trạng thái tác động trực tiếp đến bài toán. Với Q-table, kích thước bảng tăng theo |S|×|A|. Định lý hội tụ cổ điển đòi hỏi các điều kiện về MDP hữu hạn, mức thăm cặp state–action và lịch bước học; không được chuyển nguyên bảo đảm ấy sang một mạng nơ-ron bất kỳ [13].
Trong gợi ý tuần tự, trạng thái có thể chứa hồ sơ sở thích, lịch sử và phản hồi vừa nhận; hành động là phim được chọn. Nếu trạng thái bỏ sót thông tin ảnh hưởng quyết định tiếp theo, giả thiết Markov chỉ còn là xấp xỉ. Nếu chọn rating làm reward thì mục tiêu được học là rating theo định nghĩa đó, chưa phải sự hài lòng, tính đa dạng hoặc khả năng giữ chân người dùng.
### 2.6.3. Mở rộng Deep Q-Network
Khi bảng quá lớn, DQN dùng Qθ(s,a) để xấp xỉ giá trị hành động. Experience replay lưu chuyển tiếp và lấy minibatch nhằm giảm tương quan chuỗi; target network cung cấp mục tiêu tương đối ổn định. Mnih và cộng sự trình bày cơ chế này trong công trình điều khiển bằng deep reinforcement learning [14]. Các tham số bổ sung gồm độ rộng mạng, replay capacity, batch size và chu kỳ cập nhật target.
= L(θ) = E[(r + γ maxₐ′Qθ⁻(s′,a′) − Qθ(s,a))²].
DQN và Q-table chia sẻ nguyên tắc Bellman nhưng khác biểu diễn, tối ưu và tính ổn định. Khi báo cáo cần ghi đúng thuật toán đã chạy; kết quả DQN không thể được mô tả như thực nghiệm bảng Q. Mạng có thể khái quát giữa các trạng thái, song cũng có thể ước lượng Q thiếu ổn định hoặc quá cao, nên cần theo dõi reward đánh giá và checkpoint tách khỏi dữ liệu huấn luyện.
### 2.6.4. Đánh giá, ưu điểm, nhược điểm và ứng dụng
Đánh giá gồm cumulative reward, reward mỗi action, biến thiên giữa episode, độ phủ phản hồi, thời gian và đối chiếu Random/Popularity trên cùng người dùng, catalog, mask và horizon. Cumulative reward phụ thuộc số action thực tế; nếu episode dài khác nhau cần công bố mẫu số. So sánh theo cặp user hữu ích hơn ghép hai trung bình có cohort khác nhau. Reward huấn luyện tăng không đủ chứng minh chính sách tổng quát hóa.
Ưu điểm là mô hình hóa lựa chọn nhiều bước và đánh đổi ngắn hạn–dài hạn. Hạn chế lớn trong recommendation offline là chỉ quan sát rating cho một phần rất nhỏ hành động: không có phản hồi không có nghĩa là không thích. Gán reward 0 cho phần chưa quan sát là quy ước môi trường, chưa phải bằng chứng reward thật bằng 0. Do đó cần ghi rõ giả lập, độ phủ và giới hạn suy luận; triển khai thực tế cần thu thập phản hồi có thiết kế đánh giá phù hợp.
"""


EXTRA_REFERENCES = [
    '[18] Scikit-learn Developers, “LinearRegression,” tài liệu chính thức. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html. Truy cập 26/11/2026.',
    '[19] XGBoost Developers, “XGBoost Parameters,” tài liệu chính thức. https://xgboost.readthedocs.io/en/stable/parameter.html. Truy cập 26/11/2026.',
    '[20] Scikit-learn Developers, “LogisticRegression,” tài liệu chính thức. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html. Truy cập 26/11/2026.',
    '[21] Scikit-learn Developers, “Ensembles: Random Forests,” tài liệu chính thức. https://scikit-learn.org/stable/modules/ensemble.html#forest. Truy cập 26/11/2026.',
    '[22] Scikit-learn Developers, “KMeans,” tài liệu chính thức. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html. Truy cập 26/11/2026.',
    '[23] Scikit-learn Developers, “DBSCAN,” tài liệu chính thức. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html. Truy cập 26/11/2026.',
    '[24] S. Raschka, “Association Rules,” mlxtend User Guide. https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/. Truy cập 26/11/2026.',
    '[25] Scikit-learn Developers, “PCA,” tài liệu chính thức. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html. Truy cập 26/11/2026.',
    '[26] Scikit-learn Developers, “Clustering: Performance Evaluation,” tài liệu chính thức. https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation. Truy cập 26/11/2026.',
]


def insert_before(anchor, text, style):
    p = anchor.insert_paragraph_before(style=style)
    # Mark the six required theoretical aspects without adding a fourth TOC level.
    match = re.match(r"\*\*(.*?)\*\*\s*(.*)", text)
    if match:
        p.add_run(match.group(1) + " ").bold = True
        p.add_run(match.group(2))
    else:
        p.add_run(text)
    return p


def replace_theory(doc):
    paragraphs = doc.paragraphs
    start = next(i for i, p in enumerate(paragraphs) if p.text == "CHƯƠNG 2: CƠ SỞ LÝ THUYẾT")
    end = next(i for i, p in enumerate(paragraphs) if p.text == "CHƯƠNG 3: DỮ LIỆU VÀ THIẾT KẾ THỰC NGHIỆM")
    anchor = paragraphs[end]
    for p in paragraphs[start + 1:end]:
        p._element.getparent().remove(p._element)
    for line in THEORY.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("### "):
            insert_before(anchor, line[4:], "Report Heading 2")
        elif line.startswith("## "):
            insert_before(anchor, line[3:], "Report Heading 1")
        elif line.startswith("= "):
            insert_before(anchor, line[2:], "Report Equation")
        else:
            insert_before(anchor, line, "Report Body")


def add_acknowledgment(doc):
    anchor = next(p for p in doc.paragraphs if p.text == "MỤC LỤC")
    title = insert_before(anchor, "LỜI CẢM ƠN", "Report Front")
    title.paragraph_format.page_break_before = False
    insert_before(anchor,
                  "Nhóm thực hiện trân trọng cảm ơn giảng viên hướng dẫn và Khoa Công nghệ thông tin, "
                  "Trường Đại học Công Thương TP. Hồ Chí Minh đã tạo điều kiện học tập và thực hiện đồ án. "
                  "Nhóm cảm ơn GroupLens Research thuộc University of Minnesota đã công bố bộ dữ liệu "
                  "MovieLens phục vụ nghiên cứu, cùng các tác giả và cộng đồng phát triển thư viện mã nguồn "
                  "mở được sử dụng trong hệ thống. Những hạn chế về phương pháp và kết quả được nhóm "
                  "trình bày trong báo cáo và là cơ sở để tiếp tục hoàn thiện đề tài.", "Report Body")
    anchor.paragraph_format.page_break_before = True


def add_introduction(doc):
    anchor = next(p for p in doc.paragraphs if p.text == "CHƯƠNG 1: TỔNG QUAN")
    title = insert_before(anchor, "MỞ ĐẦU", "Report Chapter")
    title.paragraph_format.page_break_before = False
    anchor.paragraph_format.page_break_before = True
    insert_before(anchor,
        "Báo cáo trình bày quá trình xây dựng hệ thống dự đoán điểm đánh giá và gợi ý phim trên "
        "MovieLens 25M. Đề tài kết hợp mô hình dự đoán có giám sát, phương pháp khám phá cấu trúc "
        "sở thích và mô phỏng lựa chọn tuần tự, từ đó phân tích khả năng áp dụng vào một web demo "
        "cá nhân hóa. Câu hỏi trọng tâm là mô hình nào đem lại chất lượng dự đoán tốt trong giới hạn "
        "chi phí tính toán của hệ thống hiện có.", "Report Body")
    insert_before(anchor,
        "Cơ sở lập luận gồm bài báo nền tảng, tài liệu chính thức của thư viện và dấu vết thực nghiệm "
        "được lưu trong project. Phần lý thuyết được tách khỏi cấu hình cài đặt; số liệu đánh giá chỉ "
        "trích từ các lượt chạy hoàn tất và có nêu rõ cách chia tập, số fold, phạm vi dữ liệu và "
        "giới hạn suy luận. Những hạng mục yêu cầu nhưng chưa có kết quả được ghi nhận để bổ sung, "
        "đặc biệt phép so sánh XGBoost với PCA và TruncatedSVD.", "Report Body")
    insert_before(anchor,
        "Nội dung gồm năm chương: Chương 1 xác định vấn đề, mục tiêu và phạm vi; Chương 2 hệ thống "
        "hóa lý thuyết thuật toán; Chương 3 mô tả dữ liệu và thiết kế đánh giá; Chương 4 đối chiếu "
        "kết quả thực nghiệm cùng chi phí; Chương 5 tổng kết mức đạt được, hạn chế và hướng phát "
        "triển. Tài liệu tham khảo và phụ lục cung cấp nguồn học thuật cùng dấu vết truy nguyên số liệu.",
        "Report Body")


def compact_result_tables(doc, runs):
    replacements = {}
    regression = [["Mô hình", "RMSE CV (mean ± std)", "MAE CV (mean ± std)", "RMSE Test", "MAE Test"]]
    for name, label in [("linear", "Linear Regression"), ("xgboost", "XGBoost"), ("svd", "SVD-CF"), ("ncf", "NCF")]:
        metrics = runs["rating"]["cv"][name + "_raw"]["metrics"]
        test = next(r for r in runs["rating"]["rows"] if r["algorithm"] == name and r["split"] == "test")
        regression.append([label,
            f"{base.vi(metrics['rmse']['mean'], 4)} ± {base.vi(metrics['rmse']['std'], 4)}",
            f"{base.vi(metrics['mae']['mean'], 4)} ± {base.vi(metrics['mae']['std'], 4)}",
            base.vi(test["rmse"], 4), base.vi(test["mae"], 4)])
    replacements["Bảng 4.1:"] = regression
    classification = [["Mô hình", "Ngưỡng", "Precision Test", "Recall Test", "[[TN,FP], [FN,TP]]"]]
    for name, label in [("logistic", "Logistic"), ("random_forest", "Random Forest")]:
        row = next(r for r in runs["classification"]["rows"] if r["algorithm"] == name and r["split"] == "test")
        matrix = row["confusion_matrix"]
        classification.append([label, base.vi(runs["classification"]["best_configs"][name + "_raw"]["threshold"], 2),
            base.vi(row["precision"], 4), base.vi(row["recall"], 4),
            f"[[{base.vi_int(matrix[0][0])}, {base.vi_int(matrix[0][1])}], [{base.vi_int(matrix[1][0])}, {base.vi_int(matrix[1][1])}]]"])
    replacements["Bảng 4.4:"] = classification
    association = runs["association"]["rows"][0]
    replacements["Bảng 4.7:"] = [
        ["Chỉ tiêu", "Giá trị"],
        ["Transactions Train", base.vi_int(association["transactions"])],
        ["Ngưỡng rating yêu thích / danh mục", "≥ 4,0 sao / 80 phim"],
        ["Frequent itemsets / số luật", f"{base.vi_int(association['itemsets'])} / {base.vi_int(association['rule_count'])}"],
        ["Support trung bình", base.vi(association["support"], 4)],
        ["Confidence trung bình", base.vi(association["confidence"], 4)],
        ["Lift trung bình", base.vi(association["lift"], 4)],
    ]
    rl = [["Policy", "Cumulative reward", "Reward/action", "Feedback quan sát", "NDCG@5"]]
    for name, label in [("dqn", "DQN"), ("random", "Random"), ("popular", "Popularity")]:
        row = next(r for r in runs["reinforcement"]["rows"] if r["algorithm"] == name and r["split"] == "test")
        rl.append([label, base.vi(row["cumulative_reward"], 2), base.vi(row["average_reward"], 5),
                   base.vi(row["observed_feedback_fraction"] * 100, 2) + "%", base.vi(row["ndcg_at_horizon"], 4)])
    replacements["Bảng 4.9:"] = rl
    for p in list(doc.paragraphs):
        for prefix, rows in replacements.items():
            if p.text.startswith(prefix):
                old = p._element.getnext()
                assert old.tag == qn("w:tbl")
                before_count = len(doc.tables)
                base.add_table_rows(doc, rows)
                new = doc.tables[-1]._element
                old.addprevious(new)
                old.getparent().remove(old)
                assert len(doc.tables) == before_count
    for p in doc.paragraphs:
        if p.text.startswith("Bảng 4.4:"):
            p.text = "Bảng 4.4: Ngưỡng phân loại và ma trận nhầm lẫn trên Test"
    anchor = next(p for p in doc.paragraphs if p.text == "4.3. NHÓM THUẬT TOÁN PHÂN CỤM")
    insert_before(anchor,
        "Thời gian fit cuối của Logistic là 266,33 giây, Random Forest là 141,96 giây; inference trên "
        "batch Test lần lượt 0,881 và 2,746 µs/rating. Peak RSS ghi nhận 6.452,67 MiB và 10.101,00 MiB "
        "là bộ nhớ của tiến trình trong phép đo, không phải kích thước riêng mô hình. Kích thước "
        "serialized tương ứng khoảng 34,53 MiB và 37,90 MiB. Cần tách ba đại lượng này khi cân nhắc "
        "giới hạn RAM của máy phục vụ.", "Report Body")


def expand_design(doc, runs):
    anchor = next(p for p in doc.paragraphs if p.text == "3.3. CẤU HÌNH VÀ CÁCH ÁP DỤNG CÁC THUẬT TOÁN")
    insert_before(anchor,
        "Riêng thí nghiệm giảm chiều dự kiến, đầu vào phải là cùng feature matrix dùng cho XGBoost gốc, "
        "không thay bằng ma trận user–genre của phân cụm. Ba nhánh gồm XGBoost, PCA→XGBoost và "
        "TruncatedSVD→XGBoost; mọi bước fit reducer nằm trong phần Train của fold. Các chỉ tiêu cần bổ "
        "sung gồm RMSE, MAE, tỷ lệ phương sai giải thích, lỗi tái tạo, thời gian fit toàn chuỗi và thời "
        "gian transform+predict. Đây là thiết kế dành cho lượt chạy tiếp theo, chưa phải kết quả hoàn thành.",
        "Report Body")
    anchor = next(p for p in doc.paragraphs if p.text == "3.3.4. Môi trường và kiến trúc ứng dụng")
    insert_before(anchor,
        "Mô hình Linear trong nhánh thực nghiệm giải OLS theo các thống kê đủ được tích lũy từng batch "
        "trong StreamingLinearRegression, tránh giữ ma trận N×P hoàn chỉnh trong RAM. Đây là lựa chọn "
        "cài đặt OLS; không nhầm với Ridge(alpha=10) trong cấu hình pipeline phục vụ cũ. Tương tự, "
        "các giá trị trong result.json của từng run được dùng làm cấu hình thực nghiệm, thay vì thay "
        "bằng mặc định khác trong configs/default.toml [17].", "Report Body")
    insert_before(anchor,
        "Tính đúng của chia theo thời gian cần hiểu trong phạm vi từng user: mọi rating Test của một "
        "user đến sau Train của chính user đó. Tuy nhiên, mốc của các user không trùng nhau, nên Train "
        "của user A có thể muộn hơn Test của user B theo lịch toàn cục. Các thống kê cộng đồng vì vậy "
        "không mô phỏng đầy đủ một lần triển khai tại cùng ngày cắt. Một thực nghiệm xác nhận nên "
        "bổ sung global temporal split hoặc thống kê as-of-time; không diễn giải protocol hiện có "
        "như đánh giá thời gian toàn cục tuyệt đối.", "Report Body")
    anchor = next(p for p in doc.paragraphs if p.text == "3.4.4. Học tăng cường và giới hạn đo")
    insert_before(anchor, "3.4.4. Giảm chiều dữ liệu — kế hoạch đánh giá", "Report Heading 2")
    insert_before(anchor,
        "Phương sai giải thích được tính trên cùng biểu diễn đầu vào; lỗi tái tạo cần nêu rõ normalization "
        "và tập được đánh giá. Chỉ số quyết định của nhánh downstream vẫn là RMSE/MAE rating trên "
        "cùng các cặp user–movie. Giữ nguyên ngân sách tuning giữa các nhánh, công bố số components "
        "được chọn bằng Validation và tính cả chi phí fit/transform reducer. Không sử dụng kết quả "
        "PCA 2D trực quan hóa cụm để thay thế bảng đánh giá này.", "Report Body")
    anchor.text = "3.4.5. Học tăng cường và giới hạn đo"
    anchor = next(p for p in doc.paragraphs if p.text == "CHƯƠNG 4: THỰC NGHIỆM VÀ SO SÁNH")
    insert_before(anchor, "3.4.6. Đối chiếu yêu cầu đề tài và bằng chứng thực hiện", "Report Heading 2")
    rows = [
        ["Yêu cầu", "Bằng chứng hiện có / phạm vi kết luận"],
        ["MovieLens 25M; làm sạch, missing", "Bản dữ liệu sạch và cleaning_report.json; giữ 25.000.095 rating hợp lệ; missing không thành rating 0."],
        ["Lọc cold-start user/phim", "Có điều kiện lịch sử user và ranh giới timestamp; chưa chứng minh một bộ lọc ngưỡng tương tác phim cùng đánh giá cold-start riêng."],
        ["70/15/15 theo thời gian", "Có split trong từng user ở nhóm supervised; Apriori dùng chia thời gian toàn cục."],
        ["CV 5 fold; mean ± std", "Run hiện có ghi 3 expanding-time fold; báo cáo 3 fold, yêu cầu 5 fold còn thiếu."],
        ["GridSearchCV/RandomizedSearchCV, Optuna", "tune=false trong run chọn; chưa có kết quả search tương ứng; không xem early stopping là Optuna."],
        ["Bốn mô hình rating, learning curve NCF", "Có Linear/XGBoost/SVD-CF/NCF, metric và đường học đã lưu."],
        ["Mean/KNN baseline cho rating", "Có lớp baseline và item_knn trong mã nguồn; run rating hiện được báo cáo chưa ghi hai baseline cùng protocol."],
        ["Phân loại, cụm, Apriori", "Có artifact hoàn tất và metric theo nhóm; DBSCAN một cụm, Apriori chưa có đánh giá ngoài Train trong run chọn."],
        ["PCA/TruncatedSVD + XGBoost", "Chưa có lượt chạy so sánh; phần kết quả 4.5 để trống."],
        ["Q-Learning và môi trường Gymnasium", "Có môi trường offline tự cài và DQN; không thấy Gymnasium Env trong nhánh được chạy, chưa có kết quả Q-table riêng."],
        ["Seed, log, training/inference time", "Seed 42, run JSON/log và metric thời gian; không có bằng chứng dùng MLflow cho các run chọn."],
        ["Web demo và model triển khai", "Có FastAPI + React; model từ Studio và model endpoint phục vụ là hai luồng artifact, cần kiểm tra nạp trước triển khai."],
    ]
    # Add a table at its intended position using the existing formatting routine.
    marker = doc.paragraphs[-1]._element
    base.add_table(doc, "Bảng 3.6: Mức đáp ứng yêu cầu đề tài trong các kết quả đã lưu", rows)
    body = doc._element.body
    new = []
    reached = False
    for child in list(body):
        if child is marker:
            reached = True
            continue
        if reached and child.tag != qn("w:sectPr"):
            new.append(child)
    for child in new:
        anchor._element.addprevious(child)


def refine_results(doc, runs):
    # Use a compact results table: wide seven/eight-column tables do not fit 15 cm well.
    for table in doc.tables:
        if len(table.columns) >= 7:
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        paragraph.paragraph_format.keep_together = True
    for p in doc.paragraphs:
        if p.text.startswith("Thời gian fit model ghi nhận 0,25 giây"):
            p.text = (
                "DBSCAN dùng implementation full-data của project do số user vượt 5.000. Thời gian "
                "gán profile mới được đo theo quy tắc lõi gần nhất, cần phân biệt với thời gian fit "
                "và tạo hồ sơ. Việc chỉ tìm được một cụm khiến các chỉ số nội bộ không đủ điều kiện "
                "đánh giá chất lượng phân tách; báo N/A là đúng hơn thay bằng 0."
            )
        if p.text.startswith("Silhouette cao hơn và Davies–Bouldin thấp hơn"):
            p.add_run(" Các điều kiện xác định của chỉ số được đối chiếu tài liệu đánh giá cụm [26].")
    # Move RL environment details out of the old theoretical chapter (now replaced).
    anchor = next(p for p in doc.paragraphs if p.text == "3.4. PHƯƠNG PHÁP ĐÁNH GIÁ")
    insert_before(anchor, "3.3.5. Môi trường học tăng cường đã cài đặt", "Report Heading 2")
    insert_before(anchor,
        "Nhánh đã chạy sử dụng DQN bằng PyTorch và môi trường Python tự xây từ lịch sử MovieLens, "
        "không phải Q-table. State kết hợp profile thể loại, độ đầy đủ hồ sơ, action mask, số bước "
        "còn lại và phản hồi ở bước trước. Catalog action chọn tối đa 30 phim phổ biến từ Train; "
        "phim đã xem bị loại khỏi hành động hợp lệ. Reward cho rating quan sát bằng (rating−2,75)/2,25, "
        "nằm trong [−1,1]; thiếu feedback cho reward 0 kèm cờ chưa quan sát. Lịch sử Train được chia "
        "support/query theo thời gian để tạo episode; Validation chọn checkpoint trước Test. Mã nguồn "
        "experiment_reinforcement.py không dùng lớp Gymnasium Env, nên không khẳng định đã đáp ứng "
        "phần tích hợp Gymnasium của yêu cầu đề tài [17].", "Report Body")
    # Add full CV metrics for classification without pretending there are five folds.
    anchor = next(p for p in doc.paragraphs if p.text == "4.3. NHÓM THUẬT TOÁN PHÂN CỤM")
    for name, label in [("logistic", "Logistic Regression"), ("random_forest", "Random Forest")]:
        metrics = runs["classification"]["cv"][name + "_raw"]["metrics"]
        parts = []
        for key, shown in [("accuracy", "Accuracy"), ("f1", "F1"), ("auc_roc", "AUC-ROC")]:
            if key in metrics:
                item = metrics[key]
                parts.append(f"{shown}={base.vi(item['mean'], 4)} ± {base.vi(item['std'], 4)}")
        insert_before(anchor, f"Qua 3 fold theo thời gian, {label} đạt " + "; ".join(parts) +
                      ". Đây là biến thiên giữa các lát lịch sử của cấu hình hiện có, chưa phải biến thiên "
                      "qua 5 fold hoặc nhiều seed độc lập.", "Report Body")
    anchor = next(p for p in doc.paragraphs if p.text == "4.4. KHAI PHÁ LUẬT KẾT HỢP APRIORI")
    kmeans, dbscan = runs["clustering"]["rows"]
    insert_before(anchor,
        f"Thời gian fit phần thuật toán K-Means là {base.vi(kmeans['fit_seconds'], 3)} giây, "
        f"DBSCAN là {base.vi(dbscan['fit_seconds'], 3)} giây; bước tạo feature dùng chung mất "
        f"{base.vi(kmeans['timings']['feature_creation_seconds'], 3)} giây. Gán 2.000 profile đánh giá "
        f"mất {base.vi(kmeans['inference_seconds'] * 1000, 3)} ms với K-Means và "
        f"{base.vi(dbscan['inference_seconds'] * 1000, 3)} ms với quy tắc mở rộng DBSCAN. "
        "Các thời gian này tách bước tạo profile và không phải latency một HTTP request. DBSCAN "
        "lưu điểm lõi để gán profile mới theo lõi gần nhất trong eps, ngoài vùng lõi gán noise; đây "
        "là quy tắc phục vụ được cài thêm, không phải predict của thuật toán DBSCAN gốc.", "Report Body")
    # Keep genuine whitespace for the pending reduction results.
    for title in ["4.5.1. XGBoost Regressor", "4.5.2. XGBoost Regressor + PCA", "4.5.3. XGBoost Regressor + TruncatedSVD"]:
        p = next(p for p in doc.paragraphs if p.text == title)
        blank = OxmlElement("w:p")
        p._element.addnext(blank)
    anchor = next(p for p in doc.paragraphs if p.text == "5.3. HƯỚNG PHÁT TRIỂN")
    insert_before(anchor,
        "Các run rating hiện được trích chưa có mean-rating/KNN baseline trên cùng protocol, dù project "
        "có mã nguồn baseline và item_knn. Việc so sánh SVD-CF/NCF với hai mô hình này vì vậy vẫn là "
        "hạng mục cần chạy bổ sung. Không sử dụng số của báo cáo cũ có sample hoặc split khác để "
        "lấp vào bảng hiện tại.", "Report Body")


def improve_references(doc):
    anchor = next(p for p in doc.paragraphs if p.text == "PHỤ LỤC A: DẤU VẾT THỰC NGHIỆM")
    for text in EXTRA_REFERENCES:
        insert_before(anchor, text, "Report Reference")
    for p in doc.paragraphs:
        if p.style.name == "Report Reference" and not p.text.startswith("[17]") and "Truy cập" not in p.text:
            p.add_run(" Truy cập 26/11/2026.")
        if p.text.startswith("[9]"):
            p.text = p.text.replace("/kdd_96.pdf", "/kdd-96.pdf")
        if p.text.startswith("[17]"):
            p.text = p.text.replace("Nhóm 05", "Nhóm thực hiện đồ án")
    # Add source notes to figures and results while avoiding unexplained external graphics.
    for p in list(doc.paragraphs):
        if p.style.name == "Report Figure Caption":
            p.paragraph_format.keep_with_next = True
            note = OxmlElement("w:p")
            p._element.addnext(note)
            from docx.text.paragraph import Paragraph
            q = Paragraph(note, p._parent)
            q.style = doc.styles["Report Body"]
            q.paragraph_format.first_line_indent = Cm(0)
            q.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            q.paragraph_format.space_before = Pt(0)
            q.add_run("Nguồn: nhóm tổng hợp từ dữ liệu, mã nguồn và kết quả đã lưu của project [17].").italic = True


def finish_styles(doc):
    from docx.enum.style import WD_STYLE_TYPE
    from docx.shared import RGBColor
    for name in ["TOC 1", "TOC 2", "TOC 3", "TOC Heading"]:
        style = doc.styles[name] if name in doc.styles else doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        style.font.name = "Times New Roman"
        style.font.size = Pt(13)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.line_spacing = 1.3
        style.paragraph_format.space_before = Pt(0)
        style.paragraph_format.space_after = Pt(0)
        style.paragraph_format.first_line_indent = Cm(0)
    for name in ["Hyperlink", "FollowedHyperlink"]:
        if name in doc.styles:
            doc.styles[name].font.name = "Times New Roman"
            doc.styles[name].font.color.rgb = RGBColor(0, 0, 0)
    # The template footer includes a content control containing a PAGE field.
    # The original builder only cleared direct paragraphs and left that control.
    for section in list(doc.sections)[1:]:
        footer = section.footer._element
        for child in list(footer):
            if child.tag != qn("w:p"):
                footer.remove(child)
    field = next(n for n in doc._element.iter(qn("w:instrText")) if n.text and 'TOC \\o' in n.text)
    field.text = ' TOC \\o "1-3" \\h \\z \\u '
    base.outline_level(doc.styles["Report Front"], 0)
    # Keep all table text at the mandated 12 pt; allocate readable identifier columns.
    for table in doc.tables:
        short_names = {
            "SVD Collaborative Filtering": "SVD-CF",
            "Neural Collaborative Filtering": "NCF",
            "Linear Regression": "Linear",
            "XGBoost Regressor": "XGBoost",
            "Logistic Regression": "Logistic",
        }
        for row_index, row in enumerate(table.rows):
            cell = row.cells[0]
            if cell.text in short_names:
                cell.text = short_names[cell.text]
                for paragraph in cell.paragraphs:
                    paragraph.style = doc.styles["Report Table Text"]
                    for run in paragraph.runs:
                        base.set_run_font(run, 12, bold=row_index == 0)
        if len(table.columns) == 5:
            first = table.cell(0, 0).text
            if first == "Vế trái X":
                widths = [4.3, 4.3, 2, 2.5, 1.9]
            elif first == "Mô hình":
                widths = [3.4, 3.2, 3.2, 2.6, 2.6]
            else:
                widths = [3.2, 3.0, 3.0, 3.0, 2.8]
            for col, width in zip(table.columns, widths):
                col.width = Cm(width)
            for row in table.rows:
                for cell, width in zip(row.cells, widths):
                    cell.width = Cm(width)


def tighten_future_work(doc):
    paragraphs = list(doc.paragraphs)
    start = next(i for i,p in enumerate(paragraphs) if p.text == "5.3. HƯỚNG PHÁT TRIỂN")
    end = next(i for i,p in enumerate(paragraphs) if p.text == "TÀI LIỆU THAM KHẢO")
    anchor = paragraphs[end]
    for p in paragraphs[start+1:end]:
        p._element.getparent().remove(p._element)
    for text in [
        "Về độ tin cậy thực nghiệm, cần hoàn thành 5 fold thời gian theo yêu cầu, lặp nhiều seed và "
        "bổ sung mean-rating/KNN baseline trên cùng protocol. Quy tắc chọn mô hình phải được khóa "
        "bằng CV/Validation trước khi mở Test. Với dữ liệu theo thời gian, nên bổ sung ngày cắt "
        "toàn cục và đo riêng user/item cold-start, độ phủ dự đoán cùng chất lượng fallback.",
        "Về tối ưu mô hình, cần thực hiện search có ngân sách kiểm soát và lưu đầy đủ cấu hình, "
        "metric, RAM, thời gian cùng fingerprint dữ liệu. Phép so sánh XGBoost, XGBoost+PCA và "
        "XGBoost+TruncatedSVD phải fit reducer bên trong từng fold, báo cả chất lượng tái tạo lẫn "
        "RMSE/MAE và chi phí toàn chuỗi. Với NCF, cần ablation và nhiều seed để phân biệt lợi ích "
        "của MLP với ảnh hưởng tối ưu hoặc lựa chọn epoch.",
        "Về khám phá sở thích, cần kiểm tra độ ổn định cụm, độ phủ và chất lượng ranking thay vì "
        "chỉ tăng Silhouette. Các luật Apriori cần được kiểm chứng trên giai đoạn tương lai bằng "
        "độ phủ ứng viên, Precision/Recall và NDCG. DQN cần môi trường phản hồi có thiết kế logging "
        "phù hợp, tích hợp Gymnasium nếu giữ yêu cầu ban đầu và so với Popularity trên cùng "
        "điều kiện; reward offline hiện có chưa đủ đánh giá lợi ích dài hạn.",
        "Về triển khai, cần đưa model được chọn theo tiêu chí đã khóa vào đúng luồng phục vụ web, "
        "đo latency request đơn và chi phí chấm điểm Top-N. Hệ thống cần theo dõi lỗi, độ phủ, "
        "chất lượng theo thời gian và có fallback cho hồ sơ rỗng hoặc ID mới. Các cải tiến này "
        "chuyển trọng tâm từ một bảng điểm offline sang khả năng phục vụ ổn định và kiểm chứng được.",
    ]:
        insert_before(anchor,text,"Report Body")


def add_algorithm_discussion(doc, runs):
    anchor = next(p for p in doc.paragraphs if p.text == "4.2. NHÓM THUẬT TOÁN PHÂN LOẠI")
    details = [
        ("4.1.1. Linear Regression", "Linear là mốc so sánh có RMSE Test 0,8766 và MAE 0,6653. "
         "Fit cuối mất 58,59 giây, thấp nhất trong bốn mô hình rating của run. Đặc trưng đã có thống "
         "kê cộng đồng và hồ sơ thể loại nhưng tổ hợp tuyến tính vẫn cho sai số cao hơn ba mô hình "
         "còn lại. Kết quả này giới hạn ở feature/cấu hình đang dùng; chưa đủ kết luận mọi phương "
         "pháp tuyến tính đều không phù hợp recommendation."),
        ("4.1.2. XGBoost Regressor", "XGBoost đạt RMSE Test 0,8674 và MAE 0,6564, cải thiện "
         "khoảng 1,04% RMSE so với Linear. Tuy nhiên, fit cuối mất 593,50 giây, khoảng 10,13 lần "
         "Linear; latency batch mỗi rating cũng cao hơn. Cấu hình 150 cây, depth 6 chưa được tuning "
         "trong run, nên kết luận phù hợp là cấu hình hiện có chưa đem lại mức cân bằng tốt bằng "
         "factorization, không phải boosting không có khả năng cải thiện nếu đổi đặc trưng hoặc cấu hình."),
        ("4.1.3. SVD Collaborative Filtering", "SVD-CF đạt RMSE Test 0,8088 và MAE 0,6078, "
         "giảm khoảng 7,73% RMSE so với Linear. RMSE Validation 0,8063 gần Test 0,8088, cho thấy "
         "mức sai số tương đối nhất quán giữa hai đoạn ở lượt chạy này. CV mean 0,8504 là thấp nhất "
         "trong bốn mô hình, nhưng các fold dùng lịch sử ngắn hơn nên không so trực tiếp trị số "
         "CV với Test để kết luận có lỗi hoặc có cải thiện do chính tập Test."),
        ("4.1.4. Neural Collaborative Filtering", "NCF có RMSE Validation thấp nhất 0,8046, "
         "hơn SVD-CF khoảng 0,0017 sao theo tiêu chí lựa chọn đã khóa. Sau refit, Test RMSE là "
         "0,8424, cao hơn SVD-CF và thấp hơn hai mô hình bảng. Việc thêm MLP không bảo đảm lợi ích "
         "ngoài mẫu: khác biệt tối ưu, lịch sử và epoch refit đều có thể tác động. Không có thực "
         "nghiệm ablation hoặc nhiều seed để quy nguyên nhân chênh lệch Test chỉ cho overfitting."),
    ]
    for title, text in details:
        insert_before(anchor, title, "Report Heading 2")
        insert_before(anchor, text, "Report Body")
    anchor = next(p for p in doc.paragraphs if p.text == "4.3. NHÓM THUẬT TOÁN PHÂN CỤM")
    insert_before(anchor, "4.2.1. Logistic Regression", "Report Heading 2")
    insert_before(anchor,
        "Ma trận nhầm lẫn Test có 204.590 false negative và 814.262 false positive ở ngưỡng 0,40. "
        "Mô hình giữ Recall lớp Thích trên 91%, phù hợp nếu ưu tiên ít bỏ sót lựa chọn có thể thích, "
        "nhưng nhiều phim không thích vẫn được gán dương. AUC-ROC 0,7943 phản ánh khả năng xếp "
        "thứ tự xác suất, không có nghĩa 79,43% gợi ý trong Top-N sẽ được người dùng thích.", "Report Body")
    insert_before(anchor, "4.2.2. Random Forest", "Report Heading 2")
    insert_before(anchor,
        "Ở ngưỡng 0,45, Random Forest giảm false positive xuống 787.775 và tăng false negative "
        "lên 222.026. Precision nhỉnh hơn Logistic nhưng Recall thấp hơn, còn F1 chỉ hơn khoảng "
        "0,00008. Hai mô hình vì vậy đem lại hai điểm vận hành khác nhau; chưa có kiểm định lặp "
        "theo người dùng để xem chênh lệch nhỏ này là ổn định. Các kết quả không cho phép tuyên bố "
        "RF vượt trội rõ rệt trên mọi tiêu chí.", "Report Body")
    anchor = next(p for p in doc.paragraphs if p.text == "4.4. KHAI PHÁ LUẬT KẾT HỢP APRIORI")
    insert_before(anchor, "4.3.1. K-Means", "Report Heading 2")
    insert_before(anchor,
        "Cụm 0 có 107.417 user (67,41%), cụm 1 có 51.939 user. Trung bình preference tương đối "
        "đã làm trơn của Drama/Crime/War ở cụm 1 lần lượt khoảng 0,164/0,130/0,097; cụm 0 có "
        "Drama khoảng 0,039, Animation 0,037 và War 0,035. Đây là độ lệch tương đối theo hồ sơ "
        "được định nghĩa, không phải điểm đánh giá tuyệt đối hay tỷ lệ người thích từng thể loại. "
        "Silhouette thấp cho thấy ranh giới hai nhóm chồng lấn đáng kể; nhãn cụm chỉ hỗ trợ mô tả.", "Report Body")
    insert_before(anchor, "4.3.2. DBSCAN", "Report Heading 2")
    insert_before(anchor,
        "Cụm hợp lệ duy nhất chứa 157.754 user, còn 1.602 user là noise. Kết quả không hình thành "
        "nhiều phân đoạn sở thích để so sánh, dù coverage xấp xỉ 98,99%. Không nên gọi noise là "
        "nhóm người có sở thích bất thường theo nghĩa hành vi hoặc nhân khẩu: đó chỉ là các điểm "
        "không đạt điều kiện mật độ của biểu diễn và tham số đang dùng. Thử lại khoảng cách/chuẩn "
        "hóa hoặc tham số là hướng nghiên cứu tiếp theo, chưa phải kết quả đã chứng minh.", "Report Body")


def clarify_evaluation(doc):
    anchor = next(p for p in doc.paragraphs if p.text == "2.3. NHÓM THUẬT TOÁN PHÂN CỤM")
    insert_before(anchor, "2.2.3. Diễn giải các chỉ số phân loại", "Report Heading 2")
    insert_before(anchor,
        "Với lớp dương, TP là dự đoán dương đúng, FP là dự đoán dương sai, FN là mẫu dương bị bỏ "
        "sót và TN là mẫu âm nhận đúng. Precision đo độ đúng của tập được dự đoán dương; Recall "
        "đo tỷ lệ mẫu dương tìm thấy; F1 là trung bình điều hòa, giảm mạnh nếu một trong hai thành "
        "phần thấp. Cần chỉ rõ báo F1 của lớp nào hoặc loại trung bình nào khi có nhiều lớp [15].",
        "Report Body")
    insert_before(anchor, "Accuracy=(TP+TN)/N; Precision=TP/(TP+FP); Recall=TP/(TP+FN).", "Report Equation")
    insert_before(anchor, "F1=2TP/(2TP+FP+FN); TPR=Recall; FPR=FP/(FP+TN).", "Report Equation")
    insert_before(anchor,
        "ROC thể hiện TPR theo FPR khi quét ngưỡng; AUC tổng hợp khả năng sắp thứ tự, không đo "
        "xác suất dự đoán đã được hiệu chỉnh hay chất lượng của một danh sách Top-N cụ thể. Khi "
        "đánh giá trong điều kiện thay đổi tỷ lệ lớp, nên đọc cùng ma trận nhầm lẫn và chỉ số "
        "liên quan trực tiếp đến chi phí quyết định, đồng thời giữ nguyên quy tắc chọn ngưỡng.",
        "Report Body")
    anchor = next(p for p in doc.paragraphs if p.text == "2.4. KHAI PHÁ LUẬT KẾT HỢP")
    insert_before(anchor, "2.3.3. Cách tính chỉ số phân cụm", "Report Heading 2")
    insert_before(anchor,
        "Với điểm i, a(i) là khoảng cách trung bình tới các điểm cùng cụm, b(i) là khoảng cách "
        "trung bình nhỏ nhất tới một cụm khác. Silhouette của toàn bộ mẫu là trung bình s(i). "
        "Davies–Bouldin dùng độ phân tán Sₖ trong cụm và khoảng cách Mₖₗ giữa các tâm, phạt các "
        "cặp cụm vừa rộng vừa gần nhau [26].", "Report Body")
    insert_before(anchor, "s(i)=[b(i)−a(i)]/max[a(i),b(i)]; DB=(1/K)Σₖmaxₗ≠ₖ[(Sₖ+Sₗ)/Mₖₗ].", "Report Equation")
    insert_before(anchor,
        "Hai chỉ số đo cấu trúc theo biểu diễn đã chọn. Nếu các điểm noise bị loại khi tính, "
        "phải công bố tỷ lệ được giữ lại và định nghĩa tập đánh giá. Với tập lớn có thể lấy mẫu "
        "để tính Silhouette nhưng cần ghi cỡ mẫu và seed, tránh diễn giải chỉ số lấy mẫu như một "
        "phép tính cặp khoảng cách trên toàn bộ người dùng.", "Report Body")
    anchor = next(p for p in doc.paragraphs if p.text == "3.4.3. Phân cụm, luật và ranking")
    insert_before(anchor,
        "Trong implementation thực nghiệm hiện tại, F1 của CV dùng ngưỡng cố định 0,5; ngưỡng "
        "0,40/0,45 của Logistic/Random Forest được chọn trên Validation cho bước báo cáo cuối. "
        "Do đó, chênh lệch F1 CV–Test không chỉ do thời gian và refit mà còn do điểm vận hành "
        "khác nhau; không so hai giá trị như cùng một ngưỡng. Độ lệch chuẩn CV được tính với "
        "ddof=1 trên ba fold [17].", "Report Body")


def validate(doc, runs, output):
    template = Document(base.TEMPLATE)
    # Compare canonical XML of all cover paragraphs, including embedded drawing references.
    covers_equal = all(a.xml == b.xml for a, b in zip(list(template._element.body)[:48], list(doc._element.body)[:48]))
    assert covers_equal, "Cover pages were changed"
    texts = [p.text for p in doc.paragraphs]
    chapter2 = texts[texts.index("CHƯƠNG 2: CƠ SỞ LÝ THUYẾT") + 1:texts.index("CHƯƠNG 3: DỮ LIỆU VÀ THIẾT KẾ THỰC NGHIỆM")]
    assert not any(word in "\n".join(chapter2).lower() for word in ["trong project", "lần chạy", "artifact", "chạy hiện tại"])
    a = texts.index("4.5. NHÓM GIẢM CHIỀU DỮ LIỆU")
    b = texts.index("4.6. NHÓM HỌC TĂNG CƯỜNG")
    assert all(not text or text.startswith("4.5.") for text in texts[a + 1:b])
    assert sum(p.style.name == "Report Chapter" and p.text.startswith("CHƯƠNG") for p in doc.paragraphs) == 5
    for table in doc.tables:
        assert table.rows[0]._tr.trPr.find(qn("w:tblHeader")) is not None
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    assert p.style.name == "Report Table Text"
                    assert all(r.font.size == Pt(12) for r in p.runs)
    provenance = {
        "output": output.name,
        "template_sha256": hashlib.sha256(base.TEMPLATE.read_bytes()).hexdigest(),
        "cover_xml_preserved": covers_equal,
        "chapters": 5,
        "paragraphs": len(doc.paragraphs),
        "tables": len(doc.tables),
        "figures": len(doc.inline_shapes),
        "word_count_approx": len(" ".join(texts[48:]).split()),
        "theory_excludes_project_experiments": True,
        "reduction_results_blank": True,
        "supervised_cv_folds": {key: runs[key]["cv_folds"] for key in ["rating", "classification"]},
        "sources": {},
    }
    for name, run_id in base.RUN_IDS.items():
        path = base.RUN_ROOT / run_id / "result.json"
        provenance["sources"][name] = {"run_id": run_id, "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "protocol": runs[name]["protocol"]}
    (base.REPO / "BaoCao_MovieLens25M_kiemtra.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return provenance


def main():
    runs = {name: base.load_run(name) for name in base.RUN_IDS}
    # Use serif lettering in the two code-drawn diagrams as in the template.
    base.FONT_PATH = Path("C:/Windows/Fonts/times.ttf")
    base.FONT_BOLD_PATH = Path("C:/Windows/Fonts/timesbd.ttf")
    base.BLACK = base.GRAY = (0, 0, 0)
    assets = base.PROJECT / "docs" / "student-report-assets"
    os.environ["MPLCONFIGDIR"] = str(base.PROJECT / "artifacts" / "scratch" / "report-matplotlib")
    python = base.PROJECT / ".venv" / "Scripts" / "python.exe"
    subprocess.run([str(python), "-X", "utf8", str(Path(__file__).with_name("student_report_plots.py")), str(assets)], check=True)
    with tempfile.TemporaryDirectory(prefix="movielens_report_") as temporary:
        charts = base.create_charts(Path(temporary), runs)
        for key in ["ratings", "regression", "ncf", "classification", "clustering", "rl"]:
            charts[key] = assets / f"{key}.png"
        output = base.create_document(runs, charts)
    doc = Document(output)
    replace_theory(doc)
    add_acknowledgment(doc)
    add_introduction(doc)
    expand_design(doc, runs)
    refine_results(doc, runs)
    compact_result_tables(doc, runs)
    add_algorithm_discussion(doc, runs)
    clarify_evaluation(doc)
    tighten_future_work(doc)
    anchor = next(p for p in doc.paragraphs if p.text == "4.7. TỔNG HỢP VÀ LỰA CHỌN MÔ HÌNH DỰ ĐOÁN RATING")
    last = doc.paragraphs[-1]._element
    base.figure(doc, {"rl_curve": assets / "rl_curve.png"}, "rl_curve",
                "Hình 4.6: Reward Validation tại các checkpoint huấn luyện DQN")
    after = False
    for child in list(doc._element.body):
        if child is last:
            after = True
            continue
        if after and child.tag != qn("w:sectPr"):
            anchor._element.addprevious(child)
    insert_before(anchor,
        "Đường reward hiện lưu theo training step tại checkpoint, không phải tổng reward từng episode. "
        "Validation tốt nhất ở bước 15.003 (0,03200 reward/action), giảm còn 0,02534 ở bước 24.416. "
        "Vì vậy dùng checkpoint tốt nhất đã khóa hợp lý hơn lấy mạng ở bước cuối. Artifact không "
        "cung cấp đường cumulative reward đầy đủ cho từng episode để bổ sung một biểu đồ như vậy.",
        "Report Body")
    improve_references(doc)
    finish_styles(doc)
    doc.core_properties.author = "Nhóm thực hiện đồ án"
    doc.core_properties.last_modified_by = "Nhóm thực hiện đồ án"
    doc.save(output)
    checks = validate(Document(output), runs, output)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
