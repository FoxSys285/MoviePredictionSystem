import type { ExperimentConfigValue } from "./ExperimentConfig";
import type { LabGroup } from "../lib/lab";
import { usePreferences } from "../lib/preferences";

const groupExplanation: Record<string, string> = {
  rating: "Dự đoán rating: Linear/XGBoost nén block tương tác sở thích user × thể loại phim; Funk SVD và NCF giữ latent factors/embeddings rồi cộng nhánh hiệu chỉnh từ hồ sơ nén. TruncatedSVD là reducer, khác Funk SVD.",
  classification: "Phân loại sở thích: nén block tương tác user × thể loại trước Logistic Regression hoặc Random Forest; giữ đặc trưng nền, chọn mô hình và threshold Logistic bằng Validation.",
  clustering: "Phân cụm: nén sở thích thể loại tương đối và tần suất của user; chọn K hoặc tìm lại eps trong không gian đó. Tên/mô tả cụm vẫn lấy từ thể loại gốc.",
  association: "Luật kết hợp: nén hồ sơ user Train để tạo nhóm rồi chạy Apriori trên transaction phim boolean ban đầu. Apriori toàn cục vẫn là baseline; không nén transaction và không dùng Validation/Test để fit reducer.",
  reinforcement: "Học tăng cường: chỉ nén profile sở thích và tần suất đưa vào DQN; giữ nguyên action mask, số bước còn lại và cờ phản hồi. Random/Popularity không cần reducer; giảm chiều không khử được thiên lệch reward offline.",
};

const methodDetails: Record<"pca" | "truncated_svd", { summary: string }> = {
  pca: {
    summary: "PCA trừ trung bình của đặc trưng đã scale, sau đó giữ các hướng có phương sai lớn nhất.",
  },
  truncated_svd: {
    summary: "TruncatedSVD chiếu không trừ trung bình. Explained variance là phương sai thực tế của thành phần chiếu chia tổng phương sai đầu vào đã scale; đây không phải tỷ lệ năng lượng.",
  },
};

const groupApplications: Record<string, Record<"pca" | "truncated_svd", string>> = {
  rating: {
    pca: "PCA thay block tương tác nén cho Linear/XGBoost; SVD/NCF nhận profile nén ở nhánh hiệu chỉnh, còn bias và latent factors/embeddings được giữ.",
    truncated_svd: "TruncatedSVD thay block tương tác cho Linear/XGBoost hoặc cấp profile cho nhánh hybrid SVD/NCF; không thay Funk SVD Collaborative Filtering.",
  },
  classification: {
    pca: "Vector PCA ghép với đặc trưng nền trước Logistic/Random Forest; scaler và reducer fit riêng trong Train và từng fold, threshold Logistic chọn bằng Validation.",
    truncated_svd: "Vector TruncatedSVD ghép với đặc trưng nền trước Logistic/Random Forest; scaler và reducer fit riêng trong Train và từng fold, threshold Logistic chọn bằng Validation.",
  },
  clustering: {
    pca: "Fit scaler và PCA trên user fit, dùng không gian mới để chọn K-Means hoặc tìm eps/min_samples cho DBSCAN; mô tả cụm vẫn tính từ genre gốc.",
    truncated_svd: "Fit scaler và TruncatedSVD trên user fit, rồi khảo sát lại K-Means/DBSCAN trong không gian chiếu; mô tả cụm vẫn tính từ genre gốc.",
  },
  association: {
    pca: "PCA profile user Train để tạo nhóm; Apriori mỗi nhóm đọc transaction boolean ban đầu và giữ Apriori toàn cục làm baseline.",
    truncated_svd: "TruncatedSVD profile user Train để tạo nhóm; Apriori mỗi nhóm đọc transaction boolean ban đầu và giữ Apriori toàn cục làm baseline.",
  },
  reinforcement: {
    pca: "PCA chỉ nén profile sở thích/tần suất đưa vào DQN; action mask, bước còn lại và cờ feedback được nối nguyên trạng.",
    truncated_svd: "TruncatedSVD chỉ nén profile sở thích/tần suất đưa vào DQN; action mask, bước còn lại và cờ feedback được nối nguyên trạng.",
  },
};

export function ExperimentReductionGuide({
  group,
  config,
}: {
  group: LabGroup;
  config: ExperimentConfigValue;
}) {
  const { t } = usePreferences();
  const modes: Record<ExperimentConfigValue["reduction_mode"], string> = {
    off: "Tắt · giữ biểu diễn gốc",
    enabled: "Chỉ biểu diễn giảm chiều",
    compare: "So sánh gốc và giảm chiều",
  };
  const components = config.components.join(", ");

  return (
    <section
      className="lab-section lab-reduction-guide"
      aria-labelledby="lab-reduction-guide-title"
      data-testid="lab-reduction-guide"
      data-group-id={group.id}
      data-mode={config.reduction_mode}
    >
      <div className="lab-section-heading">
        <div>
          <span className="lab-kicker">{t("THÀNH PHẦN DÙNG CHUNG")}</span>
          <h2 id="lab-reduction-guide-title">{t("Giảm chiều dữ liệu")}</h2>
        </div>
        <span className="lab-badge" data-testid="reduction-mode">
          {t(modes[config.reduction_mode])}
        </span>
      </div>
      <p className="lab-reduction-context">{t(groupExplanation[group.id])}</p>
      <p className="lab-muted">
        {t("Mặc định đã bật PCA với các components đang chọn. Có thể đổi chế độ và phương pháp ở phần cấu hình; trạng thái bên dưới phản ánh lựa chọn hiện tại.")}
      </p>
      <div className="lab-reduction-grid">
        {(["pca", "truncated_svd"] as const).map((method) => {
          const active = config.reduction_mode !== "off" && config.reduction_methods.includes(method);
          return (
            <article
              className={`lab-reduction-card${active ? " active" : ""}`}
              key={method}
              data-testid={`reduction-card-${method}`}
              data-method={method}
              data-active={active}
            >
              <div className="lab-reduction-card-heading">
                <h3>{method === "pca" ? "PCA" : "TruncatedSVD"}</h3>
                <span className={`lab-reduction-status${active ? " active" : ""}`}>
                  {t(active ? "Đang dùng" : config.reduction_mode === "off" ? "Chưa bật" : "Không chọn")}
                </span>
              </div>
              <p>{t(methodDetails[method].summary)}</p>
              <div className="lab-reduction-application">
                <strong>{t("Trong nhóm này")}</strong>
                <p>{t(groupApplications[group.id][method])}</p>
              </div>
              <small>{t("Components đang chọn: {components}", { components })}</small>
            </article>
          );
        })}
      </div>
    </section>
  );
}
