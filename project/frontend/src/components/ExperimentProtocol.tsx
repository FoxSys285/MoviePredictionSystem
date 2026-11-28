import { usePreferences } from "../lib/preferences";
import { ExperimentDisclosure } from "./ExperimentDisclosure";

const steps = [
  "GridSearchCV 5-fold trên Train: chấm RMSE và MAE, chọn siêu tham số theo RMSE.",
  "Refit từng mô hình với siêu tham số tốt nhất trên toàn bộ Train 70%.",
  "Dùng Validation 15% cho early stopping và so sánh RMSE/MAE của các mô hình được chọn cùng baseline.",
  "Tuỳ chọn refit Train + Validation 85%: giữ siêu tham số đã chốt và số epoch/số cây từ early stopping.",
  "Đánh giá một lần trên Test 15% cho các mô hình được chọn và baseline; báo cáo RMSE và MAE.",
];

export function ExperimentProtocol() {
  const { t } = usePreferences();

  return <ExperimentDisclosure className="experiment-protocol" title="Chia dữ liệu và quy trình thực nghiệm">
    <dl className="experiment-split-summary">
      <div><dt>Train</dt><dd>70%</dd></div>
      <div><dt>Validation</dt><dd>15%</dd></div>
      <div><dt>Test</dt><dd>15%</dd></div>
    </dl>
    <p>{t("Chia theo thời gian trong từng user. Với 100 rating khác thời điểm: 70 Train, 15 Validation, 15 Test.")}</p>
    <p>{t("Các rating cùng timestamp được giữ cùng một tập; tỷ lệ thực tế có thể lệch so với mục tiêu.")}</p>
    <p><strong>Global Mean</strong>{" — "}{t("Baseline dự đoán bằng điểm trung bình của tập Train.")}</p>
    <ol>{steps.map((step) => <li key={step}>{t(step)}</li>)}</ol>
  </ExperimentDisclosure>;
}
