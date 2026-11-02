import { useEffect, useState } from "react";
import {
  Award,
  BrainCircuit,
  Check,
  Database,
  LoaderCircle,
  ShieldCheck,
  UsersRound,
  type LucideIcon,
} from "lucide-react";
import { Dialog } from "./ui";
import { usePreferences } from "../lib/preferences";
import {
  ALGORITHM_NAMES,
  api,
  type Algorithm,
  type Experiment,
} from "../lib/api";

export function ModelPanel({ onClose }: { onClose: () => void }) {
  const { t } = usePreferences();
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let active = true;
    api<Experiment>("/experiments/latest")
      .then((data) => {
        if (active) setExperiment(data);
      })
      .catch(() => {})
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);
  return (
    <Dialog
      onClose={onClose}
      title={t("Công nghệ dự đoán phim")}
      className="model-dialog"
    >
      <span className="eyebrow">BEHIND THE SCREEN</span>
      <h2>
        {t("ĐIỆN ẢNH.")}
        <br />
        <span>{t("THẤU HIỂU BẰNG DỮ LIỆU.")}</span>
      </h2>
      <p>{t("Bốn cách tiếp cận, cùng một mục tiêu: tìm bộ phim phù hợp với bạn.")}</p>
      <div className="model-list">
        {(
          [
            [
              "ridge",
              Award,
              "Mô hình tuyến tính làm mốc so sánh, có thể bổ sung regularization.",
            ],
            [
              "xgboost",
              UsersRound,
              "Học quan hệ phi tuyến từ các đặc trưng lịch sử người dùng và phim.",
            ],
            [
              "svd",
              BrainCircuit,
              "Tìm các nhân tố tiềm ẩn trong hành vi đánh giá.",
            ],
            ["ncf", Database, "Kết hợp embedding người dùng, embedding phim và mạng MLP để dự đoán rating."],
          ] as [Algorithm, LucideIcon, string][]
        ).map(([name, Icon, text]) => (
          <div key={name}>
            <span className="feature-icon">
              <Icon size={22} />
            </span>
            <div>
              <h3>
                {ALGORITHM_NAMES[name]}{" "}
                {experiment?.validation.best_algorithm === name && (
                  <span className="best-badge">
                    <Check size={10} /> {t("Tốt nhất")}
                  </span>
                )}
              </h3>
              <p>{t(text)}</p>
            </div>
          </div>
        ))}
      </div>
      {loading ? (
        <p className="subtle-note">
          <LoaderCircle className="spin" size={15} /> {t("Đang tải kết quả thực nghiệm...")}
        </p>
      ) : experiment ? (
        <div className="metrics-table">
            <h3>{t("Kết quả trên tập validation")}</h3>
          <table>
            <thead>
              <tr>
                <th>{t("Thuật toán")}</th>
                <th>RMSE ↓</th>
                <th>MAE ↓</th>
                <th>{t("Huấn luyện")}</th>
              </tr>
            </thead>
            <tbody>
              {experiment.validation.results.map((row) => (
                <tr key={row.algorithm}>
                  <td>{ALGORITHM_NAMES[row.algorithm]}</td>
                  <td>{row.rmse.toFixed(4)}</td>
                  <td>{row.mae.toFixed(4)}</td>
                  <td>{row.fit_seconds.toFixed(2)}s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="model-pending">
          <ShieldCheck size={23} />
          <p>
            {t("Chưa có kết quả huấn luyện để so sánh. Bạn vẫn có thể xem danh mục và chi tiết phim từ dataset.")}
          </p>
        </div>
      )}
    </Dialog>
  );
}
