import { useEffect, useState } from "react";
import {
  Award,
  BrainCircuit,
  Database,
  UsersRound,
  type LucideIcon,
} from "lucide-react";
import { Dialog } from "./ui";
import { usePreferences } from "../lib/preferences";
import {
  ALGORITHM_NAMES,
  api,
  type Algorithm,
} from "../lib/api";

export function ModelPanel({ onClose }: { onClose: () => void }) {
  const { t } = usePreferences();
  const [models, setModels] = useState<{ algorithm: Algorithm; trained: boolean }[]>([]);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let active = true;
    api<{ algorithm: Algorithm; trained: boolean }[]>("/models")
      .then((data) => {
        if (active) setModels(data);
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
              </h3>
              <p>{t(text)}</p>
              <p>{t(loading ? "Đang kiểm tra" : models.some((model) => model.algorithm === name && model.trained) ? "Đã huấn luyện" : "Chưa huấn luyện")}</p>
            </div>
          </div>
        ))}
      </div>
    </Dialog>
  );
}
