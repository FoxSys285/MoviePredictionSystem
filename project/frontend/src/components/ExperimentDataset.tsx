import type { Dataset } from "../lib/api";
import { usePreferences } from "../lib/preferences";
import { ExperimentDisclosure } from "./ExperimentDisclosure";

const files = [
  ["ratings.csv", "Mã người dùng, mã phim, điểm đánh giá và thời gian đánh giá."],
  ["movies.csv", "Mã phim, tên phim và thể loại."],
  ["tags.csv", "Từ khóa cộng đồng cho từng phim"],
  ["links.csv", "Mã tham chiếu phim trên IMDb và TMDB"],
] as const;

export function ExperimentDataset({ dataset, loading }: { dataset: Dataset | null; loading: boolean }) {
  const { t, language } = usePreferences();
  const locale = language === "vi" ? "vi-VN" : "en-US";
  const number = (value: number) => value.toLocaleString(locale);
  const date = (value: string) => new Date(value).toLocaleDateString(locale, { timeZone: "UTC" });

  return <ExperimentDisclosure className="experiment-dataset" title="Bộ dữ liệu sử dụng">
    <p className="experiment-dataset-description">
      <strong>{dataset?.name || "MovieLens"}</strong>
      {" — "}{t("Bộ dữ liệu đánh giá phim, gồm thông tin phim, điểm của người dùng, nhãn cộng đồng và liên kết tham chiếu.")}
    </p>
    <div aria-busy={loading}>
      {dataset ? <>
        <dl className="experiment-dataset-stats">
          {([
            ["Lượt đánh giá", dataset.rating_count],
            ["Người dùng", dataset.user_count],
            ["Phim trong danh mục", dataset.movie_count],
            ["Thể loại", dataset.genre_count],
          ] as const).map(([label, value]) => <div key={label}>
            <dt>{t(label)}</dt>
            <dd>{number(value)}</dd>
          </div>)}
        </dl>
        <dl className="experiment-dataset-facts">
          <div><dt>{t("Thang điểm đánh giá")}</dt><dd>{t("0,5–5 sao, bước 0,5")}</dd></div>
          <div><dt>{t("Giai đoạn đánh giá (UTC)")}</dt><dd>{date(dataset.rating_start)} – {date(dataset.rating_end)}</dd></div>
          <div><dt>{t("Mật độ ma trận đánh giá")}</dt><dd>{dataset.matrix_density.toLocaleString(locale, { style: "percent", maximumFractionDigits: 4 })}</dd></div>
        </dl>
      </> : <p className="experiment-dataset-status" role="status">
        {t(loading ? "Đang tải thống kê dataset MovieLens..." : "Chưa tải được thống kê bộ dữ liệu. Vui lòng tải lại trang để thử lại.")}
      </p>}
    </div>
    <h4>{t("Các tệp dữ liệu")}</h4>
    <dl className="experiment-dataset-files">
      {files.map(([filename, description]) => <div key={filename}>
        <dt><code>{filename}</code></dt>
        <dd>{t(description)}</dd>
      </div>)}
    </dl>
  </ExperimentDisclosure>;
}
