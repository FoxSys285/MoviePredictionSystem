import {
  BrainCircuit,
  Clapperboard,
  Database,
  type LucideIcon,
} from "lucide-react";
import { PageHero, SectionTitle } from "../components/ui";
import { genreLabels, numberFormat, scoreFormat } from "../data/movies";
import type { Dataset } from "../lib/api";
import { usePreferences } from "../lib/preferences";

const team = [
  {
    name: "Frontend Developer",
    role: "Trải nghiệm & giao diện",
    avatar: "frontend",
  },
  { name: "Backend Developer", role: "Dữ liệu & hệ thống", avatar: "backend" },
  {
    name: "Machine Learning",
    role: "Mô hình & thực nghiệm",
    avatar: "ml",
  },
  { name: "UI/UX Designer", role: "Thiết kế & tương tác", avatar: "design" },
];

export function About({ dataset }: { dataset: Dataset | null }) {
  const { t, language } = usePreferences();
  return (
    <>
      <PageHero
        title={t("VỀ CHÚNG TÔI")}
        subtitle={t("Chúng tôi là ai?")}
        variant="about"
        align="left"
      />
      <main id="main" className="container about-main">
        <div className="about-intro">
          <div className="about-copy">
            <SectionTitle>{t("Về chúng tôi")}</SectionTitle>
            <p>
              {t("MOVIE là đồ án khám phá phim và dự đoán điểm đánh giá từ dữ liệu MovieLens. Bạn có thể tìm kiếm phim, xem điểm cộng đồng và khám phá những bộ phim cùng thể loại.")}
            </p>
            <p>
              {t("Khám phá 11 thuật toán trong 5 nhóm nhiệm vụ, so sánh Validation và Test bằng các chỉ số chính.")}
            </p>
            <div className="feature-grid">
              <Feature
                icon={BrainCircuit}
                title={t("5 nhóm thực nghiệm")}
                description={t("11 thuật toán · 17 chỉ số · Validation và Test")}
              />
              <Feature
                icon={Clapperboard}
                title={t("Giao diện hiện đại")}
                description={t("Trải nghiệm mượt mà, trực quan")}
              />
              <Feature
                icon={Database}
                title={t("Kho phim phong phú")}
                description={dataset ? `${numberFormat(dataset.movie_count)} ${t("phim")} · ${dataset.genre_count} ${t("thể loại")}` : t("Danh mục phim từ dataset MovieLens")}
              />
            </div>
          </div>
          <figure className="theater-photo">
            <img
              src="/assets/theater.webp"
              alt={t("Rạp chiếu MOVIE với ghế nhung đỏ và màn ảnh lớn")}
            />
            <figcaption>YOUR NEXT GREAT STORY STARTS HERE.</figcaption>
          </figure>
        </div>
        <section className="dataset-section">
          <SectionTitle>{t("Dataset đang sử dụng")}</SectionTitle>
          {dataset ? <>
            <p className="dataset-intro">{t("Thống kê trực tiếp từ bốn tệp CSV của MovieLens. Điểm hiển thị được tổng hợp từ toàn bộ lượt đánh giá trong dataset.")}</p>
            <div className="dataset-stats">
              {[["Bộ phim", numberFormat(dataset.movie_count)], ["Lượt đánh giá", numberFormat(dataset.rating_count)], ["Người dùng", numberFormat(dataset.user_count)], ["Điểm trung bình /5", scoreFormat(dataset.mean_rating)]].map(([label, value]) => <div key={label}><strong>{value}</strong><span>{t(label)}</span></div>)}
            </div>
            <div className="dataset-panels">
              <div className="movie-insight-panel">
                <h3>{t("Thông tin dữ liệu")}</h3>
                <dl className="dataset-facts">
                  <div><dt>{t("Năm phát hành phim")}</dt><dd>{dataset.year_min}–{dataset.year_max}</dd></div>
                  <div><dt>{t("Giai đoạn đánh giá (UTC)")}</dt><dd>{dataset.rating_start.slice(0, 10)} – {dataset.rating_end.slice(0, 10)}</dd></div>
                  <div><dt>{t("Nhãn người dùng")}</dt><dd>{numberFormat(dataset.tag_count)}</dd></div>
                  <div><dt>{t("Liên kết IMDb / TMDB")}</dt><dd>{numberFormat(dataset.link_count)}</dd></div>
                  <div><dt>{t("Phim chưa có đánh giá")}</dt><dd>{numberFormat(dataset.unrated_movie_count)}</dd></div>
                  <div><dt>{t("Mật độ ma trận đánh giá")}</dt><dd>{(dataset.matrix_density * 100).toLocaleString(language === "vi" ? "vi-VN" : "en-US", { maximumFractionDigits: 4 })}%</dd></div>
                </dl>
              </div>
              <div className="movie-insight-panel">
                <h3>{t("Thể loại có nhiều phim nhất")}</h3>
                <div className="dataset-genre-chart">
                  {dataset.genres.slice(0, 6).map((genre) => <div key={genre.name}><span>{t(genreLabels[genre.name] || genre.name)}</span><div className="rating-bar-track"><div style={{ width: `${genre.movie_count / dataset.genres[0].movie_count * 100}%` }} /></div><b>{numberFormat(genre.movie_count)}</b></div>)}
                </div>
                <p className="subtle-note">{t("Một phim có thể thuộc nhiều thể loại.")}</p>
              </div>
            </div>
            <div className="dataset-table-wrap"><table className="dataset-table"><thead><tr><th>{t("Tệp dữ liệu")}</th><th>{t("Nội dung sử dụng")}</th></tr></thead><tbody>
              <tr><td>movies.csv</td><td>{t("Mã phim, tên phim, năm và thể loại")}</td></tr>
              <tr><td>ratings.csv</td><td>{t("Điểm 0,5–5, người đánh giá và thời gian")}</td></tr>
              <tr><td>tags.csv</td><td>{t("Từ khóa cộng đồng cho từng phim")}</td></tr>
              <tr><td>links.csv</td><td>{t("Mã tham chiếu phim trên IMDb và TMDB")}</td></tr>
            </tbody></table></div>
            <p className="subtle-note">{t("Có {withoutYear} phim chưa xác định được năm và {withoutGenres} phim chưa có thể loại. Dataset không chứa poster, cốt truyện, diễn viên hoặc thời lượng.", { withoutYear: numberFormat(dataset.quality.movies_without_year), withoutGenres: numberFormat(dataset.quality.movies_without_genres) })}</p>
          </> : <p className="subtle-note">{t("Chưa tải được thống kê dataset. Bạn có thể tải lại trang để thử lại.")}</p>}
        </section>
        <section className="team-section">
          <SectionTitle>{t("Đội ngũ phát triển")}</SectionTitle>
          <p className="team-note">
            {t("Những vai trò cùng tạo nên trải nghiệm MOVIE.")}
          </p>
          <div className="team-grid">
            {team.map(({ name, role, avatar }) => (
              <article className="team-card" key={name}>
                <div className="team-avatar">
                  <img
                    src={`/assets/team/${avatar}.webp`}
                    alt={`${t("Chân dung minh họa")} — ${name}`}
                    loading="lazy"
                  />
                </div>
                <h3>{name}</h3>
                <p>{t(role)}</p>
                <span className="team-signature">
                  MOVIE <span>TEAM</span>
                </span>
              </article>
            ))}
          </div>
        </section>
      </main>
    </>
  );
}

function Feature({
  icon: Icon,
  title,
  description,
}: {
  icon: LucideIcon;
  title: string;
  description: string;
}) {
  return (
    <div className="feature">
      <span className="feature-icon">
        <Icon size={25} />
      </span>
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}
