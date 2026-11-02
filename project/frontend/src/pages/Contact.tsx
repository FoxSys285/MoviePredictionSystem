import { useState, type FormEvent, type ReactNode } from "react";
import {
  CheckCircle2,
  ChevronDown,
  Clock3,
  Github,
  LoaderCircle,
  Mail,
  MapPin,
  Phone,
  Play,
  Send,
  UsersRound,
  type LucideIcon,
} from "lucide-react";
import { PageHero, SectionTitle, SocialMark } from "../components/ui";
import { api } from "../lib/api";
import type { Notice } from "../types";
import { usePreferences } from "../lib/preferences";

export function Contact({ notify }: { notify: (notice: Notice) => void }) {
  const { t } = usePreferences();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [done, setDone] = useState(false);
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setDone(false);
    setBusy(true);
    const form = event.currentTarget;
    const data = new FormData(form);
    try {
      await api("/contact", {
        method: "POST",
        body: JSON.stringify(Object.fromEntries(data)),
      });
      form.reset();
      setDone(true);
      notify({
        text: t("Tin nhắn của bạn đã được lưu. Cảm ơn bạn đã liên hệ!"),
        kind: "success",
      });
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const details: { icon: LucideIcon; title: string; children: ReactNode }[] = [
    {
      icon: MapPin,
      title: t("Địa chỉ"),
      children: (
        <>
          {t("Đồ án Học máy — HUIT")}
          <br />
          {t("TP. Hồ Chí Minh")}
        </>
      ),
    },
    {
      icon: Mail,
      title: t("Email"),
      children: <>{t("Hỗ trợ và góp ý qua biểu mẫu bên cạnh")}</>,
    },
    {
      icon: Phone,
      title: t("Điện thoại"),
      children: <>{t("Thông tin sẽ được cập nhật")}</>,
    },
    {
      icon: Clock3,
      title: t("Giờ làm việc"),
      children: <>{t("Thứ 2 – Thứ 6: 8:00 – 17:00")}</>,
    },
  ];
  return (
    <>
      <PageHero
        title={t("LIÊN HỆ")}
        subtitle={t("Liên hệ với chúng tôi")}
        variant="contact"
      />
      <main id="main" className="container contact-main">
        <section className="contact-info">
          <SectionTitle>{t("Thông tin liên hệ")}</SectionTitle>
          <div className="contact-details">
            {details.map(({ icon: Icon, title, children }) => (
              <div className="contact-detail" key={title}>
                <span className="contact-icon">
                  <Icon size={22} />
                </span>
                <div>
                  <h3>{title}</h3>
                  <p>{children}</p>
                </div>
              </div>
            ))}
          </div>
          <div className="social-section">
            <SectionTitle>{t("Kết nối với chúng tôi")}</SectionTitle>
            <div className="social-links">
              <button
                aria-label="Facebook"
                onClick={() =>
                  notify({ text: t("Trang Facebook của dự án sẽ được cập nhật.") })
                }
              >
                <SocialMark provider="facebook" />
              </button>
              <button
                aria-label="YouTube"
                onClick={() =>
                  notify({ text: t("Kênh YouTube của dự án sẽ được cập nhật.") })
                }
              >
                <Play size={17} fill="currentColor" />
              </button>
              <button
                aria-label="GitHub"
                onClick={() =>
                  notify({ text: t("Kho mã nguồn của nhóm sẽ được cập nhật.") })
                }
              >
                <Github size={21} />
              </button>
              <button
                aria-label={t("Cộng đồng")}
                onClick={() =>
                  notify({ text: t("Cộng đồng MOVIE sẽ sớm được mở.") })
                }
              >
                <UsersRound size={21} />
              </button>
            </div>
          </div>
        </section>
        <section className="contact-form-section">
          <h2>{t("Gửi tin nhắn")}</h2>
          <p className="form-intro">{t("Chúng tôi luôn sẵn lòng lắng nghe bạn.")}</p>
          <form onSubmit={submit} className="contact-form">
            <label>
              {t("Họ và tên")} <span>*</span>
              <input
                name="name"
                autoComplete="name"
                placeholder={t("Nhập họ và tên")}
                required
                minLength={2}
                maxLength={100}
              />
            </label>
            <label>
              Email <span>*</span>
              <input
                name="email"
                type="email"
                autoComplete="email"
                placeholder={t("Nhập email của bạn")}
                required
                maxLength={254}
              />
            </label>
            <label>
              {t("Chủ đề")} <span>*</span>
              <span className="form-select">
                <select name="subject" required defaultValue="">
                  <option value="" disabled>
                    {t("Chọn chủ đề")}
                  </option>
                  <option>{t("Góp ý về trải nghiệm")}</option>
                  <option>{t("Hỗ trợ tài khoản")}</option>
                  <option>{t("Góp ý dự đoán phim")}</option>
                  <option>{t("Hợp tác & liên hệ khác")}</option>
                </select>
                <ChevronDown size={15} />
              </span>
            </label>
            <label>
              {t("Tin nhắn")} <span>*</span>
              <textarea
                name="message"
                placeholder={t("Nhập nội dung...")}
                required
                minLength={10}
                maxLength={5000}
                rows={5}
              />
            </label>
            {error && (
              <p className="form-error" role="alert">
                {error}
              </p>
            )}
            {done && (
              <p className="form-success" role="status">
                <CheckCircle2 size={16} /> {t("Tin nhắn đã được lưu trên hệ thống.")}
              </p>
            )}
            <button className="button full-width" disabled={busy}>
              {busy ? (
                <LoaderCircle className="spin" size={17} />
              ) : (
                <Send size={17} />
              )}
              {busy ? t("Đang gửi...") : t("Gửi tin nhắn")}
            </button>
          </form>
        </section>
      </main>
    </>
  );
}
