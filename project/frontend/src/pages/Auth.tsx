import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import {
  Eye,
  EyeOff,
  LoaderCircle,
  LockKeyhole,
  Mail,
  UserRound,
} from "lucide-react";
import { Logo, SocialMark } from "../components/ui";
import { api, type User, type DemoAccounts, type DemoAccount } from "../lib/api";
import type { Notice } from "../types";
import { safeReturnPath } from "../lib/viewer";
import { usePreferences } from "../lib/preferences";

export function Auth({
  register = false,
  onLogin,
  notify,
}: {
  register?: boolean;
  onLogin: (user: User) => void;
  notify: (notice: Notice) => void;
}) {
  const { t } = usePreferences();
  const [passwordShown, setPasswordShown] = useState(false);
  const [confirmationShown, setConfirmationShown] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [demos, setDemos] = useState<DemoAccounts | null>(null);
  const [demoError, setDemoError] = useState("");
  const [demoAttempt, setDemoAttempt] = useState(0);
  const emailInput = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const destination = safeReturnPath(params.get("next"));
  useEffect(() => {
    setError("");
    setPasswordShown(false);
    setConfirmationShown(false);
  }, [register]);
  useEffect(() => {
    if (register) return;
    const controller = new AbortController();
    setDemoError("");
    api<DemoAccounts>("/auth/demo-accounts", { signal: controller.signal })
      .then((value) => { if (!controller.signal.aborted) setDemos(value); })
      .catch((cause: Error) => { if (!controller.signal.aborted) setDemoError(cause.message); });
    return () => controller.abort();
  }, [register, demoAttempt]);
  function chooseDemo(account: DemoAccount) {
    setEmail(account.email); setPassword(account.password); setError("");
    emailInput.current?.focus();
    notify({ text: t("Đã điền tài khoản MovieLens User {id}. Nhấn Đăng nhập để tiếp tục.", { id: account.dataset_user_id }) });
  }
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    const data = Object.fromEntries(new FormData(event.currentTarget));
    if (register && data.password !== data.confirmation) {
      setError(t("Mật khẩu xác nhận chưa khớp."));
      return;
    }
    setBusy(true);
    try {
      const user = await api<User>(`/auth/${register ? "register" : "login"}`, {
        method: "POST",
        body: JSON.stringify(data),
      });
      onLogin(user);
      notify({
        text: register
          ? t("Tạo tài khoản thành công. Chào mừng bạn đến với MOVIE!")
          : t("Chào mừng {name} trở lại!", { name: user.name }),
        kind: "success",
      });
      navigate(destination, { replace: true });
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <main id="main" className={`auth-stage ${register ? "auth-register" : "auth-with-demos"}`}>
      <div className="container auth-layout">
        <div className="auth-image-caption">
          <span className="tiny-line" />
          <p>
            {t("MỘT BỘ PHIM HAY.")}
            <br />
            {t("MỘT TRẢI NGHIỆM KHÓ QUÊN.")}
          </p>
        </div>
        <section
          className="auth-panel"
          aria-label={register ? t("Đăng ký tài khoản") : t("Đăng nhập tài khoản")}
        >
          <Logo />
          <h1>{register ? t("Đăng ký") : t("Đăng nhập")}</h1>
          <p className="auth-subtitle">
            {register
              ? t("Tạo tài khoản để nhận gợi ý phim cá nhân hóa")
              : t("Chào mừng bạn trở lại! Hãy đăng nhập để tiếp tục.")}
          </p>
          <form
            key={register ? "register" : "login"}
            onSubmit={submit}
            className="auth-form"
          >
            {register && (
              <label className="auth-input">
                <UserRound size={19} />
                <input
                  name="name"
                  aria-label={t("Họ và tên")}
                  placeholder={t("Họ và tên")}
                  autoComplete="name"
                  required
                  minLength={2}
                  maxLength={100}
                />
              </label>
            )}
            <label className="auth-input">
              <Mail size={19} />
              <input
                name="email"
                ref={emailInput}
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                type="email"
                aria-label="Email"
                placeholder={register ? "Email" : t("Email của bạn")}
                autoComplete="email"
                required
                maxLength={254}
              />
            </label>
            <label className="auth-input">
              <LockKeyhole size={19} />
              <input
                name="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                type={passwordShown ? "text" : "password"}
                aria-label={t("Mật khẩu")}
                placeholder={t("Mật khẩu")}
                autoComplete={register ? "new-password" : "current-password"}
                required
                minLength={register ? 8 : 1}
                maxLength={128}
              />
              <button
                type="button"
                className="password-toggle"
                aria-label={passwordShown ? t("Ẩn mật khẩu") : t("Hiện mật khẩu")}
                onClick={() => setPasswordShown(!passwordShown)}
              >
                {passwordShown ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </label>
            {register && (
              <label className="auth-input">
                <LockKeyhole size={19} />
                <input
                  name="confirmation"
                  type={confirmationShown ? "text" : "password"}
                  aria-label={t("Xác nhận mật khẩu")}
                  placeholder={t("Xác nhận mật khẩu")}
                  autoComplete="new-password"
                  required
                  minLength={8}
                  maxLength={128}
                />
                <button
                  type="button"
                  className="password-toggle"
                  aria-label={
                    confirmationShown
                      ? t("Ẩn xác nhận mật khẩu")
                      : t("Hiện xác nhận mật khẩu")
                  }
                  onClick={() => setConfirmationShown(!confirmationShown)}
                >
                  {confirmationShown ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </label>
            )}
            {register && (
              <p className="password-hint">
                {t("Sử dụng ít nhất 8 ký tự cho mật khẩu của bạn.")}
              </p>
            )}
            {error && (
              <p className="form-error" role="alert">
                {error}
              </p>
            )}
            <button className="button full-width auth-submit" disabled={busy}>
              {busy && <LoaderCircle size={17} className="spin" />}
              {busy ? t("Vui lòng chờ...") : register ? t("Đăng ký") : t("Đăng nhập")}
            </button>
          </form>
          <div className="auth-divider">
            <span>{t("Hoặc {action} với", { action: register ? t("đăng ký") : t("đăng nhập") })}</span>
          </div>
          <div className="auth-social">
            <button
              className="google-button"
              aria-label={t("Tiếp tục với Google")}
              onClick={() =>
                notify({
                  text: t("Đăng nhập Google chưa được kết nối. Vui lòng sử dụng email."),
                })
              }
            >
              <SocialMark provider="google" />
            </button>
            <button
              className="facebook-button"
              aria-label={t("Tiếp tục với Facebook")}
              onClick={() =>
                notify({
                  text: t("Đăng nhập Facebook chưa được kết nối. Vui lòng sử dụng email."),
                })
              }
            >
              <SocialMark provider="facebook" />
            </button>
          </div>
          <p className="auth-switch">
            {register ? t("Đã có tài khoản?") : t("Chưa có tài khoản?")}{" "}
            <Link to={`${register ? "/login" : "/register"}${params.get("next") ? `?next=${encodeURIComponent(destination)}` : ""}`}>
              {register ? t("Đăng nhập") : t("Đăng ký ngay")}
            </Link>
          </p>
        </section>
        {!register && <section className="auth-demo-panel" aria-labelledby="demo-heading">
          <span className="eyebrow">{t("KHÁM PHÁ SỞ THÍCH CÓ SẴN")}</span>
          <h2 id="demo-heading">{t("Thử hồ sơ MovieLens")}</h2>
          <p>{t("Chọn một trong {count} người dùng từ dataset để xem lịch sử và nhận gợi ý theo những phim đã chấm điểm.", { count: demos?.accounts.length ?? 5 })}</p>
          {demoError ? <div role="alert"><p>{demoError}</p><button className="rail-link" onClick={() => setDemoAttempt(demoAttempt + 1)}>{t("Tải lại tài khoản demo")}</button></div>
            : !demos ? <p role="status"><LoaderCircle size={15} className="spin" /> {t("Đang chuẩn bị hồ sơ từ dataset...")}</p>
            : <><div className="demo-accounts">{demos.accounts.map((account, index) =>
              <article className={`demo-account ${email === account.email ? "is-selected" : ""}`} key={account.email}>
                <div className="demo-avatar" aria-hidden="true">0{index + 1}</div>
                <div className="demo-account-content"><div className="demo-account-heading"><h3>User {account.dataset_user_id}</h3><span>{account.imported_rating_count.toLocaleString("vi-VN")} {t("đánh giá")}</span></div>
                  <dl><div><dt>{t("Email")}</dt><dd>{account.email}</dd></div><div><dt>{t("Mật khẩu")}</dt><dd>{account.password}</dd></div></dl>
                  <button type="button" disabled={busy} onClick={() => chooseDemo(account)}>{email === account.email ? t("Đã chọn · Điền lại") : t("Dùng tài khoản này")}</button>
                </div>
              </article>)}</div>
              <p className="demo-disclaimer">{t(demos.notice)} {t("Đây là các hồ sơ dùng chung cho thực nghiệm.")}</p></>}
          <div className="demo-new-user"><UserRound size={20} /><div><strong>{t("Muốn thử tài khoản mới?")}</strong><p>{t("Đăng ký để bắt đầu với lịch sử trống. Gợi ý sẽ thay đổi sau khi bạn đánh giá phim.")}</p><Link to={`/register${params.get("next") ? `?next=${encodeURIComponent(destination)}` : ""}`}>{t("Tạo hồ sơ của bạn →")}</Link></div></div>
        </section>}
      </div>
    </main>
  );
}
