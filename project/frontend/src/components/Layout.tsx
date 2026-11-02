import { useEffect, useRef, useState } from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import {
  LogIn,
  LogOut,
  Menu,
  Moon,
  Search,
  Settings,
  Sun,
  UserRound,
  X,
} from "lucide-react";
import type { User } from "../lib/api";
import { Logo } from "./ui";
import { usePreferences } from "../lib/preferences";

const navigation = [
  ["/home", "Trang chủ"],
  ["/about", "Về chúng tôi"],
  ["/gallery", "Bộ sưu tập"],
  ["/genres", "Thể loại"],
  ["/watch-history", "Lịch sử đánh giá"],
  ["/experiments", "Thực nghiệm"],
  ["/contact", "Liên hệ"],
];

export function Header({
  user,
  onLogout,
  onSettings,
}: {
  user: User | null;
  onLogout: () => void;
  onSettings: () => void;
}) {
  const [mobile, setMobile] = useState(false);
  const [profile, setProfile] = useState(false);
  const profileRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const location = useLocation();
  const { t } = usePreferences();
  useEffect(() => {
    setMobile(false);
    setProfile(false);
  }, [location.pathname]);
  useEffect(() => {
    function dismiss(event: PointerEvent) {
      if (!profileRef.current?.contains(event.target as Node)) setProfile(false);
    }
    function escape(event: KeyboardEvent) {
      if (event.key === "Escape") { setProfile(false); setMobile(false); }
    }
    document.addEventListener("pointerdown", dismiss);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", dismiss);
      document.removeEventListener("keydown", escape);
    };
  }, []);
  return (
    <header className="site-header">
      <div className="container header-inner">
        <Logo />
        <nav
          className={`main-nav ${mobile ? "is-open" : ""}`}
          aria-label={t("Điều hướng chính")}
        >
          {navigation.map(([path, label]) => (
            <NavLink key={path} to={path}>
              {t(label)}
            </NavLink>
          ))}
        </nav>
        <div className="header-actions">
          <button
            className="icon-button red-icon"
            aria-label={t("Tìm kiếm phim")}
            title={t("Tìm kiếm phim (/)")}
            onClick={() => {
              const params = new URLSearchParams(location.pathname === "/gallery" ? location.search : "");
              params.set("focus", "search");
              navigate(`/gallery?${params}`, { replace: location.pathname === "/gallery" });
            }}
          >
            <Search size={22} />
          </button>
          <button
            className="icon-button red-icon"
            aria-label={t("Dự đoán và thuật toán")}
            title={t("Công nghệ và thuật toán")}
            onClick={onSettings}
          >
            <Settings size={22} />
          </button>
          <PreferenceControls />
          <div className="profile-wrap" ref={profileRef}>
            <button
              className={`icon-button ${user ? "user-connected" : ""}`}
              aria-label={t("Tài khoản")}
              aria-expanded={profile}
              onClick={() => setProfile(!profile)}
            >
              <UserRound size={21} />
            </button>
            {profile && (
              <div className="profile-menu">
                {user ? (
                  <>
                    <span className="profile-name">{user.name}</span>
                    <span className="muted profile-email">{user.email}</span>
                    <button
                      onClick={() => {
                        onLogout();
                        setProfile(false);
                      }}
                    >
                      <LogOut size={16} /> {t("Đăng xuất")}
                    </button>
                  </>
                ) : (
                  <>
                    <Link to="/login">
                      <LogIn size={16} /> {t("Đăng nhập")}
                    </Link>
                    <Link to="/register">
                      <UserRound size={16} /> {t("Tạo tài khoản")}
                    </Link>
                  </>
                )}
              </div>
            )}
          </div>
          <button
            className="icon-button mobile-toggle"
            onClick={() => setMobile(!mobile)}
            aria-label={mobile ? t("Đóng menu") : t("Mở menu")}
            aria-expanded={mobile}
          >
            {mobile ? <X /> : <Menu />}
          </button>
        </div>
      </div>
    </header>
  );
}

export function PreferenceControls({ className = "" }: { className?: string }) {
  const { language, theme, setLanguage, setTheme, t } = usePreferences();
  return <div className={`preference-controls ${className}`}>
    <button
      className="icon-button preference-toggle"
      aria-label={theme === "dark" ? t("Chuyển sang chế độ sáng") : t("Chuyển sang chế độ tối")}
      title={theme === "dark" ? t("Chế độ sáng") : t("Chế độ tối")}
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
    >
      {theme === "dark" ? <Sun size={19} /> : <Moon size={18} />}
    </button>
    <label className="language-control">
      <span className="sr-only">{t("Ngôn ngữ")}</span>
      <select aria-label={t("Ngôn ngữ")} value={language} onChange={(event) => setLanguage(event.target.value as "vi" | "en")}>
        <option value="vi">VI</option>
        <option value="en">EN</option>
      </select>
    </label>
  </div>;
}

export function Footer() {
  const { t } = usePreferences();
  return (
    <footer className="site-footer">
      <div className="container footer-inner">
        <div>
          <Logo small />
          <p>{t("Dự đoán phim · Trải nghiệm điện ảnh thông minh")}</p>
        </div>
        <div className="footer-right">
          <nav aria-label={t("Điều hướng cuối trang")}>
            {navigation.map(([path, label]) => (
              <Link key={path} to={path}>
                {t(label)}
              </Link>
            ))}
          </nav>
          <p>© {new Date().getFullYear()} Movie. {t("All rights reserved.")}</p>
        </div>
      </div>
    </footer>
  );
}
