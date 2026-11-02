import { useEffect, useState } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { ArrowRight, BookOpen, CheckCircle2, Film, X } from "lucide-react";
import { Header, Footer, PreferenceControls } from "./components/Layout";
import { MovieDetail } from "./components/MovieDetail";
import { ModelPanel } from "./components/ModelPanel";
import { RouteFrame, ScrollExperience } from "./components/Experience";
import { Home } from "./pages/Home";
import { Gallery } from "./pages/Gallery";
import { About } from "./pages/About";
import { Contact } from "./pages/Contact";
import { Auth } from "./pages/Auth";
import { WatchHistory } from "./pages/WatchHistory";
import { Genres } from "./pages/Genres";
import { Experiments } from "./pages/Experiments";
import { ViewerContext } from "./lib/viewer";
import { api, type Dataset, type User } from "./lib/api";
import { PreferencesProvider, usePreferences } from "./lib/preferences";
import type { Notice } from "./types";

export default function App() {
  return <PreferencesProvider><AppContent /></PreferencesProvider>;
}

function AppContent() {
  const [user, setUser] = useState<User | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [ratingsVersion, setRatingsVersion] = useState(0);
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [modelsOpen, setModelsOpen] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);
  const location = useLocation();
  const { language, t } = usePreferences();
  const authPage = ["/login", "/register"].includes(location.pathname);
  useEffect(() => {
    const controller = new AbortController();
    api<Dataset>("/dataset", { signal: controller.signal })
      .then(setDataset)
      .catch(() => {});
    api<User>("/auth/me")
      .then((value) => { if (!controller.signal.aborted) setUser(value); })
      .catch(() => {})
      .finally(() => { if (!controller.signal.aborted) setAuthLoading(false); });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    const titles: Record<string, string> = {
      "/gallery": t("Khám phá bộ sưu tập MovieLens"),
      "/home": t("Điện ảnh dành cho bạn"),
      "/about": t("Về chúng tôi"),
      "/contact": t("Liên hệ với chúng tôi"),
      "/login": t("Đăng nhập"),
      "/register": t("Đăng ký"),
      "/genres": t("Tất cả thể loại"),
      "/experiments": t("Thực nghiệm so sánh mô hình"),
      "/watch-history": t("Những phim bạn đã đánh giá"),
    };
    document.title = `MOVIE — ${titles[location.pathname] || t("Khám phá thế giới phim ảnh")}`;
  }, [location.pathname, language, t]);
  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 5500);
    return () => window.clearTimeout(timer);
  }, [notice]);
  async function logout() {
    try {
      await api("/auth/logout", { method: "POST" });
      setUser(null);
      setNotice({ text: t("Bạn đã đăng xuất.") });
    } catch (cause) {
      setNotice({ text: (cause as Error).message });
    }
  }
  return (
    <ViewerContext.Provider value={{ user, authLoading, ratingsVersion,
      onRatingsChanged: () => setRatingsVersion((value) => value + 1),
      expireSession: () => setUser(null), notify: setNotice }}>
      <a className="skip-link" href="#main">
        {t("Đến nội dung chính")}
      </a>
      <div className={`app-shell ${authPage ? "auth-shell" : ""}`}>
        {authPage && <PreferenceControls className="auth-preference-controls" />}
        {!authPage && (
          <Header
            user={user}
            onLogout={logout}
            onSettings={() => setModelsOpen(true)}
          />
        )}
        <RouteFrame key={location.pathname}>
        <Routes>
          <Route path="/" element={<Navigate to="/gallery" replace />} />
          <Route
            path="/home"
            element={
              <Home
                dataset={dataset}
                onSettings={() => setModelsOpen(true)}
              />
            }
          />
          <Route path="/gallery" element={<Gallery dataset={dataset} />} />
          <Route path="/galary" element={<Navigate to="/gallery" replace />} />
          <Route path="/genres" element={<Genres dataset={dataset} />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/watch-history" element={<WatchHistory />} />
          <Route path="/movies/:movieId" element={<MovieDetail />} />
          <Route path="/about" element={<About dataset={dataset} />} />
          <Route path="/contact" element={<Contact notify={setNotice} />} />
          <Route
            path="/login"
            element={<Auth onLogin={setUser} notify={setNotice} />}
          />
          <Route
            path="/register"
            element={<Auth register onLogin={setUser} notify={setNotice} />}
          />
          <Route
            path="*"
            element={
              <main id="main" className="container not-found">
                <Film size={50} />
                <h1>{t("CẢNH PHIM NÀY CHƯA CÓ.")}</h1>
                <p>
                  {t("Trang bạn tìm có thể đã được chuyển sang một câu chuyện khác.")}
                </p>
                <Link className="button" to="/gallery">
                  {t("Trở lại Gallery")} <ArrowRight size={17} />
                </Link>
              </main>
            }
          />
        </Routes>
        </RouteFrame>
        <Footer />
      </div>
      <ScrollExperience />
      {modelsOpen && <ModelPanel onClose={() => setModelsOpen(false)} />}
      {notice && (
        <div
          className={`toast ${notice.kind === "success" ? "toast-success" : ""}`}
          role="status"
        >
          {notice.kind === "success" ? (
            <CheckCircle2 size={20} />
          ) : (
            <BookOpen size={20} />
          )}
          <span>{notice.text}</span>
          <button onClick={() => setNotice(null)} aria-label={t("Đóng thông báo")}>
            <X size={17} />
          </button>
        </div>
      )}
    </ViewerContext.Provider>
  );
}
