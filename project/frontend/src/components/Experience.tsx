import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { NavLink, useLocation, useNavigate, useNavigationType } from "react-router-dom";
import { ArrowUp, FlaskConical } from "lucide-react";
import { observeReveals, reducedMotion } from "../lib/motion";
import { usePreferences } from "../lib/preferences";

export function RouteFrame({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => ref.current ? observeReveals(ref.current) : undefined, []);
  return <div ref={ref} className="route-frame">{children}</div>;
}

export function ScrollExperience() {
  const { t } = usePreferences();
  const location = useLocation();
  const navigation = useNavigationType();
  const navigate = useNavigate();
  const [visible, setVisible] = useState(false);
  const positions = useRef(new Map<string, number>());
  const previousPath = useRef(location.pathname);
  const currentY = useRef(window.scrollY);

  useLayoutEffect(() => {
    if (previousPath.current !== location.pathname) {
      positions.current.set(previousPath.current, currentY.current);
    }
    previousPath.current = location.pathname;
    const target = navigation === "POP" ? positions.current.get(location.pathname) || 0 : 0;
    let frame = 0;
    let tries = 0;
    window.scrollTo({ top: target, behavior: "instant" });
    currentY.current = target;
    // Async galleries may need a frame to regain enough height for Back restoration.
    function restore() {
      window.scrollTo({ top: target, behavior: "instant" });
      if (Math.abs(window.scrollY - target) > 2 && ++tries < 90) {
        frame = requestAnimationFrame(restore);
      }
    }
    if (target > 0) frame = requestAnimationFrame(restore);
    const cancel = () => cancelAnimationFrame(frame);
    window.addEventListener("wheel", cancel, { passive: true, once: true });
    window.addEventListener("touchstart", cancel, { passive: true, once: true });
    return () => {
      cancel();
      window.removeEventListener("wheel", cancel);
      window.removeEventListener("touchstart", cancel);
    };
    // Search/filter updates should keep the current viewport.
  }, [location.pathname]);

  useEffect(() => {
    const previousRestoration = window.history.scrollRestoration;
    window.history.scrollRestoration = "manual";
    let frame = 0;
    const update = () => {
      frame = 0;
      const top = Math.max(0, window.scrollY);
      const distance = document.documentElement.scrollHeight - window.innerHeight;
      currentY.current = top;
      document.documentElement.style.setProperty("--scroll-progress", String(distance > 0 ? Math.min(1, top / distance) : 0));
      document.documentElement.toggleAttribute("data-scrolled", top > 24);
      setVisible(top > 500);
    };
    const schedule = () => { if (!frame) frame = requestAnimationFrame(update); };
    update();
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    const resize = new ResizeObserver(schedule);
    resize.observe(document.body);
    return () => {
      window.history.scrollRestoration = previousRestoration;
      cancelAnimationFrame(frame);
      resize.disconnect();
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
    };
  }, []);

  useEffect(() => {
    function searchShortcut(event: KeyboardEvent) {
      const target = event.target;
      if (event.key !== "/" || event.ctrlKey || event.altKey || event.metaKey || event.defaultPrevented) return;
      if (document.querySelector("dialog[open]")) return;
      if (target instanceof HTMLElement && (target.isContentEditable || target.closest("input, textarea, select"))) return;
      event.preventDefault();
      const params = new URLSearchParams(location.pathname === "/gallery" ? location.search : "");
      params.set("focus", "search");
      navigate(`/gallery?${params}`, { replace: location.pathname === "/gallery" });
    }
    window.addEventListener("keydown", searchShortcut);
    return () => window.removeEventListener("keydown", searchShortcut);
  }, [location.pathname, location.search, navigate]);

  return <>
    <div className="scroll-progress" aria-hidden="true" />
    <NavLink className="experiments-fab" to="/experiments" aria-label={t("Mở trang thực nghiệm so sánh")}>
      <FlaskConical size={22} aria-hidden="true" />
      <span className="experiments-fab-tooltip">{t("Thực nghiệm so sánh")}</span>
    </NavLink>
    <button className={`back-to-top ${visible ? "is-visible" : ""}`}
      aria-label={t("Lên đầu trang")} title={t("Lên đầu trang")} tabIndex={visible ? 0 : -1}
      aria-hidden={!visible}
      onClick={() => window.scrollTo({ top: 0, behavior: reducedMotion() ? "instant" : "smooth" })}>
      <ArrowUp size={20} />
    </button>
  </>;
}
