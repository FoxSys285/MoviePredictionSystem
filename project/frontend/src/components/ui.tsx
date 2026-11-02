import { useEffect, useRef, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { ChevronDown, X } from "lucide-react";
import { usePreferences } from "../lib/preferences";

export function Logo({ small = false }: { small?: boolean }) {
  const { t } = usePreferences();
  return (
    <Link
      className={`logo ${small ? "logo-small" : ""}`}
      to="/home"
      aria-label={`MOVIE — ${t("Trang chủ")}`}
    >
      MOVIE
    </Link>
  );
}

export function SectionTitle({ children }: { children: ReactNode }) {
  return <h2 className="section-title">{children}</h2>;
}

export function PageHero({
  title,
  subtitle,
  variant = "cinema",
  align = "center",
}: {
  title: string;
  subtitle: string;
  variant?: string;
  align?: "center" | "left";
}) {
  return (
    <section className={`page-hero hero-${variant} align-${align}`}>
      <div className="hero-shade" />
      <div className="container hero-copy">
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
    </section>
  );
}

export function Dialog({
  children,
  onClose,
  title,
  className = "",
}: {
  children: ReactNode;
  onClose: () => void;
  title: string;
  className?: string;
}) {
  const { t } = usePreferences();
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    dialog?.showModal();
    const old = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      dialog?.close();
      document.body.style.overflow = old;
    };
  }, []);
  return (
    <dialog
      ref={ref}
      aria-label={title}
      className={`dialog ${className}`}
      onCancel={onClose}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <button
        className="dialog-close icon-button"
        onClick={onClose}
        aria-label={t("Đóng")}
      >
        <X size={21} />
      </button>
      {children}
    </dialog>
  );
}

export function Select({
  label,
  value,
  onChange,
  children,
  className = "",
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  children: ReactNode;
  className?: string;
}) {
  return (
    <label className={`select-field ${className}`}>
      <select
        aria-label={label}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {children}
      </select>
      <ChevronDown size={14} />
    </label>
  );
}

export function SocialMark({ provider }: { provider: "google" | "facebook" }) {
  return (
    <span className={`social-mark ${provider}`} aria-hidden="true">
      {provider === "google" ? "G" : "f"}
    </span>
  );
}
