import { useEffect, useRef, type PointerEvent, type RefObject } from "react";

export const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export function scrollToElement(element: Element | null) {
  element?.scrollIntoView({ behavior: reducedMotion() ? "instant" : "smooth", block: "start" });
}

// One observer handles both initial content and newly fetched movie cards.
export function observeReveals(root: HTMLElement) {
  if (reducedMotion() || !("IntersectionObserver" in window)) return () => {};
  const selector = ".movie-card, .section-title, .feature, .team-card, .movie-insight-panel, .dataset-stats > div, .theater-photo, .contact-form";
  const intersection = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      (entry.target as HTMLElement).dataset.reveal = "visible";
      intersection.unobserve(entry.target);
    }
  }, { threshold: 0.08, rootMargin: "0px 0px 30px 0px" });
  function observe() {
    root.querySelectorAll<HTMLElement>(selector).forEach((element) => {
      if (element.dataset.reveal) return;
      // Content already above the viewport should stay visible when navigating Back.
      if (element.getBoundingClientRect().bottom < 0) return;
      element.dataset.reveal = "pending";
      intersection.observe(element);
    });
  }
  observe();
  const mutation = new MutationObserver((entries) => {
    if (entries.some((entry) => entry.addedNodes.length)) observe();
  });
  mutation.observe(root, { childList: true, subtree: true });
  const media = window.matchMedia("(prefers-reduced-motion: reduce)");
  const showAll = () => {
    if (media.matches) root.querySelectorAll<HTMLElement>("[data-reveal]").forEach((element) => {
      element.dataset.reveal = "visible";
    });
  };
  media.addEventListener("change", showAll);
  return () => {
    intersection.disconnect();
    mutation.disconnect();
    media.removeEventListener("change", showAll);
    root.querySelectorAll<HTMLElement>("[data-reveal]").forEach((element) => delete element.dataset.reveal);
  };
}

export function useCardMotion<T extends HTMLElement>(): {
  ref: RefObject<T | null>;
  onPointerMove: (event: PointerEvent<T>) => void;
  onPointerLeave: () => void;
} {
  const ref = useRef<T>(null);
  const frame = useRef(0);
  const pointer = useRef({ x: 0, y: 0 });
  useEffect(() => () => cancelAnimationFrame(frame.current), []);
  return {
    ref,
    onPointerMove(event) {
      if (event.pointerType !== "mouse" || reducedMotion()) return;
      pointer.current = { x: event.clientX, y: event.clientY };
      if (frame.current) return;
      frame.current = requestAnimationFrame(() => {
        frame.current = 0;
        const element = ref.current;
        if (!element) return;
        const bounds = element.getBoundingClientRect();
        const x = Math.min(1, Math.max(0, (pointer.current.x - bounds.left) / bounds.width));
        const y = Math.min(1, Math.max(0, (pointer.current.y - bounds.top) / bounds.height));
        element.style.setProperty("--pointer-x", `${x * 100}%`);
        element.style.setProperty("--pointer-y", `${y * 100}%`);
        element.style.setProperty("--tilt-x", `${(0.5 - y) * 5}deg`);
        element.style.setProperty("--tilt-y", `${(x - 0.5) * 5}deg`);
      });
    },
    onPointerLeave() {
      cancelAnimationFrame(frame.current);
      frame.current = 0;
      ref.current?.style.setProperty("--tilt-x", "0deg");
      ref.current?.style.setProperty("--tilt-y", "0deg");
    },
  };
}
