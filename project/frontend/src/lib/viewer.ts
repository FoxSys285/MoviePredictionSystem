import { createContext, useContext } from "react";
import type { User } from "./api";
import type { Notice } from "../types";

export const ViewerContext = createContext<{
  user: User | null;
  authLoading: boolean;
  ratingsVersion: number;
  onRatingsChanged: () => void;
  expireSession: () => void;
  notify: (notice: Notice) => void;
}>({ user: null, authLoading: true, ratingsVersion: 0,
  onRatingsChanged: () => {}, expireSession: () => {}, notify: () => {} });

export const useViewer = () => useContext(ViewerContext);

export function safeReturnPath(value: string | null) {
  if (!value || !value.startsWith("/") || value.startsWith("//") || /[\\\r\n]/.test(value)) return "/gallery";
  const pathname = value.split(/[?#]/)[0];
  return ["/login", "/register"].includes(pathname) ? "/gallery" : value;
}
