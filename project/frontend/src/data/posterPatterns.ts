import type { Movie } from "./movies";

// Stable palettes make the same film recognizable across cards and detail pages.
const palettes = [
  { background: "#153d50", shade: "#071c31", primary: "#5de0cb", secondary: "#f0b67c" },
  { background: "#692a45", shade: "#2c153c", primary: "#ff977d", secondary: "#f8cf7d" },
  { background: "#354471", shade: "#161e43", primary: "#b4b0ff", secondary: "#f8b8b0" },
  { background: "#126052", shade: "#0b2d37", primary: "#d2e77d", secondary: "#77d8c5" },
  { background: "#a54e32", shade: "#4c2335", primary: "#ffda8f", secondary: "#f9a0b3" },
  { background: "#35457f", shade: "#1c2250", primary: "#79d7ff", secondary: "#e3adff" },
  { background: "#785026", shade: "#352440", primary: "#f9d47e", secondary: "#df9cba" },
  { background: "#643b7b", shade: "#2c2053", primary: "#e6adff", secondary: "#91e6dc" },
];

export function posterPattern(movie: Movie) {
  const seed = (Math.imul(movie.id, 2654435761) >>> 0);
  return {
    ...palettes[seed % palettes.length],
    variant: Math.floor(seed / 8) % 6,
    rotation: (seed % 5 - 2) * 9,
  };
}
