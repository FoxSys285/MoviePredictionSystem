import { useId } from "react";
import { genreLabels, type Movie } from "../data/movies";
import { posterPattern } from "../data/posterPatterns";
import { usePreferences } from "../lib/preferences";

export function PosterPattern({ movie }: { movie: Movie }) {
  const { t } = usePreferences();
  const id = useId().replace(/:/g, "");
  const art = posterPattern(movie);
  const rings = [0, 1, 2, 3, 4, 5, 6];
  return (
    <div className={`poster-pattern pattern-${art.variant}`} aria-hidden="true">
      <svg viewBox="0 0 400 560" preserveAspectRatio="xMidYMid slice" focusable="false">
        <defs>
          <linearGradient id={`${id}-ground`} x2=".8" y2="1">
            <stop stopColor={art.background} /><stop offset="1" stopColor={art.shade} />
          </linearGradient>
          <linearGradient id={`${id}-light`} x2=".9" y2="1">
            <stop stopColor={art.primary} /><stop offset="1" stopColor={art.secondary} />
          </linearGradient>
          <pattern id={`${id}-dots`} width="12" height="12" patternUnits="userSpaceOnUse">
            <circle cx="1" cy="1" r=".65" fill="#fff" opacity=".16" />
          </pattern>
          <clipPath id={`${id}-disc`}><circle cx="200" cy="248" r="142" /></clipPath>
        </defs>
        <rect width="400" height="560" fill={`url(#${id}-ground)`} />
        <path d="M-70 520 410-40M-10 600 470 40" stroke={art.primary} strokeWidth="80" opacity=".05" />
        <g className="pattern-motif">
          <g transform={`rotate(${art.rotation} 200 248)`}>
            {art.variant === 0 && <>
              <circle cx="200" cy="245" r="118" fill={`url(#${id}-light)`} />
              <circle cx="230" cy="212" r="103" fill={art.background} />
              <g fill="none" stroke={art.primary}>
                {rings.map((n) => <ellipse key={n} cx="200" cy="250" rx={164 - n * 14} ry={44 + n * 13} transform="rotate(-34 200 250)" opacity={.75 - n * .07} />)}
              </g>
              <circle cx="78" cy="328" r="13" fill={art.secondary} />
              <circle cx="324" cy="165" r="5" fill={art.primary} />
            </>}
            {art.variant === 1 && <>
              <circle cx="262" cy="160" r="74" fill={art.secondary} opacity=".9" />
              <g transform="rotate(-28 200 250)">
                {rings.map((n) => <rect key={n} x={39 + n * 15} y={91 + n * 19} width={290 - n * 30} height={322 - n * 38} rx="72" fill="none" stroke={n % 2 ? art.secondary : art.primary} strokeWidth="9" opacity={1 - n * .07} />)}
              </g>
              <path d="M35 378H190" stroke={art.secondary} strokeWidth="2" />
            </>}
            {art.variant === 2 && <>
              <circle cx="200" cy="248" r="142" fill={`url(#${id}-light)`} />
              <g clipPath={`url(#${id}-disc)`} fill="none" stroke={art.shade} strokeWidth="13">
                {Array.from({ length: 15 }, (_, n) => <path key={n} d={`M-20 ${100 + n * 22} Q100 ${-20 + n * 22} 200 ${155 + n * 22} T440 ${110 + n * 22}`} />)}
              </g>
              <circle cx="322" cy="363" r="20" fill={art.secondary} />
              <circle cx="65" cy="119" r="6" fill={art.primary} />
            </>}
            {art.variant === 3 && <>
              <g transform="translate(200 248)">
                {Array.from({ length: 12 }, (_, n) => <ellipse key={n} cx="0" cy="-64" rx="33" ry="93" transform={`rotate(${n * 30})`} fill="none" stroke={n % 2 ? art.secondary : art.primary} strokeWidth="3" opacity=".85" />)}
                <circle r="46" fill={`url(#${id}-light)`} />
                <circle r="24" fill={art.shade} />
              </g>
              <path d="M44 88h48M68 64v48M310 365h36M328 347v36" stroke={art.primary} strokeWidth="2" />
            </>}
            {art.variant === 4 && <>
              <g transform="rotate(-30 200 248)">
                {Array.from({ length: 9 }, (_, n) => <rect key={n} x={19 + n * 39} y={115 + (n % 3) * 19} width="24" height={260 - (n % 3) * 38} rx="12" fill={n % 2 ? art.secondary : art.primary} opacity={.5 + n * .05} />)}
              </g>
              <circle cx="200" cy="248" r="118" fill="none" stroke={art.secondary} strokeWidth="1.5" />
              <circle cx="200" cy="248" r="153" fill="none" stroke={art.primary} strokeDasharray="2 9" opacity=".7" />
            </>}
            {art.variant === 5 && <>
              <circle cx="200" cy="248" r="145" fill={art.primary} opacity=".12" />
              {rings.map((n) => <path key={n} d={`M${38 + n * 24} 350 L200 ${79 + n * 27} L${362 - n * 24} 350 Z`} fill="none" stroke={n % 2 ? art.secondary : art.primary} strokeWidth="7" strokeLinejoin="round" opacity={1 - n * .08} />)}
              <circle cx="315" cy="126" r="15" fill={art.secondary} />
            </>}
          </g>
        </g>
        <rect width="400" height="560" fill={`url(#${id}-dots)`} />
        <path d="M24 58V24h34M342 24h34v34M24 490v46h34M342 536h34v-46" fill="none" stroke={art.primary} opacity=".4" />
      </svg>
      <div className="pattern-topline"><span>MOVIE<span className="pattern-edition"> COLLECTION</span></span><span>{movie.year || "CINEMA"}</span></div>
      <div className="pattern-footer"><span>{t(genreLabels[movie.genres[0]] || movie.genres[0])}</span><span className="pattern-bars" /></div>
    </div>
  );
}
