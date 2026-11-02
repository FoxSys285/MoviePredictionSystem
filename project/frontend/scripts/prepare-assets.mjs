import { mkdir, access } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const source = new URL("../../../images/generated/", import.meta.url);
const destination = new URL("../public/assets/", import.meta.url);
const posters = [
  "inception",
  "interstellar",
  "dark-knight",
  "fight-club",
  "martian",
  "godfather",
  "joker",
  "batman-superman",
  "justice-league",
  "lord-rings",
  "avengers",
  "dune",
];
await mkdir(new URL("posters/", destination), { recursive: true });
await mkdir(new URL("team/", destination), { recursive: true });
for (const name of [
  "cinema",
  "theater",
  "contact",
  "about",
  ...posters.map((name) => `posters/${name}`),
  ...["frontend", "backend", "ml", "design"].map((name) => `team/${name}`),
]) {
  const input = fileURLToPath(new URL(`${name}.png`, source));
  await access(input);
  const output = fileURLToPath(new URL(`${name}.webp`, destination));
  await sharp(input)
    .resize({
      width: name.startsWith("posters/")
        ? 480
        : name.startsWith("team/")
          ? 160
          : 1920,
      withoutEnlargement: true,
    })
    .webp({ quality: 86, effort: 6 })
    .toFile(output);
  console.log(`Prepared ${name}.webp`);
}
