import assert from "node:assert/strict";
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { chromium } from "playwright";

const frontend = fileURLToPath(new URL("../", import.meta.url));
const project = path.resolve(frontend, "..");
const output = path.join(frontend, "test-results");
const testData = await mkdtemp(path.join(tmpdir(), "movie-ui-test-"));
const dataset = path.join(testData, "dataset");
const base = "http://127.0.0.1:5175";
const backendUrl = "http://127.0.0.1:8137";
const candidates = [
  process.env.CHROME_PATH,
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
].filter(Boolean);
let executablePath;
for (const candidate of candidates) {
  try {
    await access(candidate);
    executablePath = candidate;
    break;
  } catch {}
}
if (!executablePath)
  throw new Error("Set CHROME_PATH to an installed Chrome/Edge executable.");
await mkdir(output, { recursive: true });
const children = [];
let browser;
const logs = [];
function launch(command, args, cwd, env) {
  const child = spawn(command, args, {
    cwd,
    env: { ...process.env, ...env },
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  children.push(child);
  child.stdout.on("data", (data) => logs.push(data.toString()));
  child.stderr.on("data", (data) => logs.push(data.toString()));
  return child;
}
async function ready(url) {
  for (let index = 0; index < 80; index++) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error(`Server not ready: ${url}\n${logs.join("").slice(-5000)}`);
}
async function expectText(page, text) {
  await page.getByText(text, { exact: false }).first().waitFor();
}
async function prepareScreenshot(page, route) {
  if (route === "login") await page.locator(".demo-account").last().waitFor();
  if (route === "gallery") await expectText(page, "16 bộ phim · MovieLens");
  if (route === "gallery" || route === "home") await page.locator('.recommendation-section:not([data-mode="loading"])').waitFor();
  if (route === "home") await page.waitForFunction(() => document.querySelectorAll(".home-genre-shelves .movie-rail-section").length === 5);
  if (route === "genres") await page.locator(".genre-tile").first().waitFor();
  if (route === "experiments") await expectText(page, "Chưa có kết quả thực nghiệm");
  if (route === "watch-history") {
    await page.locator(".history-gate, .history-heading").first().waitFor();
    await page.locator('.history-main [role="status"]').waitFor({ state: "hidden" });
  }
  if (route.startsWith("movies/")) await page.locator(".movie-overview").waitFor();
  await page.evaluate(() => document.fonts.ready);
  await page.locator("img").evaluateAll((images) =>
    Promise.all(
      images.map((image) => {
        image.loading = "eager";
        return image.decode();
      }),
    ),
  );
}
try {
  await mkdir(dataset);
  const fixtureMovies = [
    [79132, "Inception (2010)", "Action|Sci-Fi"],
    [109487, "Interstellar (2014)", "Sci-Fi|Adventure"],
    [858, "The Godfather (1972)", "Crime|Drama"],
    [2959, "Fight Club (1999)", "Drama|Thriller"],
    [1, "Toy Story (1995)", "Animation|Children|Comedy"],
    [2, "Jumanji (1995)", "Adventure|Children|Fantasy"],
    [58559, "The Dark Knight (2008)", "Action|Crime|Drama"],
    [134130, "The Martian (2015)", "Sci-Fi|Adventure"],
    [4993, "The Fellowship of the Ring (2001)", "Adventure|Fantasy"],
    [122912, "Avengers: Infinity War (2018)", "Action|Sci-Fi"],
    [136864, "Batman v Superman (2016)", "Action|Sci-Fi"],
    [3, "A Horror Film (2000)", "Horror"],
    [4, "A Documentary (2019)", "Documentary"],
    [5, "A Romance (2004)", "Romance"],
    [6, "A Mystery (2012)", "Mystery|Drama"],
    [999999, "Unrated Film", "(no genres listed)"],
  ];
  await writeFile(path.join(dataset, "movies.csv"), "movieId,title,genres\n" + fixtureMovies.map((row) => row.join(",")).join("\n") + "\n");
  const ratings = fixtureMovies.slice(0, -1).flatMap(([id], index) =>
    [1, 2, 3, 4].map((user) => `${user},${id},${3 + user / 2},${1500000000 + index * 4 + user}`),
  );
  await writeFile(path.join(dataset, "ratings.csv"), "userId,movieId,rating,timestamp\n" + ratings.join("\n") + "\n");
  await writeFile(path.join(dataset, "tags.csv"), "userId,movieId,tag,timestamp\n1,79132,dreams,1500000001\n2,79132,dreams,1500000002\n");
  await writeFile(path.join(dataset, "links.csv"), "movieId,imdbId,tmdbId\n79132,1375666,27205\n109487,0816692,157336\n");
  launch(
    path.join(project, ".venv/Scripts/python.exe"),
    [
      "-X",
      "utf8",
      "-m",
      "uvicorn",
      "movie_rating.api.app:app",
      "--host",
      "127.0.0.1",
      "--port",
      "8137",
    ],
    project,
    { MOVIE_ARTIFACTS_DIR: path.join(testData, "artifacts"), MOVIE_DATASET_DIR: dataset, MOVIE_FRONTEND_ORIGIN: base },
  );
  launch(
    process.execPath,
    [
      path.join(frontend, "node_modules/vite/bin/vite.js"),
      "--host",
      "127.0.0.1",
      "--port",
      "5175",
    ],
    frontend,
    { MOVIE_API_PROXY: backendUrl },
  );
  await Promise.all([ready(`${backendUrl}/api/v1/health`), ready(base)]);
  assert.equal((await fetch(`${backendUrl}/gallery`)).status, 200);
  assert.equal((await fetch(`${backendUrl}/movies/79132`)).status, 200);
  assert.equal((await fetch(`${backendUrl}/api/v1/nonexistent`)).status, 404);
  browser = await chromium.launch({ executablePath, headless: true });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    reducedMotion: "reduce",
  });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`${base}/gallery`);
  await expectText(page, "16 bộ phim · MovieLens");
  const initialAlgorithm = page.getByRole("combobox", { name: "Mô hình gợi ý phim", exact: true });
  await initialAlgorithm.waitFor();
  assert.equal(await initialAlgorithm.locator("option").count(), 5);
  assert.equal(await initialAlgorithm.locator("option:disabled").count(), 4);
  await page.locator(".catalog-grid .movie-card").last().waitFor();
  await page.evaluate(() => document.fonts.ready);
  await page
    .locator(".catalog-grid .movie-card img")
    .evaluateAll((images) =>
      Promise.all(images.map((image) => image.decode())),
    );
  assert.equal(await page.locator(".catalog-grid .movie-card").count(), 12);
  assert.ok(await page.locator('.catalog-grid .movie-card [data-artwork="pattern"]').count() > 0);
  assert.ok(await page.locator('.catalog-grid .movie-card [data-artwork="image"].is-loaded').count() > 0);
  const accents = await page.locator('.catalog-grid .movie-card [data-artwork="pattern"]').evaluateAll(
    (artworks) => artworks.map((artwork) => artwork.style.getPropertyValue("--poster-accent")),
  );
  assert.ok(new Set(accents).size >= 3, "Missing-poster patterns should have varied palettes");
  assert.equal(await page.locator('[data-reveal="pending"]').count(), 0, "Reduced-motion content stays visible");
  await page.screenshot({
    path: path.join(output, "gallery-desktop.png"),
    fullPage: true,
  });
  await page.getByRole("link", { name: "MOVIE — Trang chủ", exact: true }).first().focus();
  await page.keyboard.press("/");
  await page.waitForFunction(() => document.activeElement?.getAttribute("aria-keyshortcuts") === "/");
  assert.equal(await page.getByRole("textbox", { name: "Tìm kiếm tên phim" }).evaluate((element) => element === document.activeElement), true);
  await page.keyboard.type("/");
  await page.waitForURL(/q=%2F/);
  await page.waitForFunction(() => document.querySelector('input[aria-keyshortcuts="/"]').value === "/");
  assert.equal(await page.getByRole("textbox", { name: "Tìm kiếm tên phim" }).inputValue(), "/");
  await page
    .getByRole("textbox", { name: "Tìm kiếm tên phim" })
    .fill("Interstellar");
  await page.waitForFunction(
    () => document.querySelectorAll(".catalog-grid .movie-card").length === 1,
  );
  assert.match(await page.locator(".catalog-grid .movie-card").innerText(), /Interstellar/);
  await page.getByRole("button", { name: "Xóa tìm kiếm" }).click();
  await page.getByRole("button", { name: "Kinh dị", exact: true }).click();
  await page.waitForFunction(() => document.querySelectorAll(".catalog-grid .movie-card").length === 1);
  assert.match(await page.locator(".catalog-grid .movie-card").innerText(), /A Horror Film/);
  await page.getByRole("textbox", { name: "Tìm kiếm tên phim" }).fill("phim-khong-ton-tai");
  await expectText(page, "Chưa tìm thấy bộ phim phù hợp");
  await page.getByRole("button", { name: "Xóa bộ lọc", exact: true }).click();
  await page.getByRole("textbox", { name: "Tìm kiếm tên phim" }).fill("Inception");
  await page
    .locator(".catalog-grid")
    .getByRole("link", { name: "Xem chi tiết phim Inception", exact: true })
    .click();
  await page.waitForURL("**/movies/79132");
  await page.getByRole("heading", { name: "Inception", level: 1, exact: true }).waitFor();
  assert.equal(await page.locator("dialog[open]").count(), 0);
  assert.equal(await page.getByLabel("Thuật toán dự đoán").count(), 0);
  assert.equal(await page.locator(".rating-bar-row").count(), 10);
  assert.match(await page.locator(".movie-rating-summary").innerText(), /4,25\s*\/5/);
  assert.match(await page.locator(".movie-tag-list").innerText(), /dreams/);
  assert.equal(await page.getByRole("link", { name: "Xem trên IMDb" }).getAttribute("href"), "https://www.imdb.com/title/tt1375666/");
  await page.reload();
  await page.getByRole("heading", { name: "Inception", level: 1, exact: true }).waitFor();
  await page.locator('.movie-page-art [data-artwork="image"].is-loaded').waitFor();
  await page.route("**/assets/posters/inception.webp", (route) => route.abort());
  await page.reload();
  await page.locator('.movie-page-art [data-artwork="pattern"]').waitFor();
  assert.equal(await page.locator(".movie-page-art img").count(), 0, "Broken images fall back to a designed pattern");
  assert.equal(await page.locator(".movie-page-art .poster-pattern svg").count(), 1);
  await page.unroute("**/assets/posters/inception.webp");
  await page.goBack();
  await page.waitForURL("**/gallery?q=Inception");
  assert.equal(await page.getByRole("textbox", { name: "Tìm kiếm tên phim" }).inputValue(), "Inception");
  await page.goto(`${base}/movies/999999`);
  await expectText(page, "Phim chưa có lượt đánh giá trong dataset");
  await expectText(page, "Chưa có từ khóa cho phim này");
  assert.equal(await page.locator('.movie-page-art [data-artwork="pattern"]').count(), 1);
  await page.goto(`${base}/movies/123456789`);
  await expectText(page, "Không tìm thấy phim trong dataset");
  await page.getByRole("button", { name: "Dự đoán và thuật toán" }).click();
  await expectText(page, "Chưa có kết quả huấn luyện");
  await page.keyboard.press("Escape");
  assert.equal(await page.locator("dialog[open]").count(), 0);
  await page.goto(`${base}/movies/79132`);
  await page.getByRole("button", { name: "Chấm 4 sao", exact: true }).click();
  await expectText(page, "Vui lòng đăng nhập để đánh giá phim");
  assert.match(await page.getByRole("link", { name: "Đăng nhập để đánh giá", exact: true }).getAttribute("href"), /next=%2Fmovies%2F79132/);
  await page.locator('.main-nav a[href="/watch-history"]').click();
  await page.waitForURL("**/watch-history");
  await expectText(page, "Đăng nhập để xem lịch sử");
  await page.goto(`${base}/home`);
  await prepareScreenshot(page, "home");
  assert.ok(await page.locator(".recommendation-section .rail-item").count() <= 20);
  await page.getByRole("button", { name: "Cuộn phải Các phim có thể bạn yêu thích", exact: true }).click();
  await page.waitForFunction(() => document.querySelector(".recommendation-section .movie-rail").scrollLeft > 100);
  const genreRows = page.locator(".home-genre-shelves .movie-rail-section");
  assert.equal(await genreRows.count(), 5);
  for (const row of await genreRows.all()) assert.ok(await row.locator(".rail-item").count() <= 10);
  await page.getByRole("link", { name: "Tất cả thể loại", exact: true }).click();
  await page.waitForURL("**/genres");
  await page.getByRole("link", { name: "Khám phá thể loại Hành động", exact: true }).click();
  await page.waitForURL("**/gallery?genre=Action");
  await page.waitForFunction(() => document.querySelector('.catalog-caption')?.textContent.includes('4 bộ phim'));
  assert.equal(await page.getByRole("combobox", { name: "Thể loại", exact: true }).inputValue(), "Action");
  await page.locator('.main-nav a[href="/genres"]').click();
  await page.waitForURL("**/genres");
  await page.getByRole("link", { name: "Mở trang thực nghiệm so sánh", exact: true }).click();
  await page.waitForURL("**/experiments");
  await expectText(page, "Chưa có kết quả thực nghiệm");
  assert.equal(await page.locator(".lab-sidebar nav button").count(), 5);
  assert.equal(await page.locator(".lab-algorithms article").count(), 4);
  assert.equal(await page.locator(".lab-chart").count(), 0);
  const protocolResponse = await page.request.get(`${base}/api/v1/experiments/documents/protocol`);
  assert.equal(protocolResponse.status(), 200);
  assert.match(protocolResponse.headers()["content-disposition"], /experimental-protocol.md/);
  await page.mouse.wheel(0, 900);
  await page.getByRole("button", { name: "Lên đầu trang", exact: true }).waitFor();
  const floating = await page.locator(".experiments-fab").boundingBox();
  const scrollTop = await page.locator(".back-to-top").boundingBox();
  assert.ok(scrollTop.y + scrollTop.height < floating.y, "Floating experiment and scroll controls must not overlap");

  // Full group execution is covered by check-experiments.mjs with real data.
  await page.locator('.lab-sidebar nav button').nth(1).click();
  await page.getByRole('heading', { name: 'Phân loại', exact: true }).waitFor();
  assert.equal(await page.locator('.lab-algorithms article').count(), 2);
  await page.locator('.lab-sidebar nav button').first().click();
  await page.getByRole('heading', { name: 'Hồi quy', exact: true }).waitFor();
  for (const route of ["about", "contact", "login", "register", "home", "genres", "experiments", "watch-history", "movies/79132"]) {
    await page.goto(`${base}/${route}`);
    await prepareScreenshot(page, route);
    await page.screenshot({
      path: path.join(output, `${route.replaceAll("/", "-")}-desktop.png`),
      fullPage: true,
    });
  }
  await page.goto(`${base}/contact`);
  await page.getByLabel("Họ và tên").fill("Người kiểm tra");
  await page.getByLabel("Email", { exact: false }).fill("viewer@example.com");
  await page
    .getByLabel("Chủ đề")
    .selectOption({ label: "Góp ý về trải nghiệm" });
  await page
    .getByLabel("Tin nhắn")
    .fill("Kiểm tra gửi tin nhắn từ giao diện MOVIE.");
  await page.getByRole("button", { name: "Gửi tin nhắn" }).click();
  await expectText(page, "Tin nhắn đã được lưu trên hệ thống.");
  await page.goto(`${base}/register`);
  await page
    .getByRole("textbox", { name: "Họ và tên", exact: true })
    .fill("Movie Viewer");
  await page
    .getByRole("textbox", { name: "Email", exact: true })
    .fill("viewer@example.com");
  await page.getByLabel("Mật khẩu", { exact: true }).fill("strong-pass-123");
  await page
    .getByLabel("Xác nhận mật khẩu", { exact: true })
    .fill("different-pass-123");
  await page.getByRole("button", { name: "Đăng ký", exact: true }).click();
  await expectText(page, "Mật khẩu xác nhận chưa khớp.");
  await page
    .getByLabel("Xác nhận mật khẩu", { exact: true })
    .fill("strong-pass-123");
  await page.getByRole("button", { name: "Đăng ký", exact: true }).click();
  await page.waitForURL("**/gallery");
  await page.locator('.recommendation-section[data-mode="popular"]').waitFor();
  assert.equal((await (await context.request.get(`${base}/api/v1/me/ratings`)).json()).total, 0,
    "New account must not inherit a MovieLens history");
  await page.goto(`${base}/movies/79132`);
  await page.getByRole("combobox", { name: "Điểm đánh giá của bạn", exact: true }).selectOption("4.5");
  await page.getByRole("button", { name: "Lưu đánh giá", exact: true }).click();
  await expectText(page, "Đã lưu đánh giá của bạn vào Watch History");
  await page.goto(`${base}/watch-history`);
  await page.locator(".history-entry").waitFor();
  assert.equal(await page.locator(".history-entry").count(), 1);
  assert.match(await page.locator(".history-entry-meta").innerText(), /4,5\/5/);
  await page.reload();
  await page.getByRole("link", { name: "Sửa đánh giá", exact: true }).click();
  await page.getByRole("combobox", { name: "Điểm đánh giá của bạn", exact: true }).selectOption("2.5");
  await page.getByRole("button", { name: "Cập nhật đánh giá", exact: true }).click();
  await expectText(page, "Đã lưu đánh giá của bạn vào Watch History");
  await page.goto(`${base}/home`);
  await page.locator('.recommendation-section[data-mode="content"]').waitFor();
  assert.equal(await page.locator('.recommendation-section a[href="/movies/79132"]').count(), 0);
  assert.equal(await page.locator(".recommendation-section .prediction-caption").count(), 0, "No fabricated model scores before training");
  await page.getByRole("button", { name: "Tài khoản", exact: true }).click();
  await expectText(page, "Movie Viewer");
  await page.getByRole("button", { name: "Đăng xuất", exact: true }).click();
  await expectText(page, "Bạn đã đăng xuất.");
  await page.goto(`${base}/login?next=%2Fwatch-history`);
  await page
    .getByRole("textbox", { name: "Email", exact: true })
    .fill("viewer@example.com");
  await page.getByLabel("Mật khẩu", { exact: true }).fill("incorrect");
  await page
    .getByRole("button", { name: "Hiện mật khẩu", exact: true })
    .click();
  assert.equal(
    await page.getByLabel("Mật khẩu", { exact: true }).getAttribute("type"),
    "text",
  );
  await page.getByRole("button", { name: "Đăng nhập", exact: true }).click();
  await expectText(page, "Email hoặc mật khẩu chưa chính xác.");
  await page.getByLabel("Mật khẩu", { exact: true }).fill("strong-pass-123");
  await page.getByRole("button", { name: "Đăng nhập", exact: true }).click();
  await page.waitForURL("**/watch-history");
  await page.reload();
  await page.locator(".history-entry").waitFor();
  assert.equal(await page.locator(".history-entry").count(), 1);
  assert.match(await page.locator(".history-entry-meta").innerText(), /2,5\/5/);
  await page.screenshot({ path: path.join(output, "watch-history-rated-desktop.png"), fullPage: true });
  await page.getByRole("button", { name: "Tài khoản", exact: true }).click();
  await expectText(page, "Movie Viewer");
  await page.getByRole("button", { name: "Đăng xuất", exact: true }).click();
  await page.goto(`${base}/login?next=%2Fwatch-history`);
  await page.locator(".demo-account").last().waitFor();
  const demoAccounts = (await (await context.request.get(`${base}/api/v1/auth/demo-accounts`)).json()).accounts;
  assert.equal(demoAccounts.length, 4, "Fixture has only four dataset users");
  assert.equal(await page.locator(".demo-account").count(), 4);
  assert.ok((await page.locator(".demo-account").first().innerText()).includes(demoAccounts[0].password));
  await page.locator(".demo-account").first().getByRole("button").click();
  assert.equal(await page.getByRole("textbox", { name: "Email", exact: true }).inputValue(), demoAccounts[0].email);
  assert.equal(await page.getByLabel("Mật khẩu", { exact: true }).inputValue(), demoAccounts[0].password);
  await page.getByRole("button", { name: "Đăng nhập", exact: true }).click();
  await page.waitForURL("**/watch-history");
  await page.locator(".history-entry").last().waitFor();
  assert.equal(await page.locator(".history-entry").count(), 15);
  await expectText(page, "MovieLens User 1");
  await page.screenshot({ path: path.join(output, "watch-history-demo-desktop.png"), fullPage: true });
  await page.goto(`${base}/home`);
  await page.locator('.recommendation-section[data-mode="content"]').waitFor();
  assert.equal(await page.locator(".recommendation-section .movie-card").count(), 1);
  assert.equal(await page.locator('.recommendation-section a[href="/movies/999999"]').count(), 1);
  const importedRating = await (await context.request.get(`${base}/api/v1/me/ratings/79132`)).json();
  assert.equal(importedRating.rating, 3.5);
  assert.equal(importedRating.updated_at, 1500000001 * 1000);
  await page.goto(`${base}/movies/79132`);
  await page.waitForFunction(() => document.querySelector('select[aria-label="Điểm đánh giá của bạn"]')?.value === "3.5");
  await page.getByRole("combobox", { name: "Điểm đánh giá của bạn", exact: true }).selectOption("1.5");
  await page.getByRole("button", { name: "Cập nhật đánh giá", exact: true }).click();
  await expectText(page, "Đã lưu đánh giá của bạn vào Watch History");
  await context.request.get(`${base}/api/v1/auth/demo-accounts`);
  await page.reload();
  await page.waitForFunction(() => document.querySelector('select[aria-label="Điểm đánh giá của bạn"]')?.value === "1.5");
  assert.equal((await (await context.request.get(`${base}/api/v1/auth/me`)).json()).profile_modified, true);
  // Return to the ordinary viewer to check isolation after using a dataset demo.
  await context.request.post(`${base}/api/v1/auth/logout`);
  await context.request.post(`${base}/api/v1/auth/login`, { data: { email: "viewer@example.com", password: "strong-pass-123" } });
  await page.goto(`${base}/watch-history`);
  await page.locator(".history-entry").waitFor();
  assert.equal(await page.locator(".history-entry").count(), 1);
  assert.match(await page.locator(".history-entry-meta").innerText(), /2,5\/5/);
  for (const width of [390, 360, 768]) {
    await page.setViewportSize({ width, height: 844 });
    for (const route of [
      "gallery",
      "about",
      "contact",
      "login",
      "register",
      "home",
      "genres",
      "experiments",
      "watch-history",
      "movies/79132",
      "movies/999999",
    ]) {
      await page.goto(`${base}/${route}`);
      await prepareScreenshot(page, route);
      if (route === "login") {
        const form = await page.locator(".auth-panel").boundingBox();
        const demos = await page.locator(".auth-demo-panel").boundingBox();
        assert.ok(demos.y >= form.y + form.height, `Demo accounts must be below the form at ${width}px`);
        assert.ok(form.width >= Math.min(width - 70, 550), `Login form must remain usable at ${width}px`);
      }
      const overflows = await page.evaluate(
        () => document.documentElement.scrollWidth > window.innerWidth,
      );
      assert.equal(
        overflows,
        false,
        `Horizontal overflow: ${route} at ${width}px`,
      );
      if (width === 390)
        await page.screenshot({
          path: path.join(output, `${route.replaceAll("/", "-")}-mobile.png`),
          fullPage: true,
        });
    }
    await page.getByRole("button", { name: "Mở menu", exact: true }).click();
    await page.locator('.main-nav a[href="/genres"]').click();
    await page.waitForURL("**/genres");
    await page.waitForFunction(() => !document.querySelector(".main-nav").classList.contains("is-open"));
    await page.locator(".experiments-fab").click();
    await page.waitForURL("**/experiments");
    await expectText(page, "Chưa có kết quả thực nghiệm");
    const bounds = await page.locator(".experiments-fab").boundingBox();
    assert.ok(bounds.x >= 0 && bounds.x + bounds.width <= width);
  }

  // Exercise real motion separately from the static responsive screenshots.
  const animatedContext = await browser.newContext({
    viewport: { width: 1440, height: 1000 }, reducedMotion: "no-preference",
  });
  const animated = await animatedContext.newPage();
  animated.on("pageerror", (error) => errors.push(error.message));
  await animated.route("**/api/v1/movies?*", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 350));
    await route.continue();
  });
  await animated.goto(`${base}/gallery`);
  await animated.locator(".skeleton-grid").waitFor();
  await expectText(animated, "16 bộ phim · MovieLens");
  await animated.unroute("**/api/v1/movies?*");
  await animated.locator(".catalog-grid .movie-card").first().scrollIntoViewIfNeeded();
  await animated.locator('.catalog-grid .movie-card[data-reveal="visible"]').first().waitFor();
  const card = animated.locator(".catalog-grid .movie-card").first();
  await card.hover();
  const bounds = await card.boundingBox();
  await animated.mouse.move(bounds.x + bounds.width * .8, bounds.y + bounds.height * .3, { steps: 5 });
  await animated.waitForFunction(() => {
    const card = document.querySelector(".catalog-grid .movie-card");
    return parseFloat(card.style.getPropertyValue("--tilt-y")) > .2;
  });
  await animated.waitForFunction(() => Number(getComputedStyle(document.querySelector(".catalog-grid .movie-card .poster-hover")).opacity) > .9);
  await animated.screenshot({ path: path.join(output, "pattern-hover-desktop.png"), fullPage: true });
  await animated.mouse.move(5, 5);
  await animated.waitForFunction(() => document.querySelector(".catalog-grid .movie-card").style.getPropertyValue("--tilt-x") === "0deg");
  await animated.goto(`${base}/about`);
  await animated.locator(".dataset-stats").waitFor();
  await animated.mouse.wheel(0, 900);
  await animated.getByRole("button", { name: "Lên đầu trang", exact: true }).waitFor();
  assert.ok(await animated.evaluate(() => Number(document.documentElement.style.getPropertyValue("--scroll-progress"))) > 0);
  assert.equal(await animated.evaluate(() => document.documentElement.hasAttribute("data-scrolled")), true);
  await animated.locator('.dataset-stats > div[data-reveal="visible"]').first().waitFor();
  await animated.getByRole("button", { name: "Lên đầu trang", exact: true }).click();
  await animated.waitForFunction(() => window.scrollY < 3);
  await animated.locator('.main-nav a[href="/gallery"]').click();
  await animated.waitForURL("**/gallery");
  await animated.locator(".catalog-grid .movie-card").first().waitFor();
  assert.equal(await animated.locator(".route-frame").evaluate((element) => getComputedStyle(element).animationName), "page-arrive");
  await animated.locator(".catalog-grid .movie-card").last().scrollIntoViewIfNeeded();
  await animated.waitForFunction(() => window.scrollY > 100);
  const beforeDetail = await animated.evaluate(() => window.scrollY);
  await animated.locator(".catalog-grid .movie-card").last().click();
  await animated.locator(".movie-overview").waitFor();
  await animated.goBack();
  await animated.waitForURL("**/gallery");
  await animated.waitForFunction((expected) => Math.abs(window.scrollY - expected) < 12, beforeDetail);
  await animatedContext.close();

  // Train all four implementations on the isolated fixture, then switch them in the browser.
  const trainer = launch(path.join(project, ".venv/Scripts/python.exe"), ["-X", "utf8", "-c", [
    "from movie_rating.settings import load_settings",
    "from movie_rating.data.prepare import prepare",
    "from movie_rating.training import train",
    "s = load_settings()",
    "s.data.user_sample_fraction = 1",
    "s.models['ncf'].update(epochs=2)",
    "s.models['svd'].update(epochs=2)",
    "s.models['xgboost'].update(n_estimators=10)",
    "prepare(s)", "train(s)",
  ].join("; ")], project, { MOVIE_ARTIFACTS_DIR: path.join(testData, "artifacts"), MOVIE_DATASET_DIR: dataset });
  const trainerCode = await new Promise((resolve, reject) => {
    trainer.once("error", reject);
    trainer.once("exit", resolve);
  });
  assert.equal(trainerCode, 0, logs.join("").slice(-3000));
  const modelContext = await browser.newContext({ viewport: { width: 390, height: 844 }, reducedMotion: "reduce" });
  const modelPage = await modelContext.newPage();
  modelPage.on("pageerror", (error) => errors.push(error.message));
  await modelPage.goto(`${base}/gallery`);
  await modelPage.locator('.recommendation-section[data-mode="popular"]').waitFor();
  const algorithmSelect = modelPage.getByRole("combobox", { name: "Mô hình gợi ý phim", exact: true });
  assert.equal(await algorithmSelect.locator("option:disabled").count(), 0);
  for (const name of ["ridge", "xgboost", "svd", "ncf"]) {
    const response = modelPage.waitForResponse((value) => value.url().includes(`/recommendations?limit=20&algorithm=${name}`));
    await algorithmSelect.selectOption(name);
    const result = await (await response).json();
    assert.equal(result.mode, "popular");
    assert.ok(result.items.every((movie) => movie.predicted_rating === null));
    await modelPage.locator('.recommendation-section[data-mode="popular"]').waitFor();
  }
  await modelPage.locator(".recommendation-onboarding").waitFor();
  const registration = await modelContext.request.post(`${base}/api/v1/auth/register`, { data: {
    name: "Algorithm viewer", email: "algorithms@example.com", password: "test-password-123",
  } });
  assert.equal(registration.status(), 201);
  const savedRating = await modelContext.request.put(`${base}/api/v1/me/ratings/79132`, { data: { rating: 5 } });
  assert.equal(savedRating.status(), 200);
  await modelPage.reload();
  await modelPage.locator('.recommendation-section[data-mode="model"]').waitFor();
  for (const name of ["ridge", "xgboost", "svd", "ncf"]) {
    const response = modelPage.waitForResponse((value) => value.url().includes(`/recommendations?limit=20&algorithm=${name}`));
    await algorithmSelect.selectOption(name);
    const result = await (await response).json();
    assert.equal(result.algorithm, name);
    assert.equal(result.strategy, "profile_fold_in");
    assert.ok(result.items.length > 0);
    assert.ok(result.items.every((movie) => movie.movieId !== 79132 && movie.predicted_rating >= .5 && movie.predicted_rating <= 5));
    await modelPage.locator('.recommendation-section[data-mode="model"]').waitFor();
    assert.ok(await modelPage.locator(".recommendation-section .prediction-caption").count() > 0);
  }
  await modelPage.screenshot({ path: path.join(output, "algorithm-selector-mobile.png"), fullPage: true });
  assert.equal(await modelPage.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
  await modelPage.goto(`${base}/experiments`);
  await modelPage.locator(".lab-algorithms article").last().waitFor();
  assert.equal(await modelPage.locator(".lab-algorithms article").count(), 4);
  await expectText(modelPage, "Tiêu chí so sánh");
  await modelContext.close();
  assert.deepEqual(errors, [], "Browser JavaScript errors");
  const assets = [
    "cinema",
    "theater",
    "contact",
    "about",
    ...["frontend", "backend", "ml", "design"].map((name) => `team/${name}`),
  ];
  for (const asset of assets) {
    const response = await fetch(`${base}/assets/${asset}.webp`);
    assert.equal(response.status, 200, `Missing asset ${asset}`);
    assert.match(response.headers.get("content-type"), /image\/webp/);
  }
  const html = await readFile(path.join(frontend, "index.html"), "utf8");
  assert.ok(html.includes('lang="vi"'));
  console.log(
    "UI checks passed: dataset demo import/login/edit, new account isolation, responsive Login; Genres navigation, experiments, recommendations/genre rails, ratings/history, filters, patterns and motion.",
  );
  console.log(`Screenshots: ${output}`);
} catch (error) {
  console.error(logs.join("").slice(-5000));
  throw error;
} finally {
  await browser?.close();
  for (const child of children.reverse()) {
    if (child.exitCode === null) {
      await new Promise((resolve) => {
        child.once("exit", resolve);
        child.kill();
      });
    }
  }
  // Verify the exact temporary directory before recursive cleanup on Windows.
  const resolved = path.resolve(testData);
  if (
    path.dirname(resolved) !== path.resolve(tmpdir()) ||
    !path.basename(resolved).startsWith("movie-ui-test-")
  ) {
    throw new Error(
      "Refusing to remove a path outside the UI test temporary directory.",
    );
  }
  await rm(resolved, {
    recursive: true,
    force: true,
    maxRetries: 3,
    retryDelay: 100,
  });
}
