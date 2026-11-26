// Browser contract check for backend presets; intercepted start requests avoid model runs.
// Stored-history fixtures exercise result rendering and missing-field fallbacks only.
import assert from "node:assert/strict";
import { access } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { chromium } from "playwright";

const frontend = fileURLToPath(new URL("../", import.meta.url));
const project = path.resolve(frontend, "..");
const base = "http://127.0.0.1:8139";
let executablePath;
for (const candidate of [process.env.CHROME_PATH, "C:/Program Files/Google/Chrome/Application/chrome.exe", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"].filter(Boolean)) {
  try { await access(candidate); executablePath = candidate; break; } catch { /* try next */ }
}
assert.ok(executablePath, "Set CHROME_PATH to an installed Chromium browser");

const python = process.env.MOVIE_TEST_PYTHON || path.join(project, ".venv/Scripts/python.exe");
const server = spawn(python, ["-X", "utf8", "-m", "uvicorn", "movie_rating.api.app:app", "--host", "127.0.0.1", "--port", "8139"], {
  cwd: project, windowsHide: true, stdio: ["ignore", "pipe", "pipe"],
});
const logs = [];
server.stdout.on("data", (value) => logs.push(value.toString()));
server.stderr.on("data", (value) => logs.push(value.toString()));
let browser;

try {
  for (let attempt = 0; attempt < 150; attempt++) {
    try { if ((await fetch(`${base}/api/v1/experiment-lab/catalog`)).ok) break; } catch { /* server starting */ }
    if (attempt === 149) throw new Error(`Backend did not start: ${logs.join("").slice(-3000)}`);
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  const catalog = await (await fetch(`${base}/api/v1/experiment-lab/catalog`)).json();
  const groups = ["rating", "classification", "clustering", "association", "reinforcement"];
  assert.deepEqual(catalog.groups.filter((group) => group.id !== "pca").map((group) => group.id), groups);
  assert.ok(catalog.presets, "The backend catalog must publish the presets used by request validation");

  const expectedReduction = {
    rating: { default_pca: true, selected_pca: false, algorithm: "xgboost" },
    classification: { default_pca: true, selected_pca: false, algorithm: "random_forest" },
    clustering: { default_pca: true, selected_pca: true, algorithm: "dbscan" },
    association: { default_pca: true, selected_pca: true, algorithm: "apriori" },
    reinforcement: { default_pca: false, selected_pca: false, algorithm: "random" },
  };
  let runList = [];
  const posted = [];
  const mockedJobs = new Map();
  const currentHistory = {
    id: "20190101T000000-abcdef01", group: "rating", status: "completed", progress: 100,
    message: "schema three fixture", created_at: "2019-01-01T00:00:00Z", schema_version: 3,
    result: {
      schema_version: 3, group: "rating", rows: [{
        algorithm: "linear", base_algorithm: "linear", variant_id: "linear_raw",
        name: "Linear Regression", split: "test", representation: "raw", rmse: 1.2,
      }],
      charts: [], notes: [], provenance: {}, cv: {}, best_configs: {},
      reduction: { mode: "off", variants: [] },
    },
  };
  const oldHistory = {
    id: "20180101T000000-abcdef02", group: "rating", status: "completed", progress: 100,
    message: "schema two fixture", created_at: "2018-01-01T00:00:00Z", schema_version: 2,
    result: {
      schema_version: 2, group: "rating", rows: [{
        algorithm: "linear", name: "Linear Regression", split: "test", rmse: 1.3,
      }],
      charts: [], notes: [], provenance: {},
    },
  };
  const histories = new Map([[currentHistory.id, currentHistory], [oldHistory.id, oldHistory]]);

  browser = await chromium.launch({ executablePath, headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, reducedMotion: "reduce" });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/api/v1/experiment-lab/runs**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.method() === "POST" && url.pathname.endsWith("/runs")) {
      const payload = request.postDataJSON();
      posted.push(payload);
      const job = {
        id: `preset-check-${payload.group}`, group: payload.group, status: "completed",
        progress: 100, message: "request captured", created_at: "2026-01-01T00:00:00Z",
        request: payload, schema_version: 4,
      };
      mockedJobs.set(job.id, job);
      await route.fulfill({ status: 202, json: job });
      return;
    }
    const id = url.pathname.split("/").at(-1);
    if (histories.has(id)) {
      await route.fulfill({ json: histories.get(id) });
      return;
    }
    if (mockedJobs.has(id)) {
      await route.fulfill({ json: mockedJobs.get(id) });
      return;
    }
    await route.fulfill({ json: runList });
  });

  for (const groupId of groups) {
    const group = catalog.groups.find((item) => item.id === groupId);
    const expected = expectedReduction[groupId];
    await page.goto(`${base}/experiments?group=${groupId}`);
    await page.locator(".lab-algorithm-picker").waitFor();
    assert.equal(await page.locator(".lab-stat-grid").first().locator(".lab-stat-card").count(), 5);
    const statLabels = await page.locator(".lab-stat-grid").first().innerText();
    assert.ok(statLabels.includes("Phim có đánh giá") && statLabels.includes("Phim trong danh mục"));
    assert.ok(!statLabels.includes("Sparsity") && !statLabels.includes("Rating mean"));
    const guide = page.getByTestId("lab-reduction-guide");
    assert.equal(await guide.getAttribute("data-group-id"), groupId);
    assert.equal(await page.getByTestId("reduction-card-pca").getAttribute("data-active"), String(expected.default_pca));
    assert.equal(await page.getByTestId("reduction-card-truncated_svd").getAttribute("data-active"), "false");
    assert.equal(await page.locator(".lab-reduction-card").count(), 3);

    const config = page.locator(".lab-preset");
    assert.equal(await config.count(), 0, "Preset annotations are hidden for every algorithm group");
    const preset = catalog.presets[groupId];
    assert.ok(preset?.id?.endsWith(".standard.v1"));
    assert.equal(await page.locator(".lab-algorithm-picker input").count(), group.algorithms.length);

    const ids = group.algorithms.map((algorithm) => algorithm.id);
    assert.ok(ids.includes(expected.algorithm));
    const options = page.locator(".lab-algorithm-picker input");
    for (let index = 0; index < ids.length; index++) {
      if (ids[index] === expected.algorithm) await options.nth(index).check();
      else await options.nth(index).uncheck();
    }
    assert.equal(await page.getByTestId("reduction-card-pca").getAttribute("data-active"), String(expected.selected_pca));
    assert.equal(await page.getByTestId("reduction-card-truncated_svd").getAttribute("data-active"), "false");

    const post = page.waitForRequest((request) => request.url().endsWith("/experiment-lab/runs") && request.method() === "POST");
    await page.locator(".lab-run-actions .lab-primary").first().click();
    await post;
    assert.deepEqual(posted.at(-1), {
      group: groupId, algorithms: [expected.algorithm], preset_id: preset.id,
      reduction_choice: expected.default_pca ? "pca" : "off",
    });
    assert.equal(await config.count(), 0);

    // Exercise all choices with algorithms that can consume reduced profiles.
    for (const option of await options.all()) await option.check();
    for (const choice of ["truncated_svd", "off", "pca"]) {
      await page.locator(`input[name="reduction-choice"][value="${choice}"]`).check();
      assert.equal(await page.locator('input[name="reduction-choice"]:checked').count(), 1);
      const selectedCard = page.getByTestId(`reduction-card-${choice}`);
      assert.ok((await selectedCard.innerText()).includes("Đã chọn"));
      if (choice === "off") {
        assert.equal(await page.getByTestId("reduction-card-pca").getAttribute("data-active"), "false");
        assert.equal(await page.getByTestId("reduction-card-truncated_svd").getAttribute("data-active"), "false");
      } else {
        assert.equal(await selectedCard.getAttribute("data-active"), "true");
      }
      const capture = page.waitForRequest((request) => request.url().endsWith("/experiment-lab/runs") && request.method() === "POST");
      await page.locator(".lab-run-actions .lab-primary").first().click();
      await capture;
      assert.deepEqual({...posted.at(-1), algorithms: [...posted.at(-1).algorithms].sort()}, {
        group: groupId, algorithms: [...ids].sort(), preset_id: preset.id, reduction_choice: choice,
      });
    }
  }

  runList = [currentHistory, oldHistory];
  await page.goto("http://127.0.0.1:8139/experiments?group=rating");
  await page.locator("#lab-results").waitFor();
  const rows = await page.locator(".lab-results > .lab-result-row").allTextContents();
  assert.match(rows[0], /Kết quả giảm chiều/);
  assert.match(rows[1], /Kết quả huấn luyện/);
  assert.match(rows[2], /Biểu đồ so sánh/);
  assert.match(rows[3], /Cross Validation/);
  assert.doesNotMatch(rows[3], /\d-fold/); // Missing metadata must not invent a fold count.
  assert.ok((await page.locator(".lab-results").innerText()).includes("Không có dữ liệu"));
  for (const count of [5, 3]) {
    currentHistory.result.cv_protocol = "per_user_expanding_time_v1";
    currentHistory.result.cv_folds = count;
    currentHistory.result.cv = { linear: {
      metrics: { rmse: { mean: 1, std: 0.1, n: count }, mae: { mean: 0.8, std: 0.1, n: count } },
      folds: Array.from({ length: count }, (_, index) => ({ fold: index + 1, rmse: 1 + index / 10, mae: 0.8 + index / 10 })),
    } };
    await page.reload();
    await page.locator("#lab-cv-title").waitFor();
    assert.match(await page.locator("#lab-cv-title").innerText(), new RegExp(`${count}-fold`));
    const table = page.locator('section[aria-labelledby="lab-cv-title"] table');
    assert.equal(await table.locator("thead th").count(), 1 + 2 * (count + 1));
    assert.equal(await table.locator("tbody tr").first().locator("td").count(), 2 * (count + 1));
  }
  await page.locator(".lab-history select").selectOption(oldHistory.id);
  await page.locator("#lab-results").waitFor();
  const legacyText = await page.locator("#lab-results").innerText();
  assert.ok(legacyText.includes("Không có dữ liệu"), legacyText);
  await page.setViewportSize({ width: 390, height: 844 });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  assert.equal(await page.locator(".lab-reduction-grid").evaluate((el) => getComputedStyle(el).gridTemplateColumns.split(" ").length), 1);
  assert.deepEqual(errors, []);
  console.log("PASS five backend-owned preset payloads, algorithm-specific reduction states, read-only settings, ordered results/CV, legacy missing-field fallback, and mobile layout.");
} finally {
  await browser?.close();
  server.kill();
}
