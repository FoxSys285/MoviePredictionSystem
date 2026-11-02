// End-to-end runs on the prepared MovieLens sample, with real saved results.
import assert from "node:assert/strict";
import { access, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { chromium } from "playwright";

const frontend = fileURLToPath(new URL("../", import.meta.url));
const project = path.resolve(frontend, "..");
const output = path.join(frontend, "test-results");
const base = "http://127.0.0.1:8139";
let executablePath;
for (const candidate of [process.env.CHROME_PATH, "C:/Program Files/Google/Chrome/Application/chrome.exe", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"].filter(Boolean)) {
  try { await access(candidate); executablePath = candidate; break; } catch {}
}
assert.ok(executablePath, "Set CHROME_PATH to an installed browser");
const server = spawn(path.join(project, ".venv/Scripts/python.exe"), ["-X", "utf8", "-m", "uvicorn", "movie_rating.api.app:app", "--host", "127.0.0.1", "--port", "8139"], { cwd: project, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
const logs = [];
server.stdout.on("data", d => logs.push(d.toString())); server.stderr.on("data", d => logs.push(d.toString()));
let browser;
const summary = [];
try {
  for (let attempt=0; attempt<100; attempt++) {
    try { if ((await fetch(`${base}/api/v1/experiment-lab/catalog`)).ok) break; } catch {}
    if (attempt === 99) throw new Error(`Server did not start: ${logs.join("").slice(-3000)}`);
    await new Promise(resolve => setTimeout(resolve,200));
  }
  const catalog = await (await fetch(`${base}/api/v1/experiment-lab/catalog`)).json();
  assert.ok(catalog.data_ready, "Prepare data before running this end-to-end test");
  await mkdir(output, { recursive:true });
  browser = await chromium.launch({ executablePath, headless:true });
  const page = await browser.newPage({ viewport:{ width:1440,height:1000 }, reducedMotion:"reduce" });
  const errors = [];
  page.on("pageerror", e => errors.push(e.message));
  for (const group of catalog.groups) {
    await page.goto(`${base}/experiments?group=${group.id}`);
    await page.locator(".lab-sidebar nav button").last().waitFor();
    assert.equal(await page.locator(".lab-sidebar nav button").count(),6);
    assert.equal(await page.locator(".lab-algorithms article").count(),group.algorithms.length);
    assert.equal(await page.locator(".lab-criteria").first().locator("tbody tr").count(),group.metrics.length);
    assert.ok((await page.locator(".lab-criteria").first().innerText()).includes("Câu hỏi được trả lời"));
    await page.getByLabel("Số user tối đa",{ exact:true }).fill("80");
    if (["rating","classification","clustering","reinforcement"].includes(group.id)) await page.getByLabel("Số seed huấn luyện",{ exact:true }).fill("2");
    if (group.id === "rating") await page.getByLabel("Epochs SVD / NCF",{ exact:true }).fill("4");
    if (group.id === "reinforcement") await page.getByLabel("Episode huấn luyện",{ exact:true }).fill("300");
    if (["clustering","association"].includes(group.id)) await page.getByLabel("Số phim tối đa",{ exact:true }).fill("40");
    const button = page.getByRole("button",{ name:"Chạy thực nghiệm nhóm",exact:true });
    await button.waitFor();
    await page.waitForFunction(() => !document.querySelector(".lab-primary")?.disabled);
    const responsePromise = page.waitForResponse(r => r.url().endsWith("/experiment-lab/runs") && r.request().method() === "POST");
    await button.click();
    const response = await responsePromise;
    assert.equal(response.status(),202);
    const job = await response.json();
    await page.locator(".lab-job-completed,.lab-job-failed").waitFor({ timeout:180000 });
    const finished = await (await page.request.get(`${base}/api/v1/experiment-lab/runs/${job.id}`)).json();
    assert.equal(finished.status,"completed",finished.message);
    await page.locator(".lab-results .lab-comparison tbody tr").first().waitFor();
    assert.ok(await page.locator(".lab-chart").count() >= group.charts.length);
    assert.ok((await page.locator(".lab-chart svg").count()) > 0);
    assert.equal(finished.result.provenance.sampled_users,80);
    assert.ok(finished.result.provenance.splits.train.max_timestamp < finished.result.provenance.splits.validation.min_timestamp);
    const csv = await page.request.get(`${base}/api/v1/experiment-lab/runs/${job.id}/export/csv`);
    assert.equal(csv.status(),200); assert.ok((await csv.text()).includes("algorithm"));
    const testButton = page.getByRole("group",{ name:"Tập dữ liệu đánh giá" }).getByRole("button",{ name:"Test",exact:true });
    if (await testButton.count()) { await testButton.click(); assert.ok((await page.locator(".lab-comparison caption").innerText()).includes("test")); }
    const legend = page.locator(".lab-legend button").first();
    await legend.click(); assert.equal(await legend.getAttribute("aria-pressed"),"false"); await legend.click();
    await page.locator(".lab-results").scrollIntoViewIfNeeded();
    await page.screenshot({ path:path.join(output,`lab-${group.id}-results.png`) });
    for (const theme of ["dark","light"]) {
      await page.evaluate(v => { document.documentElement.dataset.theme=v; },theme);
      await page.setViewportSize({ width:390,height:844 });
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),`Overflow: ${group.id}/${theme}`);
    }
    await page.screenshot({ path:path.join(output,`lab-${group.id}-mobile.png`) });
    await page.setViewportSize({ width:1440,height:1000 });
    await page.evaluate(() => { document.documentElement.dataset.theme="dark"; });
    summary.push({ group:group.id,run_id:job.id,rows:finished.result.rows.length,charts:finished.result.charts.length,ratings:finished.result.provenance.sampled_ratings });
    console.log(`PASS ${group.id}: ${finished.result.rows.length} rows, ${finished.result.charts.length} charts, ${finished.result.provenance.sampled_ratings} real ratings`);
  }
  // Sidebar and deep links preserve group state through a reload.
  await page.locator('.lab-sidebar nav button').first().click();
  assert.ok(page.url().includes("group=rating"));
  await page.reload(); await page.locator(".lab-algorithms article").last().waitFor();
  assert.equal(await page.locator(".lab-algorithms article").count(),4);
  await page.screenshot({ path:path.join(output,"lab-overview-desktop.png") });
  // Recoverable API failure is explicit; no fabricated result is rendered.
  await page.route("**/experiment-lab/catalog",route => route.fulfill({ status:503,json:{detail:"Kiểm thử lỗi kết nối"} }));
  await page.reload(); await page.getByText("Kiểm thử lỗi kết nối",{exact:true}).waitFor();
  assert.equal(await page.locator(".lab-comparison").count(),0);
  await page.unroute("**/experiment-lab/catalog");
  await page.getByRole("button",{name:"Thử lại",exact:true}).click();
  await page.locator(".lab-algorithms article").last().waitFor();
  assert.deepEqual(errors,[]);
  await writeFile(path.join(output,"lab-real-runs.json"),JSON.stringify(summary,null,2));
  console.log("PASS sidebar, reload, exports, split switching, chart legends, six groups, mobile dark/light and error recovery.");
} finally {
  await browser?.close(); server.kill();
}
