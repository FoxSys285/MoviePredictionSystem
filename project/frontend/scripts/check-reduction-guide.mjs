// Lightweight Playwright coverage for the shared reduction UI and legacy PCA route.
import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { chromium } from "playwright";

const frontend=fileURLToPath(new URL("../",import.meta.url));
const project=path.resolve(frontend,"..");
const catalogPath=path.join(project,"src/movie_rating/lab_catalog.json");
const catalog=JSON.parse(await readFile(catalogPath,"utf8"));
const base="http://127.0.0.1:5195";
let executablePath;
for(const candidate of [process.env.CHROME_PATH,"C:/Program Files/Google/Chrome/Application/chrome.exe","C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"].filter(Boolean)){
  try{await access(candidate);executablePath=candidate;break;}catch{}
}
assert.ok(executablePath,"Set CHROME_PATH to an installed Chromium browser");
const server=spawn(process.execPath,[path.join(frontend,"node_modules/vite/bin/vite.js"),"--host","127.0.0.1","--port","5195","--strictPort"],
  {cwd:frontend,windowsHide:true,stdio:["ignore","pipe","pipe"]});
const serverOutput=[];
server.stdout.on("data",value=>serverOutput.push(value.toString()));
server.stderr.on("data",value=>serverOutput.push(value.toString()));
let browser;
try{
  for(let attempt=0;attempt<100;attempt++){
    try{if((await fetch(base)).ok)break;}catch{}
    if(attempt===99)throw new Error(`Vite did not start: ${serverOutput.join("").slice(-2500)}`);
    await new Promise(resolve=>setTimeout(resolve,150));
  }
  browser=await chromium.launch({executablePath,headless:true});
  const page=await browser.newPage({viewport:{width:1440,height:1000},reducedMotion:"reduce"});
  const errors=[];page.on("pageerror",error=>errors.push(error.message));
  const legacyRun={
    id:"legacy-pca-fixture",group:"pca",status:"completed",progress:100,
    message:"Legacy PCA history fixture",created_at:"2026-01-01T00:00:00Z",schema_version:2,
    result:{
      schema_version:2,
      provenance:{sampled_users:4,sampled_ratings:40},
      rows:[{algorithm:"pca",name:"PCA · 1",split:"validation",components:1,explained_variance_ratio:0.5,reconstruction_error:0.05}],
      charts:[],notes:[],
    },
  };
  await page.route("**/api/v1/**",route=>route.fulfill({status:404,json:{detail:"fixture endpoint not configured"}}));
  await page.route("**/api/v1/experiment-lab/catalog",route=>route.fulfill({json:{...catalog,data_ready:true,statistics:{ratings:100,users:20,movies:12,catalog_movies:12,genres:5,rating_min:0.5,rating_max:5,rating_mean:3.5,sparsity:0.4}}}));
  await page.route("**/api/v1/experiment-lab/runs",route=>route.fulfill({json:[legacyRun]}));
  await page.route("**/api/v1/experiment-lab/runs/legacy-pca-fixture/logs?*",route=>route.fulfill({json:{entries:[],next_cursor:0,has_more:false,reset:false}}));
  await page.route("**/api/v1/experiment-lab/runs/legacy-pca-fixture",route=>route.fulfill({json:legacyRun}));

  const primary=["rating","classification","clustering","association","reinforcement"];
  const iconByGroup={rating:"activity",classification:"target",clustering:"layers",association:"network",reinforcement:"flask-conical"};
  const marker={rating:"Funk SVD",classification:"threshold",clustering:"genre gốc",association:"transaction boolean",reinforcement:"action mask"};
  for(const group of primary){
    await page.goto(`${base}/experiments?group=${group}`);
    const nav=page.locator(".lab-sidebar nav button");
    await nav.last().waitFor();
    assert.equal(await nav.count(),5);
    assert.deepEqual(await nav.evaluateAll(items=>items.map(item=>item.dataset.groupId)),primary);
    for(const [id,icon] of Object.entries(iconByGroup)){
      assert.equal(await page.locator(`.lab-sidebar nav button[data-group-id="${id}"] svg[class*="lucide-${icon}"]`).count(),1,`Icon should be keyed by ${id}`);
    }
    const guide=page.getByTestId("lab-reduction-guide");
    assert.equal(await guide.getAttribute("data-group-id"),group);
    assert.equal(await guide.getAttribute("data-mode"),"off");
    assert.equal(await guide.locator(".lab-reduction-card").count(),2);
    assert.equal(await page.getByTestId("reduction-card-pca").getAttribute("data-active"),"false");
    assert.equal(await page.getByTestId("reduction-card-truncated_svd").getAttribute("data-active"),"false");
    assert.ok((await guide.innerText()).includes(marker[group]),`Missing ${group} reducer explanation`);
    assert.deepEqual(await guide.locator(".lab-reduction-grid").evaluate(el=>getComputedStyle(el).gridTemplateColumns.split(" ").length),2);
    assert.deepEqual(await page.evaluate(()=>{
      const guide=document.querySelector(".lab-reduction-guide");
      return [guide?.previousElementSibling?.classList.contains("lab-algorithms"),
        guide?.nextElementSibling?.classList.contains("lab-section"),
        !!guide?.nextElementSibling?.querySelector(".lab-criteria")];
    }),[true,true,true]);
  }

  await page.goto(`${base}/experiments?group=rating`);
  const guide=page.getByTestId("lab-reduction-guide");
  await page.locator(".lab-parameter-details summary").click();
  await page.locator(".lab-config select").first().selectOption("enabled");
  await page.locator(".lab-config select[multiple]").selectOption(["truncated_svd"]);
  await page.locator(".lab-config label").filter({hasText:"Components cần thử"}).locator("input").fill("1");
  assert.equal(await guide.getAttribute("data-mode"),"enabled");
  assert.equal(await page.getByTestId("reduction-card-pca").getAttribute("data-active"),"false");
  assert.equal(await page.getByTestId("reduction-card-truncated_svd").getAttribute("data-active"),"true");
  for(const theme of ["dark","light"]){
    await page.evaluate(value=>{document.documentElement.dataset.theme=value;},theme);
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),`Mobile overflow in ${theme}`);
    assert.equal(await page.locator(".lab-reduction-grid").evaluate(el=>getComputedStyle(el).gridTemplateColumns.split(" ").length),1);
  }

  await page.setViewportSize({width:1440,height:1000});
  await page.goto(`${base}/experiments?group=pca`);
  await page.getByTestId("legacy-pca-page").waitFor();
  assert.equal(await page.locator(".lab-sidebar nav button").count(),5);
  assert.equal(await page.locator(".lab-algorithm-select").count(),0);
  assert.equal(await page.locator(".lab-history select option").count(),1);
  for(const format of ["csv","xlsx","json"]){
    assert.ok(await page.locator(`.lab-results .lab-downloads a[href$="/export/${format}"]`).count(),`Missing legacy ${format} export`);
  }
  await page.evaluate(()=>localStorage.setItem("movie-language","en"));
  await page.goto(`${base}/experiments?group=rating`);
  assert.ok((await page.getByTestId("lab-reduction-guide").innerText()).includes("Dimensionality reduction"));
  assert.deepEqual(errors,[]);
  console.log("PASS five primary nav groups, group-keyed guide order, live mode/method state, bilingual content, responsive cards, and legacy PCA history/CSV-XLSX-JSON links.");
}finally{
  await browser?.close();
  server.kill();
}
