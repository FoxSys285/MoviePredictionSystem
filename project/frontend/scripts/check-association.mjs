// Full prepared Train, association-only controls, persistent logs, and responsive rendering.
import assert from "node:assert/strict";
import {access,mkdir,readFile,writeFile} from "node:fs/promises";
import path from "node:path";
import {fileURLToPath} from "node:url";
import {spawn} from "node:child_process";
import {chromium} from "playwright";

const frontend=fileURLToPath(new URL("../",import.meta.url));
const project=path.resolve(frontend,"..");
const base="http://127.0.0.1:8141";
const output=path.join(frontend,"test-results");
const manifest=JSON.parse(await readFile(path.join(project,"artifacts/data/manifest.json"),"utf8"));
let executablePath;
for(const candidate of [process.env.CHROME_PATH,"C:/Program Files/Google/Chrome/Application/chrome.exe","C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"].filter(Boolean)){
  try{await access(candidate);executablePath=candidate;break;}catch{}
}
assert.ok(executablePath,"Set CHROME_PATH to an installed Chromium browser");
const server=spawn(process.env.MOVIE_TEST_PYTHON||path.join(project,".venv/Scripts/python.exe"),
  ["-X","utf8","-m","uvicorn","movie_rating.api.app:app","--host","127.0.0.1","--port","8141"],
  {cwd:project,windowsHide:true,stdio:["ignore","pipe","pipe"]});
const serverOutput=[];
server.stdout.on("data",value=>serverOutput.push(value.toString()));
server.stderr.on("data",value=>serverOutput.push(value.toString()));
let browser;
try{
  for(let attempt=0;attempt<150;attempt++){
    try{if((await fetch(`${base}/api/v1/health`)).ok)break;}catch{}
    if(attempt===149)throw new Error(serverOutput.join("").slice(-3000));
    await new Promise(resolve=>setTimeout(resolve,200));
  }
  await mkdir(output,{recursive:true});
  browser=await chromium.launch({executablePath,headless:true});
  const page=await browser.newPage({viewport:{width:1440,height:1000},reducedMotion:"reduce"});
  const errors=[];page.on("pageerror",error=>errors.push(error.message));
  await page.goto(`${base}/experiments?group=rating`);
  await page.locator(".lab-parameter-details summary").click();
  await page.getByLabel("Dùng toàn bộ dữ liệu đã chuẩn bị",{exact:true}).uncheck();
  await page.getByRole("button",{name:/Association Rules/}).click();
  const reductionGuide=page.getByTestId("lab-reduction-guide");
  assert.equal(await reductionGuide.getAttribute("data-group-id"),"association");
  assert.equal(await reductionGuide.locator(".lab-reduction-card").count(),2);
  await page.locator(".lab-parameter-details summary").click();
  assert.equal(await page.getByLabel("Phạm vi khai phá",{exact:true}).inputValue(),"Toàn bộ tập Train đã chuẩn bị");
  assert.equal(await page.getByLabel("Số user tối đa",{exact:true}).count(),0);
  assert.equal(await page.getByLabel("Dùng toàn bộ dữ liệu đã chuẩn bị",{exact:true}).count(),0);
  assert.equal(await page.getByLabel("Giới hạn phim ứng viên",{exact:true}).inputValue(),"80");
  await page.getByLabel("min_support",{exact:true}).fill("0.01");
  await page.locator(".lab-config select").first().selectOption("enabled");
  await page.locator(".lab-config label").filter({hasText:"Components cần thử"}).locator("input").fill("1");
  assert.equal(await reductionGuide.getAttribute("data-mode"),"enabled");
  assert.equal(await page.getByTestId("reduction-card-pca").getAttribute("data-active"),"true");
  assert.equal(await page.getByTestId("reduction-card-truncated_svd").getAttribute("data-active"),"true");
  await page.locator(".lab-config label").filter({hasText:"Apriori · K số nhóm user"}).locator("input").fill("2");
  const source=page.getByRole("region",{name:"Nguồn dữ liệu Apriori",exact:true});
  assert.ok((await source.innerText()).includes(manifest.splits.train.ratings.toLocaleString("vi-VN")));
  await source.screenshot({path:path.join(output,"association-source-desktop.png")});
  await page.waitForFunction(()=>!document.querySelector(".lab-primary")?.disabled);
  const response=page.waitForResponse(r=>r.url().endsWith("/experiment-lab/runs")&&r.request().method()==="POST");
  await page.locator(".lab-primary").click();
  const createdResponse=await response;
  const payload=createdResponse.request().postDataJSON();
  assert.equal(payload.use_all_data,true,"Shared sampling state must not limit association Train");
  const started=await createdResponse.json();
  await page.locator(".lab-job-completed,.lab-job-failed").waitFor({timeout:180000});
  const finished=await(await page.request.get(`${base}/api/v1/experiment-lab/runs/${started.id}`)).json();
  assert.equal(finished.status,"completed",finished.message);
  assert.equal(finished.request.use_all_data,true);
  assert.equal(finished.request.reduction_mode,"enabled");
  assert.deepEqual(finished.request.reduction_methods,["pca","truncated_svd"]);
  assert.deepEqual(finished.request.components,[1]);
  const data=finished.result.association_data;
  assert.equal(data.train_ratings,manifest.splits.train.ratings);
  assert.equal(data.train_users,manifest.splits.train.users);
  assert.equal(data.candidate_movies,80);
  assert.deepEqual(finished.request.k_values,[2]);
  assert.ok(finished.result.grouped_variants.some(variant=>variant.k===2));
  assert.ok(finished.result.group_rules.length>0);
  assert.equal(finished.result.provenance.association_source.prepared_user_sample_fraction,manifest.settings.data.user_sample_fraction);
  await page.locator(".lab-log-success").last().waitFor();
  const text=await page.locator(".lab-log-viewport").innerText();
  for(const step of ["train.parquet","Transactions","min_support","min_confidence","Validation/Test"])assert.ok(text.includes(step),step);
  await page.getByRole("region",{name:"Phạm vi dữ liệu của lần chạy Apriori",exact:true}).waitFor();
  await page.getByRole("heading",{name:"Apriori theo nhóm · support và coverage",exact:true}).waitFor();
  await page.getByRole("heading",{name:"Luật Apriori trong nhóm",exact:true}).waitFor();
  await page.locator(".lab-results").screenshot({path:path.join(output,"association-grouped-results.png")});
  await page.getByText("Tệp dữ liệu của lần chạy",{exact:true}).click();
  await page.locator(".lab-log").screenshot({path:path.join(output,"association-logs-desktop.png")});
  const download=await page.request.get(`${base}/api/v1/experiment-lab/runs/${started.id}/logs/download`);
  assert.equal(download.status(),200);
  assert.ok((await download.text()).includes('"level": "success"'));
  const csv=await page.request.get(`${base}/api/v1/experiment-lab/runs/${started.id}/export/csv`);
  assert.equal(csv.status(),200);assert.ok((await csv.text()).includes("association_group_rule"));
  const xlsx=await page.request.get(`${base}/api/v1/experiment-lab/runs/${started.id}/export/xlsx`);
  assert.equal(xlsx.status(),200);assert.equal((await xlsx.body()).subarray(0,2).toString(),"PK");
  for(const theme of ["dark","light"]){
    await page.evaluate(value=>document.documentElement.dataset.theme=value,theme);
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),theme);
    await page.locator(".lab-log").screenshot({path:path.join(output,`association-logs-${theme}-mobile.png`)});
    await page.getByRole("region",{name:"Phạm vi dữ liệu của lần chạy Apriori",exact:true}).screenshot({path:path.join(output,`association-scope-${theme}-mobile.png`)});
  }
  await page.reload();
  await page.locator(".lab-log-success").last().waitFor();
  await page.getByRole("button",{name:/Classification/}).click();
  await page.locator(".lab-parameter-details summary").click();
  assert.ok(await page.getByLabel("Dùng toàn bộ dữ liệu đã chuẩn bị",{exact:true}).isVisible());
  // Reload resets defaults. Other groups still retain their own sampling controls.
  await page.getByLabel("Dùng toàn bộ dữ liệu đã chuẩn bị",{exact:true}).uncheck();
  assert.ok(await page.getByLabel("Số user tối đa",{exact:true}).isVisible());
  assert.deepEqual(errors,[]);
  const logs=await(await page.request.get(`${base}/api/v1/experiment-lab/runs/${started.id}/logs?limit=500`)).json();
  await writeFile(path.join(output,"association-check.json"),JSON.stringify({run_id:started.id,data,rows:finished.result.rows,grouped_variants:finished.result.grouped_variants,group_rules:finished.result.group_rules.length,log_lines:logs.entries.length,runtime:finished.result.runtime,checks:"full prepared Train, profile PCA and grouped Apriori, global baseline, support/fallback coverage, exports, persistent logs, dark/light mobile, other group controls"},null,2));
  console.log(`PASS full-Train association job ${started.id}: ${data.train_ratings} ratings, ${data.train_users} users, ${finished.result.group_rules.length} grouped rules, ${logs.entries.length} log lines.`);
}finally{
  await browser?.close();
  server.kill();
}
