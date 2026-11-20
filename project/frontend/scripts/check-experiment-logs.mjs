// Real group jobs, including reduced DQN, plus controlled log transport for scroll/reconnect behavior.
import assert from "node:assert/strict";
import {access,mkdir,writeFile} from "node:fs/promises";
import path from "node:path";
import {fileURLToPath} from "node:url";
import {spawn} from "node:child_process";
import {chromium} from "playwright";

const frontend=fileURLToPath(new URL("../",import.meta.url));
const project=path.resolve(frontend,"..");
const base="http://127.0.0.1:8140";
const output=path.join(frontend,"test-results");
let executablePath;
for(const candidate of [process.env.CHROME_PATH,"C:/Program Files/Google/Chrome/Application/chrome.exe","C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"].filter(Boolean)){
  try{await access(candidate);executablePath=candidate;break;}catch{}
}
assert.ok(executablePath,"Set CHROME_PATH to an installed Chromium browser");
const server=spawn(process.env.MOVIE_TEST_PYTHON||path.join(project,".venv/Scripts/python.exe"),
  ["-X","utf8","-m","uvicorn","movie_rating.api.app:app","--host","127.0.0.1","--port","8140"],
  {cwd:project,windowsHide:true,stdio:["ignore","pipe","pipe"]});
const serverOutput=[];
server.stdout.on("data",value=>serverOutput.push(value.toString()));
server.stderr.on("data",value=>serverOutput.push(value.toString()));
let browser;
const summary=[];
try{
  for(let attempt=0;attempt<150;attempt++){
    try{if((await fetch(`${base}/api/v1/experiment-lab/catalog`)).ok)break;}catch{}
    if(attempt===149)throw new Error(serverOutput.join("").slice(-3000));
    await new Promise(resolve=>setTimeout(resolve,200));
  }
  await mkdir(output,{recursive:true});
  browser=await chromium.launch({executablePath,headless:true});
  const page=await browser.newPage({viewport:{width:1440,height:1000},reducedMotion:"reduce"});
  const errors=[];page.on("pageerror",error=>errors.push(error.message));
  let latestJob,actualEvents;
  for(const group of ["rating","classification","clustering","reinforcement"]){
    await page.goto(`${base}/experiments?group=${group}`);
    await page.locator(".lab-algorithm-select input").last().waitFor();
    await page.locator(".lab-parameter-details summary").click();
    await page.getByLabel("Dùng toàn bộ dữ liệu đã chuẩn bị",{exact:true}).uncheck();
    await page.getByLabel("Số user tối đa",{exact:true}).fill("40");
    if(group==="rating"){
      await page.getByLabel("Epochs SVD / NCF",{exact:true}).fill("2");
      await page.getByLabel("XGBoost · Trees",{exact:true}).fill("10");
    }else if(group==="classification"){
      await page.getByLabel("Forest · Trees",{exact:true}).fill("10");
    }else if(group==="reinforcement"){
      await page.getByLabel("DQN · User Train tối đa",{exact:true}).fill("40");
      await page.getByLabel("DQN · Giới hạn training steps",{exact:true}).fill("100");
      await page.getByLabel("Replay warmup",{exact:true}).fill("8");
      await page.getByLabel("DQN · Batch size",{exact:true}).fill("8");
      await page.locator(".lab-config select").first().selectOption("compare");
      await page.locator(".lab-config label").filter({hasText:"Components cần thử"}).locator("input").fill("1");
    }
    await page.locator(".lab-parameter-details summary").click();
    await page.waitForFunction(()=>!document.querySelector(".lab-primary")?.disabled);
    const response=page.waitForResponse(r=>r.url().endsWith("/experiment-lab/runs")&&r.request().method()==="POST");
    await page.locator(".lab-primary").click();
    const started=await(await response).json();
    await page.locator(".lab-job-completed,.lab-job-failed").waitFor({timeout:180000});
    const finished=await(await page.request.get(`${base}/api/v1/experiment-lab/runs/${started.id}`)).json();
    assert.equal(finished.status,"completed",finished.message);
    await page.locator(".lab-log-success").last().waitFor();
    const text=await page.locator(".lab-log-viewport").innerText();
    const stages=group==="clustering"?["K-Means","DBSCAN","Hồ sơ","bao phủ","noise","ổn định","Recall@10","NDCG@10","Validation","Test"]:group==="reinforcement"?["DQN","reducer","checkpoint Validation","Train","Test"]:["Validation","CV","refit","Test"];
    for(const step of stages)assert.ok(text.includes(step),step);
    if(group==="rating")assert.ok(text.includes("cây"));
    if(group==="classification")assert.ok(text.includes("SAGA"));
    if(group==="reinforcement"){
      assert.ok(finished.result.reduction.variants.some(v=>v.method==="pca"&&v.components===1));
      assert.ok(finished.result.rows.some(row=>row.variant_id==="dqn_pca_1"&&row.split==="test"));
      assert.ok(finished.result.rows.some(row=>row.variant_id==="dqn_raw"&&row.split==="test"));
    }
    const desktop=await page.locator(".lab-log-viewport").boundingBox();
    assert.ok(Math.abs(desktop.height-500)<2);
    assert.equal(await page.locator(".lab-log-viewport").evaluate(e=>getComputedStyle(e).overflowY),"auto");
    assert.ok(await page.locator(".lab-log-viewport").evaluate(e=>e.scrollHeight>e.clientHeight));
    await page.locator(".lab-log").screenshot({path:path.join(output,`logs-${group}-desktop.png`)});
    for(const theme of ["dark","light"]){
      await page.evaluate(value=>document.documentElement.dataset.theme=value,theme);
      await page.setViewportSize({width:390,height:844});
      await page.waitForFunction(()=>{const e=document.querySelector(".lab-log-viewport");return e&&Math.abs(e.getBoundingClientRect().height-window.innerHeight/2)<2;});
      await page.waitForFunction(()=>{const e=document.querySelector(".lab-log-viewport");return e.scrollHeight-e.scrollTop-e.clientHeight<2;});
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
      const mobile=await page.locator(".lab-log-viewport").boundingBox();
      assert.ok(Math.abs(mobile.height-422)<2);
      await page.locator(".lab-log").screenshot({path:path.join(output,`logs-${group}-${theme}-mobile.png`)});
    }
    await page.setViewportSize({width:1440,height:1000});
    await page.evaluate(()=>document.documentElement.dataset.theme="dark");
    const download=await page.request.get(`${base}/api/v1/experiment-lab/runs/${started.id}/logs/download`);
    assert.equal(download.status(),200);
    assert.ok((await download.text()).includes('"message"'));
    await page.reload();
    await page.locator(".lab-log-success").last().waitFor();
    const all=await(await page.request.get(`${base}/api/v1/experiment-lab/runs/${started.id}/logs?limit=500`)).json();
    if(group==="rating"){
      latestJob=finished;
      actualEvents=all.entries;
    }
    summary.push({group,run_id:started.id,lines:all.entries.length,desktop_height:desktop.height});
  }

  // Hold the status running and append real recorded lines to exercise follow/pause semantics.
  await page.goto(`${base}/experiments?group=rating`);
  await page.locator(".lab-history select").selectOption(latestJob.id);
  await page.locator(".lab-log-success").last().waitFor();
  const logUrl=`**/experiment-lab/runs/${latestJob.id}/logs?*`;
  let available=actualEvents.slice(0,50),failOnce=false;
  await page.route(`**/experiment-lab/runs/${latestJob.id}`,route=>route.fulfill({json:{...latestJob,status:"running",result:undefined}}));
  await page.route(logUrl,route=>{
    if(failOnce){failOnce=false;return route.fulfill({status:503,json:{detail:"Temporary test outage"}});}
    const cursor=Number(new URL(route.request().url()).searchParams.get("cursor")||0);
    return route.fulfill({json:{entries:available.slice(cursor),next_cursor:available.length,has_more:false,reset:false}});
  });
  await page.reload();
  await page.waitForFunction(()=>document.querySelectorAll(".lab-log-line").length===50);
  const viewport=page.locator(".lab-log-viewport");
  await viewport.evaluate(e=>{e.scrollTop=0;});
  await page.getByRole("button",{name:"Về log mới nhất",exact:true}).waitFor();
  available=actualEvents;
  await page.waitForFunction(count=>document.querySelectorAll(".lab-log-line").length===count,available.length);
  assert.ok(await viewport.evaluate(e=>e.scrollTop<2),"New lines must preserve a paused scroll position");
  await page.getByRole("button",{name:"Về log mới nhất",exact:true}).click();
  await page.waitForFunction(()=>{const e=document.querySelector(".lab-log-viewport");return e.scrollHeight-e.scrollTop-e.clientHeight<2;});
  const previous=available.length;
  available=[...available,{...available.at(-1),id:available.at(-1).id+1,message:"Kiểm tra tự theo dõi log mới"}];
  await page.waitForFunction(count=>document.querySelectorAll(".lab-log-line").length===count,previous+1);
  assert.ok(await viewport.evaluate(e=>e.scrollHeight-e.scrollTop-e.clientHeight<2));
  failOnce=true;
  await page.getByText("Mất kết nối nhật ký, đang thử kết nối lại…",{exact:true}).waitFor();
  await page.locator(".lab-log-notice").waitFor({state:"hidden",timeout:10000});
  assert.equal(await page.locator(".lab-log-line").count(),available.length);
  await page.unrouteAll({behavior:"wait"});
  await page.goto(`${base}/experiments?group=pca`);
  await page.getByTestId("legacy-pca-page").waitFor();
  assert.equal(await page.locator(".lab-sidebar nav button").count(),5);
  assert.equal(await page.locator(".lab-algorithm-select").count(),0);
  assert.ok((await page.locator(".lab-legacy-pca").innerText()).includes("Đường dẫn PCA cũ"));
  assert.deepEqual(errors,[]);
  await writeFile(path.join(output,"experiment-logs-check.json"),JSON.stringify({summary,scroll:"pass",reconnect:"pass",other_groups:"unchanged"},null,2));
  console.log("PASS real regression/classification/clustering/reinforcement logs, half-screen desktop/mobile, history, download, pause/follow, reconnect and legacy PCA route.");
}finally{
  await browser?.close();
  server.kill();
}
