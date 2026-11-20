// Real MovieLens runs through the UI; no mocked metrics or model responses.
import assert from "node:assert/strict";
import { access, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { chromium } from "playwright";

const frontend=fileURLToPath(new URL("../",import.meta.url));
const project=path.resolve(frontend,"..");
const output=path.join(frontend,"test-results");
const base="http://127.0.0.1:8139";
let executablePath;
for(const candidate of [process.env.CHROME_PATH,"C:/Program Files/Google/Chrome/Application/chrome.exe","C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"].filter(Boolean)){
  try{await access(candidate);executablePath=candidate;break;}catch{}
}
assert.ok(executablePath,"Set CHROME_PATH to an installed Chromium browser");
const python=process.env.MOVIE_TEST_PYTHON||path.join(project,".venv/Scripts/python.exe");
const server=spawn(python,["-X","utf8","-m","uvicorn","movie_rating.api.app:app","--host","127.0.0.1","--port","8139"],{cwd:project,windowsHide:true,stdio:["ignore","pipe","pipe"]});
const logs=[];
server.stdout.on("data",d=>logs.push(d.toString()));server.stderr.on("data",d=>logs.push(d.toString()));
server.on("error",e=>logs.push(e.message));
let browser;
const summary=[];
async function waitForRun(runId,timeoutMs){
  const deadline=Date.now()+timeoutMs;
  let lastStatus="unknown";
  while(Date.now()<deadline){
    const response=await fetch(`${base}/api/v1/experiment-lab/runs/${runId}`);
    const run=await response.json();
    if(!response.ok)throw new Error(`Could not read run ${runId}: ${JSON.stringify(run)}`);
    lastStatus=run.status;
    if(lastStatus==="completed"||lastStatus==="failed")return run;
    await new Promise(resolve=>setTimeout(resolve,500));
  }
  throw new Error(`Run ${runId} did not finish before timeout; last status: ${lastStatus}`);
}
try{
  for(let attempt=0;attempt<150;attempt++){
    try{if((await fetch(`${base}/api/v1/experiment-lab/catalog`)).ok)break;}catch{}
    if(attempt===149)throw new Error(`Server did not start: ${logs.join("").slice(-3000)}`);
    await new Promise(r=>setTimeout(r,200));
  }
  const catalog=await(await fetch(`${base}/api/v1/experiment-lab/catalog`)).json();
  assert.equal(catalog.groups.length,6,'Legacy PCA stays in the API catalog for old history');assert.ok(catalog.statistics.ratings>0);
  const primaryGroups=catalog.groups.filter(group=>group.id!=="pca");
  assert.deepEqual(primaryGroups.map(group=>group.id),["rating","classification","clustering","association","reinforcement"]);
  await mkdir(output,{recursive:true});
  browser=await chromium.launch({executablePath,headless:true});
  const page=await browser.newPage({viewport:{width:1440,height:1000},reducedMotion:"reduce"});
  const errors=[];page.on("pageerror",e=>errors.push(e.message));
  const iconByGroup={rating:"activity",classification:"target",clustering:"layers",association:"network",reinforcement:"flask-conical"};
  const guideMarker={rating:"Funk SVD",classification:"threshold",clustering:"genre gốc",association:"transaction boolean",reinforcement:"action mask"};
  for(const group of primaryGroups){
    await page.goto(`${base}/experiments?group=${group.id}`);
    await page.locator(".lab-sidebar nav button").last().waitFor();
    assert.equal(await page.locator(".lab-sidebar nav button").count(),5);
    for(const [id,icon] of Object.entries(iconByGroup)){
      assert.ok((await page.locator(`.lab-sidebar nav button[data-group-id="${id}"] svg`).getAttribute("class")).includes(`lucide-${icon}`),`Icon for ${id} must be keyed by group ID`);
    }
    assert.equal(await page.locator(".lab-algorithms article").count(),group.algorithms.length);
    const guide=page.locator('.lab-reduction-guide');
    assert.equal(await guide.count(),1);
    assert.equal(await guide.getAttribute('data-group-id'),group.id);
    assert.equal(await guide.getAttribute('data-mode'),'enabled');
    assert.ok((await guide.innerText()).includes(guideMarker[group.id]),`Missing group-specific reduction details for ${group.id}`);
    assert.equal(await page.getByTestId('reduction-card-pca').getAttribute('data-active'),'true');
    assert.equal(await page.getByTestId('reduction-card-truncated_svd').getAttribute('data-active'),'false');
    assert.equal(await guide.locator('.lab-reduction-card').count(),2);
    assert.deepEqual(await page.evaluate(()=>{
      const cards=[...document.querySelectorAll('.lab-content > .lab-reduction-guide')];
      const guide=cards[0],algorithms=guide?.previousElementSibling,metrics=guide?.nextElementSibling;
      return [algorithms?.classList.contains('lab-algorithms'),metrics?.classList.contains('lab-section'),!!metrics?.querySelector('.lab-criteria')];
    }),[true,true,true]);
    assert.equal((await guide.locator('.lab-reduction-grid').evaluate(el=>getComputedStyle(el).gridTemplateColumns.split(' ').length)),2);
    const parameters=page.locator('.lab-parameter-details');
    assert.equal(await parameters.getAttribute('open'),null);
    assert.equal(await page.locator('.lab-criteria input[type="number"]').count(),group.metrics.length);
    await parameters.locator('summary').click();
    if(group.id!=="association"){
      const allData=page.getByLabel("Dùng toàn bộ dữ liệu đã chuẩn bị",{exact:true});
      if(await allData.isChecked())await allData.uncheck();
      await page.getByLabel("Số user tối đa",{exact:true}).fill("40");
    }
    if(group.id==="rating"){
      await page.getByLabel("Epochs SVD / NCF",{exact:true}).fill("2");
      await page.getByLabel("XGBoost · Trees",{exact:true}).fill("10");
      await page.getByLabel("XGBoost · Max depth",{exact:true}).fill("2");
      await page.locator(".lab-config select").first().selectOption("compare");
      await page.locator(".lab-config select[multiple]").selectOption(["pca","truncated_svd"]);
      await page.locator(".lab-config label").filter({hasText:"Components cần thử"}).locator("input").fill("1");
    }
    if(["rating","classification"].includes(group.id)){
      const tuning=page.getByLabel("Tuning cấu hình nhẹ / mặc định / mạnh hơn",{exact:true});
      assert.equal(await tuning.isChecked(),false);
      await tuning.uncheck();
    }
    if(group.id==="reinforcement"){
      await page.getByLabel("DQN · Giới hạn training steps",{exact:true}).fill("100");
      await page.getByLabel("DQN · User Train tối đa",{exact:true}).fill("40");
    }
    if(group.id==="clustering"){
      await page.locator(".lab-config select").first().selectOption("compare");
      await page.locator(".lab-config select[multiple]").selectOption(["pca","truncated_svd"]);
      await page.locator(".lab-config label").filter({hasText:"Components cần thử"}).locator("input").fill("1");
      await page.locator(".lab-config label").filter({hasText:"K-Means · Danh sách K"}).locator("input").fill("2");
      await page.locator(".lab-config label").filter({hasText:"DBSCAN · Danh sách min_samples"}).locator("input").fill("5");
    }
    await parameters.locator('summary').click();
    assert.equal(await parameters.getAttribute('open'),null);
    const button=page.locator(".lab-primary");
    await button.waitFor();await page.waitForFunction(()=>!document.querySelector(".lab-primary")?.disabled);
    const responsePromise=page.waitForResponse(r=>r.url().endsWith("/experiment-lab/runs")&&r.request().method()==="POST");
    await button.click();const response=await responsePromise;assert.equal(response.status(),202);
    const job=await response.json();
    const finished=await waitForRun(job.id,240000);
    assert.equal(finished.status,"completed",finished.message);
    await page.locator(".lab-results .lab-comparison tbody tr").first().waitFor();
    assert.equal(finished.result.schema_version,4);
    if(group==="rating"){
      assert.equal(finished.request.reduction_mode,"compare");
      assert.deepEqual(finished.request.reduction_methods,["pca","truncated_svd"]);
      assert.deepEqual(finished.request.components,[1]);
      for(const variant of ["ncf_pca_1","ncf_truncated_svd_1"]){
        assert.ok(finished.result.rows.some(row=>row.variant_id===variant&&row.split==="test"));
        assert.ok(finished.result.charts.some(chart=>chart.title.startsWith("NCF hybrid · Correction Branch")&&chart.title.includes(variant.slice(4))));
      }
    }
    if(group==="clustering"){
      assert.equal(finished.request.reduction_mode,"compare");
      for(const method of ["pca","truncated_svd"]){
        assert.ok(finished.result.rows.some(row=>row.variant_id===`kmeans_${method}_1`));
        assert.ok(finished.result.rows.some(row=>row.variant_id===`dbscan_${method}_1`));
        assert.ok(finished.result.search.some(row=>row.variant_id===`dbscan_${method}_1`));
      }
    }
    if(group.id!=="association"){
      assert.equal(finished.request.max_users,40,'Collapsing parameters must preserve edited values');
      assert.equal(finished.request.use_all_data,false,'Full-data toggle must preserve the sample-run choice');
    }else{
      assert.equal(finished.request.use_all_data,true,'Association Rules always use the complete prepared Train split');
    }
    assert.ok(await page.locator('.lab-comparison [data-verdict]').count()>0);
    assert.equal(await page.locator(".lab-chart").count(),finished.result.charts.length);
    assert.equal(await page.locator(".lab-results .lab-comparison tbody tr").count(),finished.result.rows.length);
    assert.ok(await page.locator(".lab-stat-grid").count()>=2);
    if(["rating","classification"].includes(group.id)){
      assert.equal(Object.keys(finished.result.cv).length,group.algorithms.length);
      assert.ok(Object.values(finished.result.cv).every(c=>c.folds.length===3));
      assert.ok((await page.locator(".lab-results").innerText()).includes("Mean ± Std"));
    }
    if(group.id==="rating"){
      assert.ok(await page.getByRole("img",{name:/NCF · Rating MSE \/ Early Stopping/}).count());
      assert.ok(await page.getByRole("img",{name:/NCF hybrid · Correction Branch.*pca_1/}).count());
      assert.ok(await page.getByRole("img",{name:/NCF hybrid · Correction Branch.*truncated_svd_1/}).count());
    }
    if(group.id==="classification")assert.ok((await page.locator(".lab-results").innerText()).includes("Confusion Matrix"));
    if(group.id==="classification"){
      const name=`Mốc ${group.metrics.find(m=>m.key==='accuracy').name}`;
      await page.getByLabel(name,{exact:true}).fill('1');
      await page.getByLabel(name,{exact:true}).press('Tab');
      assert.equal(await page.locator('.lab-comparison [data-metric="accuracy"] [data-verdict="bad"]').count(),finished.result.rows.length);
      await page.reload();
      await page.locator('.lab-results .lab-comparison tbody tr').first().waitFor();
      assert.equal(await page.getByLabel(name,{exact:true}).inputValue(),'1');
      const unchanged=await(await page.request.get(`${base}/api/v1/experiment-lab/runs/${job.id}`)).json();
      assert.deepEqual(unchanged.result.rows,finished.result.rows,'Changing a reference must not change model metrics');
      await page.getByRole('button',{name:'Khôi phục mốc mặc định',exact:true}).click();
      assert.equal(await page.getByLabel(name,{exact:true}).inputValue(),'0.7');
    }
    if(group.id==="association"){
      await page.getByLabel("Sắp xếp luật",{exact:true}).selectOption("confidence");
      assert.equal(await page.getByLabel("Sắp xếp luật",{exact:true}).inputValue(),"confidence");
    }
    for(const format of ["csv","xlsx","json"]){
      const exported=await page.request.get(`${base}/api/v1/experiment-lab/runs/${job.id}/export/${format}`);
      assert.equal(exported.status(),200);
      if(format==="xlsx")assert.equal((await exported.body()).subarray(0,2).toString(),"PK");
    }
    await page.locator(".lab-results").scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(output,`lab-${group.id}-results.png`)});
    for(const theme of ["dark","light"]){
      await page.evaluate(v=>{document.documentElement.dataset.theme=v;},theme);
      await page.setViewportSize({width:390,height:844});
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),`Overflow ${group.id}/${theme}`);
      assert.equal(await page.locator('.lab-reduction-grid').evaluate(el=>getComputedStyle(el).gridTemplateColumns.split(' ').length),1,`Reduction cards must stack on mobile (${group.id}/${theme})`);
    }
    await page.screenshot({path:path.join(output,`lab-${group.id}-mobile.png`)});
    await page.setViewportSize({width:1440,height:1000});
    await page.evaluate(()=>{document.documentElement.dataset.theme="dark";});
    await page.reload();await page.locator(".lab-results .lab-comparison tbody tr").first().waitFor();
    summary.push({group:group.id,run_id:job.id,rows:finished.result.rows.length,charts:finished.result.charts.length,ratings:finished.result.provenance.sampled_ratings});
    console.log(`PASS ${group.id}: ${finished.result.rows.length} rows, ${finished.result.charts.length} charts`);
  }
  await page.goto(`${base}/experiments?group=classification`);
  await page.locator(".lab-algorithm-select input").last().waitFor();
  for(const box of await page.locator(".lab-algorithm-select input").all())await box.uncheck();
  assert.ok(await page.locator(".lab-primary").isDisabled());
  await page.locator(".lab-algorithm-select input").first().check();
  assert.equal(await page.locator(".lab-primary").innerText(),"Chạy thuật toán đã chọn");
  await page.locator('.lab-parameter-details summary').click();
  const allData=page.getByLabel("Dùng toàn bộ dữ liệu đã chuẩn bị",{exact:true});
  if(await allData.isChecked())await allData.uncheck();
  await page.getByLabel("Số user tối đa",{exact:true}).fill('');
  await page.locator('.lab-parameter-details summary').click();
  await page.locator('.lab-primary').click();
  assert.notEqual(await page.locator('.lab-parameter-details').getAttribute('open'),null,'Invalid hidden parameters must expand for correction');
  await page.getByLabel("Số user tối đa",{exact:true}).fill("20");
  await page.getByLabel("Tuning cấu hình nhẹ / mặc định / mạnh hơn",{exact:true}).uncheck();
  const singleResponse=page.waitForResponse(r=>r.url().endsWith("/experiment-lab/runs")&&r.request().method()==="POST");
  await page.locator(".lab-primary").click();const single=await(await singleResponse).json();
  assert.deepEqual(single.request.algorithms,["logistic"]);
   await waitForRun(single.id,120000);
  assert.equal(await page.locator(".lab-job-failed").count(),0);
  await page.locator(".lab-content").scrollIntoViewIfNeeded();
  await page.screenshot({path:path.join(output,"lab-overview-desktop.png")});
  await page.route("**/experiment-lab/catalog",route=>route.fulfill({status:503,json:{detail:"Kiểm thử lỗi kết nối"}}));
  await page.reload();await page.getByText("Kiểm thử lỗi kết nối",{exact:true}).waitFor();
  assert.equal(await page.locator(".lab-comparison").count(),0);
  await page.unroute("**/experiment-lab/catalog");
  await page.getByRole("button",{name:"Thử lại",exact:true}).click();
  await page.locator(".lab-algorithms article").last().waitFor();
  await page.goto(`${base}/experiments?group=pca`);
  await page.getByTestId("legacy-pca-page").waitFor();
  assert.equal(await page.locator(".lab-sidebar nav button").count(),5);
  assert.equal(await page.locator(".lab-legacy-pca").count(),1);
  assert.equal(await page.locator(".lab-algorithm-select").count(),0,'Legacy route must not expose standalone PCA jobs');
  assert.ok((await page.locator(".lab-content").innerText()).includes("Lịch sử PCA cũ"));
  await page.evaluate(()=>localStorage.setItem("movie-language","en"));
  await page.goto(`${base}/experiments?group=rating`);
  await page.getByTestId("lab-reduction-guide").waitFor();
  assert.ok((await page.locator(".lab-reduction-guide").innerText()).includes("Dimensionality reduction"));
  assert.deepEqual(errors,[]);
  await writeFile(path.join(output,"lab-real-runs.json"),JSON.stringify(summary,null,2));
  console.log("PASS five primary groups, shared reduction guide, real training, 3-Fold CV, single/multiple selection, legacy PCA history route, exports, mobile dark/light and error recovery.");
}finally{await browser?.close();server.kill();}
