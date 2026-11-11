"""Local report demo: honest validation-selected baseline and ML challengers."""

import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from movie_rating.services.recommendations import RecommendationService
from report_temporal_experiments import filter_training

ROOT=Path(__file__).resolve().parents[1]
POINTER=json.loads((ROOT/'artifacts/report-latest.json').read_text(encoding='utf-8'))
RUN=Path(POINTER['directory'])
RESULT=json.loads((RUN/'results.json').read_text(encoding='utf-8'))
ROWS=RESULT['supervised']['rows']
RATING_IDS=('linear','xgboost','svd','ncf','mean','item_knn')
SELECTED=min([r for r in ROWS if r['split']=='validation' and r['algorithm'] in RATING_IDS],key=lambda r:r['rmse'])['algorithm']
FIT=filter_training(pd.concat([pd.read_parquet(RUN/f'{s}.parquet') for s in ('train','validation')],ignore_index=True))
MOVIES=pd.read_parquet(ROOT/'artifacts/data/movies.parquet').set_index('movieId')
POPULARITY=FIT.groupby('movieId').size().sort_values(ascending=False,kind='stable')
MODELS={}
app=FastAPI(title='MovieLens — demo của báo cáo theo thời gian')


class Rating(BaseModel):
    movie_id:int=Field(gt=0)
    rating:float=Field(ge=.5,le=5,multiple_of=.5)


class Request(BaseModel):
    history:list[Rating]=Field(default_factory=list,max_length=100)
    algorithm:Literal['auto','linear','xgboost','svd','ncf']='auto'
    limit:int=Field(default=10,ge=1,le=20)


@app.get('/api/report/metadata')
def metadata():
    return {'run_id':RESULT['run_id'],'selected_algorithm':SELECTED,
            'selection':'lowest Validation RMSE including baselines',
            'models':list(RATING_IDS),'source':'real MovieLens temporal run',
            'movies':[{'id':int(m),'title':str(MOVIES.loc[m,'title'])} for m in POPULARITY.head(80).index]}


@app.post('/api/report/recommend')
def recommend(request:Request):
    history=[r.model_dump() for r in request.history]
    seen={r['movie_id'] for r in history}
    if any(m not in MOVIES.index for m in seen):raise HTTPException(422,'Movie ID không thuộc catalog.')
    algorithm=SELECTED if request.algorithm=='auto' else request.algorithm
    if algorithm=='mean':
        scores=pd.Series(float(FIT.rating.mean()),index=POPULARITY.index)
        # All score ties are broken by Train+Validation popularity, never Test.
        strategy='Mean Rating + phá hòa bằng popularity; baseline an toàn, chưa cá nhân hóa'
    else:
        if algorithm not in MODELS:MODELS[algorithm]=joblib.load(RUN/f'{algorithm}.joblib')
        model=MODELS[algorithm]
        scores=RecommendationService.profile_scores(model,history,algorithm) if history else None
        if scores is None:
            scores=pd.Series(float(FIT.rating.mean()),index=POPULARITY.index)
            strategy='Chưa có lịch sử hợp lệ: fallback mean/popularity'
        else:strategy='Profile fold-in; chỉ cập nhật hồ sơ user mới, giữ model phim cố định'
    candidates=scores.loc[~scores.index.isin(seen)].sort_values(ascending=False,kind='stable').head(request.limit)
    return {'algorithm':algorithm,'strategy':strategy,'selection_split':'validation',
            'items':[{'movie_id':int(m),'title':str(MOVIES.loc[m,'title']),'predicted_rating':float(np.clip(v,.5,5))} for m,v in candidates.items()]}


@app.get('/',response_class=HTMLResponse)
def home():
    return '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>MovieLens · Báo cáo thực nghiệm</title>
<style>body{font:16px system-ui;background:#f1f4f8;color:#173149;margin:0}main{max-width:1000px;margin:40px auto;padding:32px;background:white;border-radius:16px}h1{margin-top:0}label{display:block;margin:18px 0}select,input,button{font:inherit;padding:10px;border:1px solid #b6c3cf;border-radius:6px}select{max-width:100%}button{background:#244f78;color:white;cursor:pointer}table{width:100%;border-collapse:collapse;margin-top:18px}th,td{text-align:left;padding:12px;border-bottom:1px solid #dce2ea}.note{background:#e9eff5;padding:16px;border-radius:8px;line-height:1.7}.row{display:flex;gap:12px;flex-wrap:wrap}.history{padding:10px 0}small{color:#536878}</style>
<main><small>MOVIELENS 25M / DEMO CỦA BÁO CÁO THEO THỜI GIAN</small><h1>Gợi ý phim từ lịch sử đánh giá</h1>
<p class="note" id="meta">Đang đọc artifact thực nghiệm…</p><div class="row"><label>Phim đã xem<br><select id="movie"></select></label><label>Rating<br><input id="rating" type="number" min="0.5" max="5" step="0.5" value="4.5"></label></div><button id="add">Thêm vào lịch sử</button><div id="history" class="history"></div>
<label>Mô hình<br><select id="algorithm"><option value="auto">Auto · chọn bằng Validation, có xét baseline</option><option value="xgboost">XGBoost · ứng viên ML tốt nhất trên Validation</option><option value="linear">Linear Regression</option><option value="svd">SVD Collaborative Filtering</option><option value="ncf">Neural Collaborative Filtering</option></select></label><button id="run">Gợi ý Top-10</button><p id="status" role="status"></p><table><thead><tr><th>Phim gợi ý</th><th>Rating dự đoán</th></tr></thead><tbody id="results"></tbody></table><p><small>Kết quả từ model thật. Phim đã đánh giá bị loại khỏi gợi ý; không suy ra hiệu quả online từ benchmark offline.</small></p></main>
<script>const history=[];let movies=[];const $=id=>document.getElementById(id);fetch('/api/report/metadata').then(r=>r.json()).then(m=>{movies=m.movies;$('meta').textContent=`Run ${m.run_id}. Auto chọn ${m.selected_algorithm} vì RMSE Validation thấp nhất khi có xét baseline. XGBoost tốt nhất trong 4 mô hình học nhưng chưa vượt Mean trên Validation; lựa chọn mô hình không dựa vào Test.`;for(const item of movies){const o=document.createElement('option');o.value=item.id;o.textContent=item.title;$('movie').append(o)}});$('add').onclick=()=>{const movie=Number($('movie').value),rating=Number($('rating').value);if(!Number.isFinite(rating)||rating<.5||rating>5||rating*2%1)return;const old=history.find(r=>r.movie_id===movie);if(old)old.rating=rating;else history.push({movie_id:movie,rating});$('history').textContent=history.map(h=>`${movies.find(m=>m.id===h.movie_id)?.title}: ${h.rating} sao`).join(' · ')};$('run').onclick=async()=>{$('status').textContent='Đang dự đoán…';try{const response=await fetch('/api/report/recommend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({history,algorithm:$('algorithm').value,limit:10})});const data=await response.json();if(!response.ok)throw Error(JSON.stringify(data.detail));$('status').textContent=`${data.algorithm} · ${data.strategy}`;$('results').replaceChildren();for(const item of data.items){const tr=document.createElement('tr');for(const text of [item.title,item.predicted_rating.toFixed(3)]){const td=document.createElement('td');td.textContent=text;tr.append(td)}$('results').append(tr)}}catch(e){$('status').textContent=e.message}};</script></html>'''
