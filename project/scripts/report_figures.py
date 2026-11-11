"""Publication figures generated solely from the logged temporal experiment."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
POINTER = json.loads((ROOT/'artifacts/report-latest.json').read_text(encoding='utf-8'))
RUN = Path(POINTER['directory'])
DATA = json.loads((RUN/'results.json').read_text(encoding='utf-8'))
ASSETS = ROOT/'docs/report-assets'
ASSETS.mkdir(exist_ok=True)
font_path = Path('C:/Windows/Fonts/times.ttf')
if font_path.exists(): font_manager.fontManager.addfont(str(font_path))
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'axes.titlesize':12,
                     'axes.labelsize':11,'legend.fontsize':9,'figure.dpi':130,
                     'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False})
COLORS=['#244f78','#d88c37','#42857d','#8b5e8b','#757575','#a4a4a4']
LABEL={'linear':'Linear','xgboost':'XGBoost','svd':'SVD-CF','ncf':'NCF',
       'logistic':'Logistic','random_forest':'RF','mean':'Mean','item_knn':'Item-KNN',
       'q_learning':'Q-Learning','random':'Random','popular':'Popularity','kmeans':'K-Means','dbscan':'DBSCAN'}


def save(fig,name):
    fig.savefig(ASSETS/(name+'.png'),bbox_inches='tight',facecolor='white')
    fig.savefig(ASSETS/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def box(ax,xy,text,width=.24,height=.2,color='#e9f0f5'):
    rect=FancyBboxPatch(xy,width,height,boxstyle='round,pad=.015',fc=color,ec='#52708c',lw=1.2)
    ax.add_patch(rect)
    ax.text(xy[0]+width/2,xy[1]+height/2,text,ha='center',va='center',fontsize=11)


def diagram_figures():
    fig,ax=plt.subplots(figsize=(6.2,3.7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,(.04,.64),'Người dùng\nLịch sử đánh giá')
    box(ax,(.38,.64),'React / TypeScript\nWeb demo')
    box(ax,(.71,.64),'FastAPI\nAPI dự đoán')
    box(ax,(.71,.18),'Mô hình / baseline\nArtifact + cấu hình')
    box(ax,(.38,.18),'Pipeline thực nghiệm\nTrain – Val – Test')
    box(ax,(.04,.18),'MovieLens 25M\nCSV / Parquet')
    for a,b in [((.29,.74),(.36,.74)),((.63,.74),(.69,.74)),((.83,.62),(.83,.4)),((.63,.28),(.69,.28)),((.29,.28),(.36,.28))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#244f78','lw':1.6})
    save(fig,'system_architecture')
    fig,ax=plt.subplots(figsize=(6.2,3.5));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    box(ax,(.02,.66),'User ID\nEmbedding eᵤ',.24,.17)
    box(ax,(.02,.22),'Movie ID\nEmbedding eᵢ',.24,.17)
    box(ax,(.36,.43),'Ghép vector\n[eᵤ, eᵢ, eᵤ * eᵢ]',.27,.2)
    box(ax,(.73,.43),'MLP + ReLU\nDropout → 1 giá trị',.25,.2)
    for a,b in [((.27,.75),(.35,.55)),((.27,.3),(.35,.5)),((.64,.53),(.72,.53))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.5})
    ax.text(.68,.12,'r̂ᵤᵢ = μ + bᵤ + bᵢ + MLP(·)\nLoss MSE; đầu ra cắt về [0,5; 5]',ha='center',fontsize=11)
    save(fig,'ncf_architecture')
    parts=DATA['provenance']['splits']
    fig,ax=plt.subplots(figsize=(6.2,2.4));left=0
    for name,color in zip(('train','validation','test'),COLORS):
        value=parts[name]['ratings'];ax.barh(0,value,left=left,color=color,height=.6)
        ax.text(left+value/2,0,f'{name.title()}\n{value:,} ratings',ha='center',va='center',color='white',fontsize=11)
        left+=value
    ax.set(yticks=[],xlabel='Thứ tự thời gian tăng dần',xlim=(0,left));ax.set_xticks([])
    ax.text(.5,-.12,'Train: 5 cửa sổ thời gian → chọn trên Validation → refit 85% → Test',transform=ax.transAxes,ha='center',fontsize=10)
    save(fig,'temporal_protocol')


def dataset_figures():
    sample=pd.concat([pd.read_parquet(RUN/f'{s}.parquet') for s in ('train','validation','test')])
    values=sample.rating.value_counts().sort_index()
    fig,ax=plt.subplots(figsize=(6.2,3.3));ax.bar(values.index,values.values,color=COLORS[0],width=.35)
    ax.set(xlabel='Rating (sao)',ylabel='Số ratings trong mẫu',xticks=np.arange(.5,5.1,.5))
    ax.grid(axis='y',alpha=.2);save(fig,'rating_distribution')
    parts=DATA['provenance']['splits'];x=np.arange(3)
    fig,ax=plt.subplots(figsize=(6.2,3.3))
    ax.bar(x-.18,[parts[s]['known_user_fraction']*100 for s in parts],.36,label='User đã biết từ Train',color=COLORS[0])
    ax.bar(x+.18,[parts[s]['known_item_fraction']*100 for s in parts],.36,label='Phim đã biết từ Train',color=COLORS[1])
    ax.set(xticks=x,xticklabels=[s.title() for s in parts],ylim=(0,110),ylabel='Tỷ lệ ratings (%)');ax.legend(loc='upper right');save(fig,'cold_start_coverage')


def supervised_figures():
    sup=DATA['supervised'];rows=sup['rows'];aids=['linear','xgboost','svd','ncf']
    get=lambda a,s:next(r for r in rows if r['algorithm']==a and r['split']==s)
    fig,axes=plt.subplots(1,2,figsize=(6.2,3.5));x=np.arange(4)
    cv=[sup['cv'][a]['metrics']['rmse'] for a in aids]
    axes[0].bar(x,[s['mean'] for s in cv],yerr=[s['std'] for s in cv],capsize=4,color=COLORS[:4]);axes[0].set(title='5-fold theo thời gian',ylabel='RMSE (sao) ↓',xticks=x,xticklabels=[LABEL[a] for a in aids])
    axes[1].bar(x-.18,[get(a,'validation')['rmse'] for a in aids],.36,color=COLORS[0],label='Validation')
    axes[1].bar(x+.18,[get(a,'test')['rmse'] for a in aids],.36,color=COLORS[1],label='Test sau refit')
    axes[1].set(title='Validation và Test',xticks=x,xticklabels=[LABEL[a] for a in aids]);axes[1].legend()
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();save(fig,'regression_rmse')
    all_ids=aids+['mean','item_knn'];x=np.arange(len(all_ids))
    fig,ax=plt.subplots(figsize=(6.2,3.5));ax.bar(x-.18,[get(a,'validation')['mae'] for a in all_ids],.36,label='Validation',color=COLORS[0]);ax.bar(x+.18,[get(a,'test')['mae'] for a in all_ids],.36,label='Test',color=COLORS[1])
    ax.axhline(.75,color='#777',ls='--',label='Mốc MAE 0,75');ax.set(xticks=x,xticklabels=[LABEL[a] for a in all_ids],ylabel='MAE (sao) ↓');ax.legend(ncols=2,fontsize=9);save(fig,'regression_mae_baselines')
    fig,axes=plt.subplots(1,2,figsize=(6.2,3.3))
    for ax,key,title in [(axes[0],'fit_seconds','Fit cuối (s) ↓'),(axes[1],'inference_us_per_row','Inference (µs/rating) ↓')]:
        ax.bar(np.arange(4),[get(a,'test')[key] for a in aids],color=COLORS[:4]);ax.set(title=title,xticks=np.arange(4),xticklabels=[LABEL[a] for a in aids]);ax.ticklabel_format(axis='y',style='plain');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();save(fig,'regression_cost')
    curve=sup['learning_curves']['ncf'];fig,ax=plt.subplots(figsize=(6.2,3.4))
    ax.plot([r['epoch'] for r in curve],[r['train_rating_mse'] for r in curve],marker='o',color=COLORS[0],label='Train MSE')
    ax.plot([r['epoch'] for r in curve],[r['validation_rmse']**2 for r in curve],marker='s',color=COLORS[1],label='Validation MSE')
    ax.axvline(sup['best_configs']['ncf']['epochs'],color='#777',ls='--',label='Epoch được chọn')
    ax.set(xlabel='Epoch',ylabel='MSE rating ↓');ax.legend();ax.grid(alpha=.2);save(fig,'ncf_learning_curve')
    ids=['logistic','random_forest'];x=np.arange(3);fig,ax=plt.subplots(figsize=(6.2,3.5))
    for j,aid in enumerate(ids):ax.bar(x+(j-.5)*.34,[get(aid,'test')[k] for k in ['accuracy','f1','auc_roc']],.34,color=COLORS[j],label=LABEL[aid])
    ax.axhline(.7,color='#777',ls='--',label='Mốc dự án 0,70');ax.set(xticks=x,xticklabels=['Accuracy','F1 lớp Thích','AUC-ROC'],ylim=(0,1.04),ylabel='Giá trị Test ↑');ax.legend(ncols=2);save(fig,'classification_quality')
    test=pd.read_parquet(RUN/'test.parquet');fig,ax=plt.subplots(figsize=(6.2,3.7))
    for j,aid in enumerate(ids):
        probability=np.load(RUN/f'{aid}-test-predictions.npy');fpr,tpr,_=roc_curve(test.rating>=3.5,probability)
        ax.plot(fpr,tpr,color=COLORS[j],label=f'{LABEL[aid]}: AUC={get(aid,"test")["auc_roc"]:.3f}')
    ax.plot([0,1],[0,1],ls='--',color='gray',label='Không phân biệt');ax.set(xlabel='False Positive Rate',ylabel='True Positive Rate',xlim=(0,1),ylim=(0,1));ax.legend();save(fig,'classification_roc')
    fig,axes=plt.subplots(1,2,figsize=(6.2,3.2))
    for ax,aid in zip(axes,ids):
        cm=np.asarray(get(aid,'test')['confusion_matrix']);ax.imshow(cm,cmap='Blues')
        for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black',fontsize=13)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Không thích','Thích'],yticklabels=['Không thích','Thích'],xlabel='Dự đoán',ylabel='Thực tế',title=LABEL[aid])
    fig.tight_layout();save(fig,'classification_confusion')


def unsupervised_figures():
    cluster=DATA['clustering'];fig,axes=plt.subplots(1,2,figsize=(6.2,3.5))
    for ax,aid in zip(axes,['kmeans','dbscan']):
        c=next(c for c in cluster['charts'] if c['title'].lower().startswith(aid))
        for i,s in enumerate(c['series']):
            ax.scatter([p['x'] for p in s['points']],[p['y'] for p in s['points']],s=10,label=s['label'],color='#aaa' if s['label']=='Noise' else COLORS[i%len(COLORS)],alpha=.7)
        ax.set(title=LABEL[aid],xlabel='PC1',ylabel='PC2');ax.legend(fontsize=8,loc='best')
    fig.tight_layout();save(fig,'clustering_pca')
    rows=[r for r in cluster['rows'] if r['split']=='test'];fig,axes=plt.subplots(1,3,figsize=(6.2,3.2))
    for ax,key,title in [(axes[0],'silhouette','Silhouette ↑'),(axes[1],'dbi','DBI ↓'),(axes[2],'noise_fraction','Noise ↓')]:
        vals=[r[key] if key!='noise_fraction' else 100*r[key] for r in rows]
        ax.bar([LABEL[r['algorithm']] for r in rows],vals,color=COLORS[:2]);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[2].set_ylabel('% user Test');fig.tight_layout();save(fig,'clustering_quality_noise')
    search=cluster['search'];fig,axes=plt.subplots(1,2,figsize=(6.2,3.3));km=[r for r in search if r['algorithm']=='kmeans']
    axes[0].plot([r['config']['n_clusters'] for r in km],[r['silhouette'] for r in km],marker='o',color=COLORS[0]);axes[0].set(xlabel='K',ylabel='Silhouette Train ↑',title='K-Means')
    for m in sorted({r['config']['min_samples'] for r in search if r['algorithm']=='dbscan'}):
        ds=[r for r in search if r['algorithm']=='dbscan' and r['config']['min_samples']==m and r['silhouette'] is not None]
        axes[1].plot([r['config']['eps'] for r in ds],[r['silhouette'] for r in ds],marker='o',label=f'min_samples={m}')
    axes[1].set(xlabel='eps',ylabel='Silhouette Train ↑',title='DBSCAN');axes[1].legend(fontsize=8);fig.tight_layout();save(fig,'clustering_search')
    rules=DATA['association']['rules'];best=rules[:5];fig,ax=plt.subplots(figsize=(6.2,3.3))
    ax.barh(np.arange(len(best)),[r['lift'] for r in best],color=COLORS[0]);ax.set(yticks=np.arange(len(best)),yticklabels=[f'Luật {i+1}' for i in range(len(best))],xlabel='Lift ↑');ax.invert_yaxis();ax.axvline(1,ls='--',color='gray');save(fig,'association_lift')
    fig,ax=plt.subplots(figsize=(6.2,3.5));s=ax.scatter([r['support'] for r in rules],[r['confidence'] for r in rules],c=[r['lift'] for r in rules],s=14,cmap='viridis',alpha=.65)
    ax.set(xlabel='Support',ylabel='Confidence');fig.colorbar(s,ax=ax,label='Lift');ax.axvline(.08,color='gray',ls='--');ax.axhline(.3,color='gray',ls='--');save(fig,'association_support_confidence')
    red=DATA['pca'];fig,ax=plt.subplots(figsize=(6.2,3.4))
    for i,aid in enumerate(['pca','truncated_svd']):
        rr=[r for r in red['rows'] if r['split']=='test' and r['algorithm'].startswith(aid+'_')]
        ax.plot([r['components'] for r in rr],[r['explained_variance_ratio'] for r in rr],marker=['o','x'][i],label=aid,color=COLORS[i])
    ax.axhline(.9,ls='--',color='gray',label='Mốc 90%');ax.set(xlabel='Số components',ylabel='Phương sai giữ lại Train ↑',ylim=(0,1.05));ax.legend();save(fig,'reduction_variance')
    fig,ax=plt.subplots(figsize=(6.2,3.5))
    for i,aid in enumerate(['pca','truncated_svd']):
        for split,style in [('validation','-'),('test','--')]:
            rr=[r for r in red['rows'] if r['split']==split and r['algorithm'].startswith(aid+'_')]
            ax.plot([r['components'] for r in rr],[r['reconstruction_error'] for r in rr],style,marker=['o','x'][i],color=COLORS[i],label=f'{aid} / {split}')
    ax.axhline(.1,ls=':',color='gray');ax.set(xlabel='Số components',ylabel='MSE tái tạo ↓');ax.legend(fontsize=8);save(fig,'reduction_reconstruction')
    fig,ax=plt.subplots(figsize=(6.2,3.4));aids=['pca','truncated_svd','mean','item_knn'];x=np.arange(4)
    for j,split in enumerate(['validation','test']):
        ax.bar(x+(j-.5)*.36,[next(r['rmse'] for r in red['downstream'] if r['algorithm']==a and r['split']==split) for a in aids],.36,color=COLORS[j],label=split.title())
    ax.set(xticks=x,xticklabels=['PCA 12','SVD 12','Mean','Item-KNN'],ylabel='RMSE rating ↓');ax.legend();save(fig,'reduction_downstream')


def rl_figures():
    rl=DATA['reinforcement'];fig,ax=plt.subplots(figsize=(6.2,3.5))
    ax.plot([r['episode'] for r in rl['curve']],[r['average_reward'] for r in rl['curve']],marker='o',ms=3,color=COLORS[0],label='Q-Learning / Validation')
    for i,aid in enumerate(['random','popular']):
        base=next(r['average_reward'] for r in rl['rows'] if r['algorithm']==aid and r['split']=='validation');ax.axhline(base,color=COLORS[i+1],ls='--',label=LABEL[aid])
    ax.set(xlabel='Episode huấn luyện',ylabel='Reward / action trên Validation ↑');ax.legend();ax.grid(alpha=.2);save(fig,'rl_learning_curve')
    rows=[r for r in rl['rows'] if r['split']=='test'];fig,axes=plt.subplots(1,2,figsize=(6.2,3.4))
    for ax,key,title in [(axes[0],'cumulative_reward','Cumulative Reward Test ↑'),(axes[1],'observed_feedback_fraction','Phản hồi đã quan sát Test')]:
        ax.bar([LABEL[r['algorithm']] for r in rows],[r[key] for r in rows],color=COLORS[:3]);ax.set_title(title);ax.tick_params(axis='x',rotation=15);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();save(fig,'rl_test_reward_coverage')


if __name__=='__main__':
    diagram_figures();dataset_figures();supervised_figures();unsupervised_figures();rl_figures()
    (ASSETS/'provenance.json').write_text(json.dumps({'run_id':DATA['run_id'],'source':str(RUN/'results.json'),'figure_count':len(list(ASSETS.glob('*.png')))},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Created report figures:',len(list(ASSETS.glob('*.png'))),ASSETS)
