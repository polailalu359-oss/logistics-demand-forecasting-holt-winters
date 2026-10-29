"""
WEEK 5: Forecasting and Trend Analysis in Demand Planning
Run: pip install numpy pandas matplotlib  then  python week5_code.py
PROCESS
1. Simulate 48 months of shipment volume (seed 7): trend + seasonality + promotion - fuel effect + noise + one anomaly (Jul 2024).
2. Decompose: trend (12-month centred moving average), seasonal index (median of actual/trend), residual z-score for anomalies.
3. Models: moving average, simple exponential smoothing, seasonal naive, Holt-Winters (grid search alpha, beta, gamma).
4. Hold out last 6 months; compare MAE, RMSE, MAPE.
5. Re-fit best model on all data; forecast 12 months of 2026 with approx 80% interval (+/- 1.28 x RMSE).
OUTPUT: output/week5_demand_dataset.csv, output/results_week5.json, output/img/w5_*.png
"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
os.makedirs('output/img',exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
NAVY='#1F3A5F'; TEAL='#2A9D8F'; ORG='#E76F51'; GOLD='#E9C46A'; GREY='#8D99AE'
R={}
# ================= WEEK 5 =================
rng=np.random.default_rng(7)
idx=pd.date_range('2022-01-01',periods=48,freq='MS')
t=np.arange(48)
seas_f=np.array([-0.10,-0.12,-0.05,0.0,0.03,-0.02,-0.04,0.0,0.05,0.12,0.20,0.17])
fuel=100+np.cumsum(rng.normal(0,2,48)); promo=np.zeros(48); promo[[10,11,22,23,34,35,46,47]]=1
base=12000+95*t
dem=base*(1+seas_f[idx.month-1]) + promo*900 - (fuel-fuel.mean())*8 + rng.normal(0,350,48)
dem[30]*=0.80  # anomaly
dem=np.round(dem).astype(int)
d5=pd.DataFrame({'month':idx.strftime('%Y-%m'),'shipment_volume':dem,'fuel_price_index':np.round(fuel,1),'promotion_flag':promo.astype(int)})
d5.to_csv('output/week5_demand_dataset.csv',index=False)
y=pd.Series(dem,index=idx,dtype=float)
# classical decomposition
trend=y.rolling(12,center=True).mean().rolling(2).mean().shift(-1)
detr=y/trend
sidx=detr.groupby(detr.index.month).median(); sidx=sidx/sidx.mean()
resid=y/(trend*sidx.reindex(idx.month).values)
# anomaly detection via z-score of residual
z=(resid-resid.mean())/resid.std(); anomalies=[(i.strftime('%b %Y'),int(y[i]),round(float(z[i]),2)) for i in z.index if abs(z[i])>2.2]
R['w5']={'mean':float(y.mean()),'min':int(y.min()),'max':int(y.max()),'std':float(y.std()),
 'yoy':[float(y[str(yr)].sum()) for yr in [2022,2023,2024,2025]],
 'seasonal':{pd.Timestamp(2000,m,1).strftime('%b'):round(float(sidx[m]),3) for m in range(1,13)},
 'anomalies':anomalies,'corr_fuel':float(y.corr(d5.set_index(idx).fuel_price_index)),'promo_uplift':float(y[promo==1].mean()/y[promo==0].mean()-1)}
sl=np.polyfit(t,y.values,1); R['w5']['slope']=float(sl[0])
train,test=y[:42],y[42:]
def ma_fc(tr,h,k=3): return np.repeat(tr[-k:].mean(),h)
def ses(tr,h,a):
    l=tr[0]
    for v in tr[1:]: l=a*v+(1-a)*l
    return np.repeat(l,h)
def hw(tr,h,a,b,g,p=12):
    tr=np.asarray(tr,float); n=len(tr)
    l=tr[:p].mean(); bt=(tr[p:2*p].mean()-tr[:p].mean())/p; s=list(tr[:p]-l)
    for i in range(p,n):
        ls=l; l=a*(tr[i]-s[i-p])+(1-a)*(l+bt); bt=b*(l-ls)+(1-b)*bt; s.append(g*(tr[i]-l)+(1-g)*s[i-p])
    return np.array([l+(k+1)*bt+s[n-p+(k%p)] for k in range(h)]),(l,bt,s)
def sse_hw(par,tr):
    a,b,g=par; n=len(tr); p=12; tr=np.asarray(tr,float)
    l=tr[:p].mean(); bt=(tr[p:2*p].mean()-tr[:p].mean())/p; s=list(tr[:p]-l); e=0
    for i in range(p,n):
        f=l+bt+s[i-p]; e+=(tr[i]-f)**2
        ls=l; l=a*(tr[i]-s[i-p])+(1-a)*(l+bt); bt=b*(l-ls)+(1-b)*bt; s.append(g*(tr[i]-l)+(1-g)*s[i-p])
    return e
best=None
for a in np.arange(.1,1,.1):
  for b in [.01,.05,.1,.2]:
    for g in np.arange(.1,1,.1):
      e=sse_hw((a,b,g),train.values)
      if best is None or e<best[0]: best=(e,a,b,g)
_,a,b,g=best
bs=min(np.arange(.1,1,.1),key=lambda aa: sum((train.values[i]-ses(train.values[:i],1,aa)[0])**2 for i in range(3,42)))
def met(f): 
    e=test.values-f; return dict(MAE=float(np.mean(np.abs(e))),RMSE=float(np.sqrt(np.mean(e**2))),MAPE=float(np.mean(np.abs(e)/test.values)*100))
# seasonal naive & MA
snaive=y.values[30:36]
# trend-adjusted seasonal naive (same month last year * growth)
fc_ma=ma_fc(train.values,6); fc_ses=ses(train.values,6,bs); fc_hw,_=hw(train.values,6,a,b,g)
R['w5']['models']={'Moving Avg (3)':met(fc_ma),'Simple Exp. Smoothing':met(fc_ses),'Seasonal Naive':met(snaive),'Holt-Winters (additive)':met(fc_hw)}
R['w5']['hw_params']=[round(a,2),round(b,2),round(g,2)]; R['w5']['ses_alpha']=round(float(bs),2)
# final forecast 12 months with HW on full data, with approximate 80% interval from holdout RMSE
fc12,_=hw(y.values,12,a,b,g)
rm=R['w5']['models']['Holt-Winters (additive)']['RMSE']
fidx=pd.date_range('2026-01-01',periods=12,freq='MS')
R['w5']['forecast']=[[d.strftime('%b %Y'),int(round(v))] for d,v in zip(fidx,fc12)]
R['w5']['fc_total']=float(fc12.sum()); R['w5']['growth_vs_2025']=float(fc12.sum()/y['2025'].sum()-1)
R['w5']['peak']=R['w5']['forecast'][int(np.argmax(fc12))]; R['w5']['trough']=R['w5']['forecast'][int(np.argmin(fc12))]
fig,ax=plt.subplots(figsize=(7.5,3.4)); ax.plot(y.index,y.values,color=NAVY); ax.plot(y.index,np.polyval(sl,t),'--',color=ORG,label='Linear trend'); ax.set_title('Monthly Shipment Volume 2022-2025'); ax.set_ylabel('Shipments'); ax.legend()
for lab,v,_z in anomalies: pass
plt.tight_layout(); plt.savefig('output/img/w5_series.png'); plt.close()
fig,axs=plt.subplots(3,1,figsize=(7.5,6),sharex=True); axs[0].plot(y.index,y.values,color=NAVY); axs[0].set_ylabel('Observed'); axs[1].plot(trend.index,trend.values,color=ORG); axs[1].set_ylabel('Trend'); axs[2].plot(y.index,(sidx.reindex(idx.month).values),color=TEAL); axs[2].set_ylabel('Seasonal index'); fig.suptitle('Classical Multiplicative Decomposition',fontsize=10); plt.tight_layout(); plt.savefig('output/img/w5_decomp.png'); plt.close()
fig,ax=plt.subplots(figsize=(7,3.2)); ax.bar(list(R['w5']['seasonal'].keys()),[v for v in R['w5']['seasonal'].values()],color=[ORG if v>1.05 else GREY for v in R['w5']['seasonal'].values()]); ax.axhline(1,color='k',lw=.8); ax.set_title('Seasonal Index by Month (1.00 = average)'); plt.tight_layout(); plt.savefig('output/img/w5_season.png'); plt.close()
fig,ax=plt.subplots(figsize=(7.5,3.4)); ax.plot(train.index,train.values,color=NAVY,label='Train'); ax.plot(test.index,test.values,color='k',marker='o',label='Actual (test)')
ax.plot(test.index,fc_ma,'--',label='Moving avg'); ax.plot(test.index,fc_ses,'--',label='SES'); ax.plot(test.index,fc_hw,color=ORG,marker='s',label='Holt-Winters'); ax.set_xlim(pd.Timestamp('2024-06-01'),pd.Timestamp('2025-12-31')); ax.legend(fontsize=7); ax.set_title('Model Comparison on Hold-out (Jul-Dec 2025)'); plt.tight_layout(); plt.savefig('output/img/w5_models.png'); plt.close()
fig,ax=plt.subplots(figsize=(7.5,3.4)); ax.plot(y.index,y.values,color=NAVY,label='History'); ax.plot(fidx,fc12,color=ORG,marker='o',label='Forecast 2026'); ax.fill_between(fidx,fc12-1.28*rm,fc12+1.28*rm,color=ORG,alpha=.2,label='~80% interval'); ax.legend(); ax.set_title('12-Month Demand Forecast (Holt-Winters)'); ax.set_ylabel('Shipments'); plt.tight_layout(); plt.savefig('output/img/w5_forecast.png'); plt.close()


json.dump(R,open('output/results_week5.json','w'),indent=1,default=float)
print('Done. Check the output folder.')