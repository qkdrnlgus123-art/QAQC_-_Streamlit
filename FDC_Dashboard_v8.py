import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score
import os

st.set_page_config(page_title="FDC Dashboard", page_icon="⚙️",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.stApp { background:#F4F6F9; }
.main .block-container { padding-top:0.8rem; padding-bottom:2rem; max-width:100%; }
.top-bar { background:#1C2B3A; padding:13px 22px; border-radius:10px;
    display:flex; align-items:center; justify-content:space-between; margin-bottom:14px; }
.top-bar-title { font-size:1.1rem; font-weight:600; color:#fff; }
.top-bar-sub   { font-size:.78rem; color:rgba(255,255,255,.55); margin-top:3px; }
.live-badge { background:#253B4D; color:#7CC9DA; border:1px solid #336B80;
    padding:2px 10px; border-radius:3px; font-size:.75rem; font-weight:600; }
.stTabs [data-baseweb="tab-list"] { background:#fff; border-bottom:1px solid #D8E0E8; padding:0 4px; gap:0; }
.stTabs [data-baseweb="tab"] { font-size:.85rem; font-weight:500; color:#6B7C8D;
    padding:10px 16px; border-bottom:2px solid transparent; }
.stTabs [aria-selected="true"] { color:#1C2B3A !important;
    border-bottom:2px solid #2E4057 !important; background:transparent !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top:1rem; }
.sec-lbl { font-size:.75rem; font-weight:600; color:#6B7C8D; text-transform:uppercase;
    letter-spacing:.5px; padding-bottom:5px; border-bottom:1px solid #E0E7EE; margin-bottom:.6rem; }
.kpi-card { background:#fff; border:1px solid #D8E0E8; border-radius:8px;
    padding:14px 18px; height:100%; border-top:3px solid; }
.kpi-lbl { font-size:.75rem; color:#6B7C8D; text-transform:uppercase;
    letter-spacing:.4px; margin-bottom:7px; font-weight:500; }
.kpi-val { font-size:2.1rem; font-weight:500; line-height:1; color:#1C2B3A; }
.kpi-sub { font-size:.75rem; color:#9BA8B2; margin-top:5px; }
.kpi-val.burg  { color:#7D1C2F; }
.kpi-val.navy  { color:#2E4057; }
.kpi-val.slate { color:#3D5166; }
.alarm-banner { background:#F5E6E9; border-left:4px solid #7D1C2F; border-radius:0 7px 7px 0;
    padding:9px 14px; display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
.alarm-text { font-size:.85rem; color:#5A1020; }
.alarm-pill { background:#7D1C2F; color:#fff; padding:2px 10px; border-radius:3px;
    font-size:.75rem; font-weight:600; flex-shrink:0; }
.content-card { background:#fff; border:1px solid #D8E0E8; border-radius:8px;
    padding:13px 15px; margin-bottom:10px; }
.card-title { font-size:.85rem; font-weight:500; color:#1C2B3A;
    display:flex; align-items:center; justify-content:space-between; margin-bottom:9px; }
.card-badge { background:#EEF1F5; color:#4A6274; padding:2px 8px; border-radius:10px;
    font-size:.75rem; font-weight:500; }
.action-card { padding:9px 12px; margin-bottom:7px; }
.action-lv   { font-size:.72rem; font-weight:600; text-transform:uppercase; letter-spacing:.3px; margin-bottom:2px; }
.action-title { font-size:.85rem; font-weight:500; color:#1C2B3A; }
.action-desc  { font-size:.78rem; color:#6B7C8D; margin-top:2px; }
.wafer-table { width:100%; border-collapse:collapse; font-size:.82rem; }
.wafer-table th { padding:7px 9px; color:#6B7C8D; font-weight:500; font-size:.72rem;
    text-transform:uppercase; letter-spacing:.3px; border-bottom:1px solid #D8E0E8;
    background:#F8FAFC; text-align:left; }
.wafer-table td { padding:7px 9px; border-bottom:1px solid #EEF1F5; vertical-align:middle; }
.risk-badge { padding:2px 7px; border-radius:3px; font-size:.72rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

EV_NAMES = ['BCl3 Flow','Cl2 Flow','RF Btm Pwr','RF Btm Rfl Pwr','Endpt A','He Press',
            'Pressure','RF Tuner','RF Load','RF Phase Err','RF Pwr','RF Impedance',
            'TCP Tuner','TCP Phase Err','TCP Impedance','TCP Top Pwr','TCP Rfl Pwr',
            'TCP Load','Vat Valve']
FAULT_GROUP = {
    'TCP +10':'TCP 관련','TCP +20':'TCP 관련','TCP +30':'TCP 관련',
    'TCP +50':'TCP 관련','TCP -15':'TCP 관련','TCP -20':'TCP 관련',
    'RF +8':'RF 관련','RF +10':'RF 관련','RF -12':'RF 관련',
    'Pr +1':'압력 관련','Pr +2':'압력 관련','Pr +3':'압력 관련','Pr -2':'압력 관련',
    'BCl3 +5':'가스 관련','BCl3 -5':'가스 관련','BCl3 +10':'가스 관련',
    'Cl2 +5':'가스 관련','Cl2 -5':'가스 관련','Cl2 -10':'가스 관련',
    'He Chuck':'He Chuck','calibration':'정상',
}
FAULT_KEY_VAR = {
    'TCP 관련':'RF Pwr', 'RF 관련':'RF Pwr',
    '압력 관련':'Vat Valve', '가스 관련':'Vat Valve', 'He Chuck':'Vat Valve',
    '정상':'Pressure',
}
GROUP_COLOR = {
    '압력 관련':'#7D1C2F','TCP 관련':'#2E4057','RF 관련':'#3D5166',
    '가스 관련':'#4A6274','He Chuck':'#6B7C8D','정상':'#A0ADB8',
}
RISK_COLOR = {'위험':'#7D1C2F','주의':'#3D5166','확인':'#4A6274','정상':'#A0ADB8'}
RISK_BG    = {'위험':'#FDF1F3','주의':'#E8EDF2','확인':'#EEF1F5','정상':'#F4F6F9'}
ACTION_MAP = {
    '압력 관련':[('긴급','#7D1C2F','#F5E6E9','Pressure Sensor Calibration','챔버 압력 변동 — Vat Valve 개도 이상'),
                ('우선','#2E4057','#EAF0F6','Vat Valve 개도 확인','이상 개도로 인한 압력 변동'),
                ('확인','#4A6274','#F0F4F8','Gas Flow 계통 점검','BCl3/Cl2 유량 변동 이상 여부')],
    'TCP 관련':[('긴급','#7D1C2F','#F5E6E9','TCP Generator 점검','TCP 파워 이탈 — RF Pwr/Load 동반'),
               ('우선','#2E4057','#EAF0F6','TCP Tuner 매칭 확인','TCP Tuner 임피던스 이상'),
               ('확인','#4A6274','#F0F4F8','플라즈마 상태 확인','TCP 전력 이상 시 플라즈마 불안정')],
    'RF 관련':[('긴급','#7D1C2F','#F5E6E9','RF 매칭 네트워크 점검','RF Pwr/Load 이탈 — 임피던스 확인'),
              ('우선','#2E4057','#EAF0F6','RF Generator 설정값 확인','RF 출력 이상 — 설정값 재확인'),
              ('확인','#4A6274','#F0F4F8','Pressure Sensor Calibration','RF 이상이 압력 변동 유발 가능')],
    '가스 관련':[('긴급','#7D1C2F','#F5E6E9','Gas Flow 계통 점검','BCl3/Cl2 유량 이탈 — MFC 교정'),
               ('우선','#2E4057','#EAF0F6','가스 공급 배관 누설 확인','유량 이상 시 배관 누설 점검'),
               ('확인','#4A6274','#F0F4F8','Pressure Sensor Calibration','가스 유량 변동이 압력 변동으로 연결')],
    'He Chuck':[('긴급','#7D1C2F','#F5E6E9','He Chuck 압력 확인','웨이퍼 냉각 헬륨 척 압력 이상'),
               ('우선','#2E4057','#EAF0F6','웨이퍼 척킹 상태 확인','He Chuck 이상은 웨이퍼 온도 불균일'),
               ('확인','#4A6274','#F0F4F8','냉각 계통 점검','He 공급 라인 및 냉각 시스템 확인')],
    '정상':[],
}
BASE_LAYOUT = dict(font=dict(family='Arial, sans-serif', size=13),
                   plot_bgcolor='white', paper_bgcolor='white',
                   margin=dict(l=55, r=25, t=38, b=40))

@st.cache_data(show_spinner='모델 로딩 중...')
    # 수정
def load_and_run():
    # 수정
    url = 'https://drive.google.com/uc?id=1UN6Z09TDZBmubJZYeoVpaf_D1CRRYndm&export=download'
    df = pd.read_csv(url, index_col=0)
    df['fault_name'] = df['fault_name'].str.strip()
    META = ['wafer_id','t_norm','fault','fault_name','Step Number']
    rfm_cols = [c for c in df.columns if c.startswith('S') and c not in META]
    ev_cols  = [c for c in EV_NAMES if c in df.columns]
    OES_SELECTED = ["395.8","395.8.1","395.8.2",
                   "336.98","336.98.1","336.98.2",
                   "250.0","250.0.1","250.0.2",
                   "725.0","725.0.1","725.0.2",
                   "272.2","272.2.1","272.2.2",
                   "532.6","532.6.1","532.6.2"]
    oes_cols = [c for c in OES_SELECTED if c in df.columns]
    N = 100
    n_vars = len(rfm_cols) + len(oes_cols) + len(ev_cols)
    all_feat = rfm_cols + oes_cols + ev_cols
    wids = sorted(df['wafer_id'].unique())
    X_list,fl,fnl,el,wa = [],[],[],[],[]
    for wid in wids:
        g = df[df['wafer_id']==wid].sort_values('t_norm')
        X_list.append(g[all_feat].values.flatten())
        fl.append(g['fault'].iloc[0])
        fnl.append(g['fault_name'].iloc[0])
        el.append(str(wid)[:2])
        wa.append(str(int(wid)))
    X_fusion = np.array(X_list)
    fault_labels = np.array(fl)
    fault_name_lbl = np.array(fnl)
    exp_labels = np.array(el)
    wafer_arr = np.array(wa)
    normal_mask = fault_labels == 'normal'
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fusion)
    pca_models, T2_lim, SPE_lim = {}, {}, {}
    for exp in np.unique(exp_labels):
        mn = (exp_labels==exp) & normal_mask
        if mn.sum() < 7: continue
        nc = min(7, mn.sum()-1)
        pe = PCA(n_components=nc)
        Tn = pe.fit_transform(X_scaled[mn])
        T2n  = np.sum((Tn/pe.explained_variance_)**2, axis=1)
        SPEn = np.sum((X_scaled[mn]-pe.inverse_transform(Tn))**2, axis=1)
        T2_lim[exp]  = float(np.percentile(T2n,  98))
        SPE_lim[exp] = float(np.percentile(SPEn, 98))
        pca_models[exp] = pe
    scores_all, resids_all, alarms, T2s, SPEs = [], [], [], [], []
    for i in range(len(X_scaled)):
        x = X_scaled[i]; be, bs = None, np.inf
        for exp, pm in pca_models.items():
            t = pm.transform(x.reshape(1,-1))
            spe = float(np.sum((x - pm.inverse_transform(t))**2))
            if spe < bs: bs, be = spe, exp
        pm = pca_models[be]
        t  = pm.transform(x.reshape(1,-1)).flatten()
        res = x - pm.inverse_transform(t.reshape(1,-1)).flatten()
        t2  = float(np.sum((t/pm.explained_variance_)**2))
        spe = float(np.sum(res**2))
        scores_all.append(t); resids_all.append(res)
        T2s.append(t2); SPEs.append(spe)
        alarms.append((t2>T2_lim[be]) | (spe>SPE_lim[be]))
    scores_all = np.array(scores_all)
    resids_all = np.array(resids_all)
    T2s = np.array(T2s); SPEs = np.array(SPEs)
    alarms = np.array(alarms)
    def _spe_ts(res):
        return np.array([
            sum(res[v*N + t]**2 for v in range(n_vars))
            for t in range(N)
        ])
    normal_spe_ts = np.array([_spe_ts(resids_all[i]) for i in np.where(normal_mask)[0]])
    spe_ts_q95  = np.percentile(normal_spe_ts, 95, axis=0)
    spe_ts_mean = normal_spe_ts.mean(axis=0)
    spe_ts_q95_safe = np.where(spe_ts_q95 > 0, spe_ts_q95, 1.0)
    spe_ts_all      = np.array([_spe_ts(resids_all[i])/spe_ts_q95_safe for i in range(len(X_scaled))])
    spe_ts_q95_norm = np.ones(N)
    spe_ts_mean_norm= spe_ts_mean / spe_ts_q95_safe
    fm = fault_labels != 'normal'
    fsc = scores_all[fm]; ffn = fault_name_lbl[fm]
    classified = {}
    for idx in np.where(alarms)[0]:
        d = cdist(scores_all[idx].reshape(1,-1), fsc, metric='euclidean').flatten()
        classified[idx] = ffn[np.argmin(d)]
    cal = df[df['fault_name']=='calibration'].copy()
    cal['t_bin'] = (cal['t_norm']*99).round().astype(int)
    cal_bands = {}
    for col in ev_cols:
        grp = cal.groupby('t_bin')[col].agg(['mean','std']).reset_index()
        grp['t_norm'] = grp['t_bin'] / 99
        grp['upper']  = grp['mean'] + 3*grp['std']
        grp['lower']  = grp['mean'] - 3*grp['std']
        cal_bands[col] = grp
    wafer_ts = {}
    wafer_time = {}   # wafer_id → Time 배열 (100포인트)
    for wid in wids:
        g = df[df['wafer_id']==wid].sort_values('t_norm')
        wafer_ts[str(int(wid))] = g[['t_norm']+ev_cols].copy()
        if 'Time' in g.columns:
            wafer_time[str(int(wid))] = g['Time'].values
    alarm_rows = []
    for idx in np.where(alarms)[0]:
        fn  = classified.get(idx, fault_name_lbl[idx])
        grp = FAULT_GROUP.get(fn, '기타')
        exp = exp_labels[idx]
        lim = SPE_lim.get(exp, float(np.mean(list(SPE_lim.values()))))
        ratio = SPEs[idx] / lim
        status = '위험' if ratio>2.0 else ('주의' if ratio>1.2 else '확인')
        is_fp  = bool(fault_labels[idx] == 'normal')
        alarm_rows.append({'idx':int(idx), 'wafer':wafer_arr[idx],
                           'Fault 유형':fn, '그룹':grp,
                           'SPE':round(float(SPEs[idx]),1),
                           '등급':status, 'is_fp':is_fp, 'exp':exp})
    alarm_rows.sort(key=lambda r: r['SPE'], reverse=True)
    return dict(
        wafer_arr=wafer_arr, fault_labels=fault_labels,
        fault_name_lbl=fault_name_lbl, exp_labels=exp_labels,
        normal_mask=normal_mask, alarms=alarms,
        T2s=T2s, SPEs=SPEs,
        scores_all=scores_all, resids_all=resids_all,
        classified=classified,
        cal_bands=cal_bands, wafer_ts=wafer_ts, wafer_time=wafer_time, ev_cols=ev_cols,
        spe_ts_all=spe_ts_all,
        spe_ts_q95_norm=spe_ts_q95_norm,
        spe_ts_mean_norm=spe_ts_mean_norm,
        N=N, n_vars=n_vars,
        pca_models=pca_models, T2_lim=T2_lim, SPE_lim=SPE_lim,
        alarm_rows=alarm_rows, wids=wids,
        rfm_cols=rfm_cols, oes_cols=oes_cols, all_feat=all_feat,
    )

R = load_and_run()
wafer_arr       = R['wafer_arr']
fault_labels    = R['fault_labels']
fault_name_lbl  = R['fault_name_lbl']
exp_labels      = R['exp_labels']
normal_mask     = R['normal_mask']
alarms          = R['alarms']
SPEs            = R['SPEs']
classified      = R['classified']
cal_bands       = R['cal_bands']
wafer_ts        = R['wafer_ts']
ev_cols         = R['ev_cols']
spe_ts_all      = R['spe_ts_all']
spe_ts_q95_norm = R['spe_ts_q95_norm']
spe_ts_mean_norm= R['spe_ts_mean_norm']
N               = R['N']
SPE_lim         = R['SPE_lim']
alarm_rows      = R['alarm_rows']
wafer_time      = R['wafer_time']

# ── 신호 기여도 계산 함수 ──────────────────────────────────────
def get_step_contribution(wafer_idx, step_idx, top_n=10):
    """특정 웨이퍼의 특정 t_norm 시점에서 센서별 SPE 기여도"""
    _resids  = R['resids_all']
    _rfm     = R['rfm_cols']
    _oes     = R['oes_cols']
    _ev      = R['ev_cols']
    _n_vars  = R['n_vars']
    _N       = R['N']
    all_feat = R['all_feat']

    res = _resids[wafer_idx]  # (n_vars * N,)
    contrib = np.array([res[v * _N + step_idx] ** 2 for v in range(_n_vars)])

    groups = {
        'RFM': float(contrib[:len(_rfm)].sum()),
        'OES': float(contrib[len(_rfm):len(_rfm)+len(_oes)].sum()),
        'EV' : float(contrib[len(_rfm)+len(_oes):].sum()),
    }
    top_idx  = np.argsort(contrib)[-top_n:][::-1]
    top_feat = [all_feat[i] for i in top_idx]
    top_val  = contrib[top_idx].tolist()
    return groups, top_feat, top_val

def render_actions(fault_group):
    for lv,color,bg,title,desc in ACTION_MAP.get(fault_group, []):
        st.markdown(f'''
        <div class="action-card" style="background:{bg};border-left:3px solid {color}">
          <div class="action-lv" style="color:{color}">{lv}</div>
          <div class="action-title">{title}</div>
          <div class="action-desc">{desc}</div>
        </div>''', unsafe_allow_html=True)

def make_cal_band_fig(wid, var, height=260):
    band = cal_bands.get(var)
    ts   = wafer_ts.get(wid)
    fig  = go.Figure()
    if band is None or ts is None or var not in ts.columns:
        return fig
    fig.add_trace(go.Scatter(
        x=list(band['t_norm'])+list(band['t_norm'][::-1]),
        y=list(band['upper'])+list(band['lower'][::-1]),
        fill='toself', fillcolor='rgba(74,143,168,0.10)',
        line=dict(color='rgba(0,0,0,0)'), name='정상 범위 (±3σ)'))
    fig.add_trace(go.Scatter(
        x=band['t_norm'], y=band['mean'], mode='lines',
        name='정상 평균', line=dict(color='#4A8FA8', width=1.5, dash='dash')))
    fig.add_trace(go.Scatter(
        x=ts['t_norm'].values, y=ts[var].values, mode='lines',
        name=f'{wid}.txm', line=dict(color='#7D1C2F', width=2)))
    fig.update_layout(**BASE_LAYOUT, height=height,
        xaxis_title='정규화 시간 (t_norm)', yaxis_title=var,
        legend=dict(orientation='h', y=1.12, x=0, font_size=12))
    return fig

def make_spe_ts_fig(idx, fn, height=280, use_time=False):
    # use_time=True → 실제 Time(초) 축 (Tab1), False → t_norm 정규화 축 (Tab3)
    if use_time:
        wid_key = wafer_arr[idx]
        t_vals_real = wafer_time.get(wid_key, None)
        if t_vals_real is not None and len(t_vals_real) == N:
            t_x = t_vals_real
            x_label = '공정 시간 (초)'
        else:
            t_x = np.linspace(0, 1, N)
            x_label = '정규화 시간 (t_norm)'
    else:
        t_x = np.linspace(0, 1, N)
        x_label = '공정 진행률'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(t_x)+list(t_x[::-1]),
        y=list(spe_ts_q95_norm)+list(spe_ts_mean_norm[::-1]),
        fill='toself', fillcolor='rgba(74,143,168,0.12)',
        line=dict(color='rgba(0,0,0,0)'), name='정상 SPE 범위'))
    fig.add_trace(go.Scatter(
        x=t_x, y=spe_ts_mean_norm, mode='lines',
        name='정상 평균', line=dict(color='#4A8FA8', width=1.5, dash='dash')))
    fig.add_hline(y=1.0, line_dash='dot', line_color='#7D1C2F',
                  line_width=1.5, opacity=0.8,
                  annotation_text='정상 한계 (q95)',
                  annotation_position='bottom right',
                  annotation_font_size=12,
                  annotation_font_color='#7D1C2F')
    fig.add_trace(go.Scatter(
        x=t_x, y=spe_ts_all[idx], mode='lines',
        name=f'{wafer_arr[idx]}.txm ({fn})',
        line=dict(color='#7D1C2F', width=2.5)))
    exceed = spe_ts_all[idx] > 1.0
    if exceed.any():
        fig.add_trace(go.Scatter(
            x=t_x[exceed], y=spe_ts_all[idx][exceed],
            mode='markers', name='이탈 시점',
            marker=dict(color='#7D1C2F', size=7, symbol='circle')))
    fig.update_layout(**{k:v for k,v in BASE_LAYOUT.items() if k != 'margin'},
        height=height,
        xaxis_title=x_label,
        yaxis_title='SPE / 정상 q95 (1.0 = 정상 한계)',
        margin=dict(l=80, r=25, t=38, b=40),
        legend=dict(orientation='h', y=1.12, x=0, font_size=12))
    return fig

n_total    = len(wafer_arr)
n_fault    = int(alarms.sum())
n_danger   = sum(1 for r in alarm_rows if r['등급']=='위험' and not r['is_fp'])
fault_rate = n_fault / n_total * 100
top_alarm  = max((r for r in alarm_rows if not r['is_fp']), key=lambda r: r['SPE'], default=alarm_rows[0])

st.markdown(f'''
<div class="top-bar">
  <div>
    <div class="top-bar-title">⚙️ FDC Dashboard — 금속 에칭 공정 이상탐지</div>
    <div class="top-bar-sub">all_Time_data.csv &nbsp;·&nbsp;
      RFM {len(R['rfm_cols'])}채널 + EV {len(ev_cols)}채널 + OES {len(R['oes_cols'])}채널
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:12px">
    <span style="font-size:.72rem;color:rgba(255,255,255,.45)">ev_data · oes_data · rfm_data</span>
    <span class="live-badge">LIVE</span>
  </div>
</div>''', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(['📋  공정 현황', '⚠️  이상 웨이퍼 목록', '📋  웨이퍼 처리 이력'])

with tab1:
    st.markdown(f'''
    <div class="alarm-banner">
      <span class="alarm-text">
        ⚠ <strong>이상 감지:</strong>&nbsp;
        {top_alarm['wafer']} ({top_alarm['Fault 유형']}) —
        SPE {top_alarm['SPE']:.0f} 임계값 초과 · 즉시 점검 필요
      </span>
      <span class="alarm-pill">위험</span>
    </div>''', unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">공정 현황 KPI</div>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    for cw,lbl,val,sub,color,border in [
        (k1,'전체 웨이퍼',   str(n_total),      '처리 완료',      'navy', '#2E4057'),
        (k2,'이상 웨이퍼',   str(n_fault),       '이상 발생 건수', 'burg', '#7D1C2F'),
        (k3,'즉시 점검 필요',str(n_danger),      '위험 등급',      'burg', '#7D1C2F'),
        (k4,'이상 발생률',   f'{fault_rate:.1f}%','전체 대비',     'slate','#3D5166'),
    ]:
        cw.markdown(f'''
        <div class="kpi-card" style="border-top-color:{border}">
          <div class="kpi-lbl">{lbl}</div>
          <div class="kpi-val {color}">{val}</div>
          <div class="kpi-sub">{sub}</div>
        </div>''', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    col_l, col_r = st.columns([1.6, 1])
    with col_l:
        st.markdown('<div class="sec-lbl">SPE 시점별 시계열 — 웨이퍼 단일 분석</div>',
                    unsafe_allow_html=True)
        fault_options = [(r['idx'], r['wafer'], r['Fault 유형'])
                         for r in alarm_rows if not r['is_fp']]
        sel_idx_t1 = st.selectbox(
            '웨이퍼 선택',
            options=[x[0] for x in fault_options],
            format_func=lambda i: next(
                f"{x[1]}  ·  {x[2]}"
                for x in fault_options if x[0]==i),
            key='spe_ts_sel')
        sel_fn_t1 = next(x[2] for x in fault_options if x[0]==sel_idx_t1)
        sel_grp_t1 = FAULT_GROUP.get(sel_fn_t1, '기타')
        col_spe = GROUP_COLOR.get(sel_grp_t1, '#7D1C2F')
        exceed_mask = spe_ts_all[sel_idx_t1] > 1.0
        n_exceed = int(exceed_mask.sum())
        # 최초 이탈 시점: Time(초) 기준
        _t_x_t1 = wafer_time.get(wafer_arr[sel_idx_t1], np.linspace(0,1,N))
        t_first  = float(_t_x_t1[exceed_mask][0]) if n_exceed>0 else None
        _x_unit  = '초' if wafer_arr[sel_idx_t1] in wafer_time else ''
        if n_exceed > 0:
            st.markdown(
                f"<div style='background:#FDF1F3;border-left:3px solid {col_spe};"
                f"padding:6px 12px;border-radius:0 4px 4px 0;font-size:.82rem;margin-bottom:6px;'>"
                f"<b style='color:{col_spe};'>{sel_fn_t1} 이탈 감지</b>&nbsp;·&nbsp;"
                f"정상 범위 초과 구간: <b>{n_exceed}</b>포인트 &nbsp;·&nbsp;"
                f"최초 이탈 시점: <b>t={t_first:.1f}{_x_unit}</b></div>",
                unsafe_allow_html=True)
        st.plotly_chart(make_spe_ts_fig(sel_idx_t1, sel_fn_t1, 280, use_time=True),
                        use_container_width=True)
    with col_r:
        st.markdown('<div class="sec-lbl">공정 상태 비율</div>', unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            labels=['정상','이상'], values=[n_total-n_fault, n_fault],
            hole=0.58, marker_colors=['#3D5166','#7D1C2F'], textinfo='none',
            hovertemplate='%{label}: %{value}개 (%{percent})<extra></extra>'))
        fig_donut.update_layout(
            font=dict(family='Arial, sans-serif', size=13),
            plot_bgcolor='white', paper_bgcolor='white',
            height=175, margin=dict(l=0,r=0,t=10,b=10),
            showlegend=True,
            legend=dict(orientation='v', x=1.0, y=0.5, font=dict(size=13)),
            annotations=[dict(
                text=f'<b>{100-fault_rate:.1f}%</b><br>정상',
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=13, color='#1C2B3A'))])
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('<div style="font-size:.75rem;color:#6B7C8D;font-weight:500;margin:4px 0 8px">Fault Group 분포</div>', unsafe_allow_html=True)
        gc = {}
        for r in alarm_rows:
            g = r['그룹']
            if g != '정상': gc[g] = gc.get(g,0)+1
        if gc:
            gdf = pd.DataFrame(list(gc.items()),columns=['grp','cnt']).sort_values('cnt',ascending=True)
            fig_fg = go.Figure(go.Bar(
                x=gdf['cnt'], y=gdf['grp'], orientation='h',
                marker_color=[GROUP_COLOR.get(g,'#6B7C8D') for g in gdf['grp']],
                text=[f'{v}종' for v in gdf['cnt']], textposition='outside'))
            fig_fg.update_layout(
                font=dict(family='Arial, sans-serif', size=13),
                plot_bgcolor='white', paper_bgcolor='white',
                height=210, margin=dict(l=10,r=50,t=5,b=20),
                xaxis=dict(range=[0, gdf['cnt'].max()*1.35], title=''),
                yaxis_title='')
            st.plotly_chart(fig_fg, use_container_width=True)

with tab2:
    col_tbl, col_act = st.columns([1.2, 1.0])
    with col_tbl:
        f1c,f2c,f3c = st.columns(3)
        with f1c: wf=st.selectbox('웨이퍼',['전체']+[r['wafer'] for r in alarm_rows],key='wf')
        with f2c: ff=st.selectbox('Fault 유형',['전체']+sorted({r['Fault 유형'] for r in alarm_rows}),key='ff')
        with f3c: sf=st.selectbox('등급',['전체','위험','주의','확인'],key='sf')
        disp = [r for r in alarm_rows
                if (wf=='전체' or r['wafer']==wf)
                and (ff=='전체' or r['Fault 유형']==ff)
                and (sf=='전체' or r['등급']==sf)]
        if not disp: disp = alarm_rows
        sel_w = st.selectbox('웨이퍼 선택', [r['wafer'] for r in disp],
                             key='sel_w2', label_visibility='collapsed')
        st.markdown('<div class="sec-lbl">⚠&nbsp; 이상 웨이퍼 목록</div>', unsafe_allow_html=True)
        st.markdown('<div class="content-card" style="padding:0">', unsafe_allow_html=True)
        tbl = '''<table class="wafer-table"><thead><tr>
          <th style="width:22px">#</th><th>웨이퍼</th>
          <th>Fault 유형</th>
          <th style="text-align:right">SPE</th>
          <th style="text-align:center">등급</th>
        </tr></thead><tbody>'''
        for i,r in enumerate(disp[:15]):
            rc  = RISK_COLOR.get(r['등급'], '#6B7C8D')
            rbg = RISK_BG.get(r['등급'], '#F4F6F9')
            rowbg = '#FDF1F3' if r['wafer']==sel_w else ('white' if i%2==0 else '#F8FAFC')
            fp_mark = ' (FP)' if r['is_fp'] else ''
            tbl += f'''
            <tr style="border-bottom:1px solid #EEF1F5;background:{rowbg}">
              <td style="color:#A0ADB8;font-size:.75rem">{i+1}</td>
              <td style="color:#2E4057;font-weight:500">{r['wafer']}{fp_mark}</td>
              <td style="color:{rc};font-size:.82rem">{r['Fault 유형']}</td>
              <td style="text-align:right;color:{rc};font-weight:500">{r['SPE']:.0f}</td>
              <td style="text-align:center">
                <span class="risk-badge" style="background:{rbg};color:{rc}">{r['등급']}</span>
              </td>
            </tr>'''
        tbl += '</tbody></table></div>'
        st.markdown(tbl, unsafe_allow_html=True)
        if len(disp) > 15:
            st.caption(f'↓ 외 {len(disp)-15}개')
    with col_act:
        row = next((r for r in alarm_rows if r['wafer']==sel_w), None)
        if row:
            grp = row['그룹']
            st.markdown('<div class="sec-lbl">조치 권고</div>', unsafe_allow_html=True)
            render_actions(grp)

with tab3:
    st.markdown('<div class="sec-lbl">웨이퍼 처리 이력 — 전체 처리 순서 한눈에</div>',
                unsafe_allow_html=True)
    lc = st.columns(8)
    for ci,(lb,co) in enumerate([
        ('실험29','#2E4057'),('실험31','#3D5166'),('실험33','#4A6274'),
        ('TCP','#2E4057'),('RF','#3D5166'),('압력','#7D1C2F'),
        ('가스','#4A6274'),('He','#6B7C8D')]):
        lc[ci].markdown(f"<span style='color:{co};font-size:.78rem;'>■ {lb}</span>",
                        unsafe_allow_html=True)
    alarm_set = set(wafer_arr[np.where(alarms)[0]])
    fp_set    = set(wafer_arr[np.where(alarms & (fault_labels=='normal'))[0]])
    gcm = {'TCP 관련':'#2E4057','RF 관련':'#3D5166','압력 관련':'#7D1C2F',
           '가스 관련':'#4A6274','He Chuck':'#6B7C8D','정상':'#A0ADB8'}
    all_wids = [str(int(w)) for w in R['wids']]
    def fmt3(w):
        ix = np.where(wafer_arr==w)[0]
        if len(ix)==0: return w
        fn_ = fault_name_lbl[ix[0]]
        grp_= FAULT_GROUP.get(fn_,'정상')
        if w in fp_set:    tag = 'FP'
        elif w in alarm_set: tag = f'⚠ {classified.get(ix[0], fn_)}'
        else:              tag = '정상'
        return f'{w}  [{tag}]'
    sel3 = st.selectbox('웨이퍼 선택', all_wids, format_func=fmt3, key='sel3')
    ix3 = np.where(wafer_arr==sel3)[0]

    # ── 연속 이상 감지 경고 패널 (웨이퍼 선택 아래, 그리드 위) ──
    if len(ix3) > 0:
        i3_pre  = ix3[0]
        win_pre  = [i for i in range(max(0,i3_pre-2), min(len(wafer_arr),i3_pre+3))]
        cons_pre = [i for i in win_pre if alarms[i]]
        if len(cons_pre) >= 2:
            ns = ' → '.join([
                f"{wafer_arr[i]}({'FP' if fault_labels[i]=='normal' else classified.get(i,'?')})"
                for i in cons_pre])
            st.markdown(f'''<div class="alarm-banner">
              <span class="alarm-text">⚠ <strong>연속 이상 감지</strong> — {ns}
              &nbsp;·&nbsp; 챔버 전반 불안정 가능성. 전체 점검 권고.</span>
              <span class="alarm-pill">주의</span>
            </div>''', unsafe_allow_html=True)

    sh = "<div style='display:flex;flex-wrap:wrap;gap:3px;margin-bottom:10px;'>"
    for w in all_wids:
        ix = np.where(wafer_arr==w)[0]
        if len(ix)==0: continue
        iw   = ix[0]
        fn_  = fault_name_lbl[iw]
        grp_ = FAULT_GROUP.get(fn_,'정상')
        co   = gcm.get(grp_, '#A0ADB8')
        isal = w in alarm_set
        isfp = w in fp_set
        issel= w == sel3
        brd  = f'2.5px solid {co}' if issel else '1px solid #D8E0E8'
        bg   = ('#FDF1F3' if issel and isal and not isfp
                else ('#FFF8F8' if isal and not isfp
                else ('#FFFBF0' if isfp
                else ('#EEF1F5' if issel else '#fff'))))
        bdg_ = ''
        if isal and not isfp:
            bdg_ = (f'<span style="position:absolute;top:-4px;right:-4px;background:{co};'
                    f'color:#fff;border-radius:50%;width:13px;height:13px;'
                    f'font-size:7px;font-weight:700;display:flex;align-items:center;'
                    f'justify-content:center;">!</span>')
        elif isfp:
            bdg_ = ('<span style="position:absolute;top:-4px;right:-4px;background:#4A6274;'
                    'color:#fff;border-radius:50%;width:13px;height:13px;'
                    'font-size:6px;font-weight:700;display:flex;align-items:center;'
                    'justify-content:center;">FP</span>')
        lbl = fn_[:5] if isal and not isfp else ('FP' if isfp else '정상')
        sh += (f"<div style='position:relative;flex:0 0 46px;border-radius:5px;"
               f"padding:4px 3px;text-align:center;border:{brd};background:{bg};'>"
               f"{bdg_}"
               f"<div style='font-size:13px;font-weight:600;color:#1C2B3A;'>{w}</div>"
               f"<div style='font-size:11px;color:{co};margin-top:1px;'>{lbl}</div>"
               f"</div>")
    sh += "</div>"
    st.markdown(sh, unsafe_allow_html=True)

    if len(ix3) > 0:
        i3   = ix3[0]
        fn3  = fault_name_lbl[i3]
        grp3 = FAULT_GROUP.get(fn3,'정상')
        isal3= bool(alarms[i3])
        isfp3= bool(fault_labels[i3]=='normal') and isal3
        pfn3 = classified.get(i3, fn3)
        exp3 = exp_labels[i3]
        lim3 = SPE_lim.get(exp3, float(np.mean(list(SPE_lim.values()))))
        r3   = SPEs[i3] / lim3
        if isfp3:       st3 = 'FP(오경보)'
        elif not isal3: st3 = '정상'
        elif r3>2.0:    st3 = '위험'
        elif r3>1.2:    st3 = '주의'
        else:           st3 = '확인'
        co3  = gcm.get(grp3,'#A0ADB8')
        rc3  = RISK_COLOR.get(st3, co3)
        rbg3 = RISK_BG.get(st3,'#F4F6F9')
        st.markdown(
            f'<div class="sec-lbl">'
            f'<span style="color:#2E4057;font-weight:600">{sel3}.txm</span>'
            f'&nbsp;·&nbsp;실험 {exp3}&nbsp;·&nbsp;처리 순서 #{i3}'
            f'&nbsp;&nbsp;<span class="risk-badge" style="background:{rbg3};color:{rc3};font-size:.7rem">'
            f'{pfn3 if isal3 and not isfp3 else ("FP" if isfp3 else "정상")}'
            f'</span></div>',
            unsafe_allow_html=True)

        # ── 중단: d1(웨이퍼 정보) + d2(SPE 시계열) + d3(조치 권고) ──
        d1, d2, d3 = st.columns([0.65, 2.8, 1.4])

        with d1:
            p2 = wafer_arr[i3-2] if i3>=2 else '-'
            p1 = wafer_arr[i3-1] if i3>=1 else '-'
            n1 = wafer_arr[i3+1] if i3<len(wafer_arr)-1 else '-'
            n2 = wafer_arr[i3+2] if i3<len(wafer_arr)-2 else '-'
            def wl(w):
                if w=='-': return '-'
                xi = np.where(wafer_arr==w)[0]
                if len(xi)==0: return w
                fn_ = fault_name_lbl[xi[0]]
                g_  = FAULT_GROUP.get(fn_,'정상')
                return f'{w}({g_ if g_!="정상" else "정상"})'
            for k,v in [('웨이퍼 ID',sel3),('실험 배치',f'실험 {exp3}'),
                         ('처리 순서',f'#{i3}/{len(wafer_arr)}'),
                         ('Fault 유형',fn3),('이상 점수',f'{SPEs[i3]:.0f}'),
                         ('상태',st3),
                         ('이전 2개',f'{wl(p2)} → {wl(p1)}'),
                         ('이후 2개',f'{wl(n1)} → {wl(n2)}')]:
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:1rem;padding:3px 0;border-bottom:1px solid #EEF1F5;'>"
                    f"<span style='color:#6B7C8D;'>{k}</span>"
                    f"<span style='color:#1C2B3A;'>{v}</span></div>",
                    unsafe_allow_html=True)

        with d2:
            st.markdown(
                '<div class="sec-lbl">SPE 시점별 이탈 — 점을 클릭하면 신호 기여도 확인</div>',
                unsafe_allow_html=True)
            fig_spe = make_spe_ts_fig(i3, pfn3, 320)
            event = st.plotly_chart(
                fig_spe,
                use_container_width=True,
                on_select='rerun',
                key=f'spe_fig_{sel3}'
            )

        with d3:
            st.markdown('<div class="sec-lbl">조치 권고</div>', unsafe_allow_html=True)
            if isal3 and not isfp3:
                for lv, color, bg, title, desc in ACTION_MAP.get(grp3, []):
                    st.markdown(f'''
                    <div style="background:{bg};border-left:3px solid {color};
                         border-radius:0 6px 6px 0;padding:9px 12px;margin-bottom:8px;">
                      <div style="font-size:.72rem;font-weight:600;text-transform:uppercase;
                           letter-spacing:.3px;color:{color};margin-bottom:3px;">{lv}</div>
                      <div style="font-size:.85rem;font-weight:500;color:#1C2B3A;">{title}</div>
                      <div style="font-size:.78rem;color:#6B7C8D;margin-top:3px;">{desc}</div>
                    </div>''', unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div style='font-size:.82rem;color:#A0ADB8;padding:8px 4px;'>"
                    "이상 없음 — 조치 불필요</div>",
                    unsafe_allow_html=True)

        # 클릭된 시점 추출 (x축 t_norm 기준)
        t_val    = None
        step_idx = None
        if event and event.selection and event.selection.get('points'):
            t_val    = float(event.selection['points'][0]['x'])
            step_idx = max(0, min(N - 1, int(round(t_val * (N - 1)))))

        # ── 하단: 파이차트 + bar chart 상시 표시 ──────────────
        _step = step_idx if step_idx is not None else 0
        _t    = t_val    if t_val    is not None else 0.0
        _t_unit = ''
        groups, top_feat, top_val = get_step_contribution(i3, _step)
        spe_at = float(spe_ts_all[i3][_step])
        exceed = spe_at > 1.0

        lbl_suffix = f'{"⚠ 이탈" if exceed else "정상 범위"}  (t={_t:.2f})'
        st.markdown(
            f'<div class="sec-lbl" style="margin-top:10px">신호 기여도 — {lbl_suffix}</div>',
            unsafe_allow_html=True)

        c_pie, c_bar = st.columns([1, 2])

        with c_pie:
            fig_pie = go.Figure(go.Pie(
                labels=list(groups.keys()),
                values=list(groups.values()),
                hole=0.4,
                marker_colors=['#2E4057','#4A8FA8','#7D1C2F']
            ))
            fig_pie.update_layout(
                **{k:v for k,v in BASE_LAYOUT.items() if k != 'margin'},
                height=300,
                showlegend=True,
                legend=dict(orientation='h', y=-0.12, font_size=12),
                margin=dict(t=10, b=30, l=10, r=10)
            )
            st.plotly_chart(fig_pie, use_container_width=True,
                            key=f'pie_{sel3}_{_step}')

        with c_bar:
            bar_colors = ['#7D1C2F' if v > float(np.mean(top_val))*1.5
                          else '#4A6274' for v in top_val]
            fig_bar = go.Figure(go.Bar(
                x=top_feat, y=top_val,
                marker_color=bar_colors,
                text=[f'{v:.4f}' for v in top_val],
                textposition='outside', textfont_size=12
            ))
            fig_bar.update_layout(
                **{k:v for k,v in BASE_LAYOUT.items() if k != 'margin'},
                height=300,
                xaxis_tickangle=-35,
                xaxis_tickfont_size=12,
                yaxis_title='기여도 (잔차²)',
                margin=dict(t=30, b=110, l=40, r=10)
            )
            st.plotly_chart(fig_bar, use_container_width=True,
                            key=f'bar_{sel3}_{_step}')

        if isal3 and not isfp3:
            pass  # 조치 권고는 오른쪽 d3 컬럼에서 표시