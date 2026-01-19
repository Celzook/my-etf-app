import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# -----------------------------------------------------------
# 1. 페이지 설정
# -----------------------------------------------------------
st.set_page_config(layout="wide", page_title="ETF Pro Dashboard")

st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; }
    [data-testid="stDataFrameResizable"] { border: 1px solid #ddd; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 2. 데이터 수집 및 전처리
# -----------------------------------------------------------
@st.cache_data(ttl=3600)
def get_etf_list():
    try:
        df_etf = fdr.StockListing('ETF/KR')
        if df_etf.empty: return ['069500', '102110'] 
        
        if 'MarCap' in df_etf.columns:
            df_etf['MarCap_Clean'] = df_etf['MarCap'].astype(str).str.replace(',', '').str.replace('억원', '')
            df_etf['MarCap_Clean'] = pd.to_numeric(df_etf['MarCap_Clean'], errors='coerce').fillna(0)
            
            cond_won = df_etf['MarCap_Clean'] >= 20_000_000_000
            cond_ukwon = (df_etf['MarCap_Clean'] < 100_000) & (df_etf['MarCap_Clean'] >= 200)
            
            df_filtered = df_etf[cond_won | cond_ukwon]
            
            if len(df_filtered) < 10:
                return df_etf.head(70)['Symbol'].tolist()
            return df_filtered['Symbol'].tolist()
        else:
            return df_etf['Symbol'].head(70).tolist()
    except:
        return ['069500', '102110']

def calculate_indicators(df):
    if len(df) < 20: return df
    
    # 이평선
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # 이평선 이격도
    df['MA_Gap'] = ((df['MA5'] - df['MA20']) / df['MA20']) * 100
    
    # 볼린저밴드
    df['BB_Mid'] = df['MA20']
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    
    # %B
    df['PctB'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # RSI
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def analyze_data(ticker_list):
    results = []
    status_text = st.empty()
    progress_bar = st.progress(0)
    total = len(ticker_list)
    
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    try:
        etf_meta = fdr.StockListing('ETF/KR')
        name_map = dict(zip(etf_meta['Symbol'], etf_meta['Name']))
    except:
        name_map = {}

    for i, ticker in enumerate(ticker_list):
        if i % 10 == 0:
            progress_bar.progress((i + 1) / total)
            status_text.text(f"데이터 분석 중... {i+1}/{total} ({ticker})")

        try:
            df = fdr.DataReader(ticker, start_date)
            # 날짜 인덱스 보정
            df.index = pd.to_datetime(df.index)
            
            if len(df) < 60: continue
            
            df = calculate_indicators(df)
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            name = name_map.get(ticker, ticker)
            
            # Signals
            ma_sig_text = ""
            if prev['MA5'] <= prev['MA20'] and curr['MA5'] > curr['MA20']: ma_sig_text = "✅ Golden Cross"
            elif prev['MA5'] >= prev['MA20'] and curr['MA5'] < curr['MA20']: ma_sig_text = "✅ Dead Cross"
            elif curr['MA5'] > curr['MA20']: ma_sig_text = "Golden Zone"
            elif curr['MA5'] < curr['MA20']: ma_sig_text = "Dead Zone"
            
            rsi_val = curr['RSI']
            rsi_sig_text = ""
            if rsi_val >= 60: rsi_sig_text = "✅ Overbought"
            elif rsi_val <= 40: rsi_sig_text = "✅ Oversold"
            
            bb_pct = curr['PctB']
            bb_sig_text = ""
            if bb_pct >= 0.95: bb_sig_text = "✅ Near Upper"
            elif bb_pct <= 0.05: bb_sig_text = "✅ Near Lower"

            results.append({
                'Ticker': ticker,
                'Name': name,
                'Close': curr['Close'],
                'MA_Signal': ma_sig_text,
                'MA_Gap': round(curr['MA_Gap'], 2),
                'RSI_Signal': rsi_sig_text,
                'RSI_Value': round(rsi_val, 2),
                'BB_Signal': bb_sig_text,
                'BB_PctB': round(bb_pct, 2),
                'Data': df
            })
            
        except: continue
            
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results)

# -----------------------------------------------------------
# 3. 차트 그리기 (Y축 자동 스케일링 적용)
# -----------------------------------------------------------
def plot_chart(row):
    df = row['Data'].copy()
    df.index = pd.to_datetime(df.index)
    
    # 최근 120일 데이터만 사용
    df = df.iloc[-120:]
    
    # -----------------------------------------
    # [핵심] Y축 범위 동적 계산 (Dynamic Scaling)
    # -----------------------------------------
    # 캔들(저가, 고가)와 볼린저밴드(상단, 하단) 중 가장 낮은 값과 높은 값을 찾음
    min_candidates = [df['Low'].min(), df['BB_Lower'].min()]
    max_candidates = [df['High'].max(), df['BB_Upper'].max()]
    
    # NaN 값 제외하고 최소/최대 찾기
    y_min = min([x for x in min_candidates if not np.isnan(x)])
    y_max = max([x for x in max_candidates if not np.isnan(x)])
    
    # 여백(Padding) 5% 추가 (차트 위아래가 답답하지 않게)
    padding = (y_max - y_min) * 0.05
    y_range = [y_min - padding, y_max + padding]
    # -----------------------------------------

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3],
                        subplot_titles=(f"{row['Name']}", ""))

    # Price
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'], name='Price', showlegend=False), row=1, col=1)
    
    # MA
    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='orange', width=1), name='MA5'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='blue', width=1), name='MA20'), row=1, col=1)
    
    # BB (Red, No Fill)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='red', width=1), name='BB Up'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='red', width=1), name='BB Low'), row=1, col=1)

    # Annotation
    ma_txt = row['MA_Signal'] if "Cross" in row['MA_Signal'] else row['MA_Signal'].replace("✅ ", "")
    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.98,
        text=f"<b>{ma_txt}</b>", showarrow=False, font=dict(color="black"),
        bgcolor="rgba(255,255,255,0.8)", borderwidth=1, row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#008080', width=2), name='RSI'), row=2, col=1)
    fig.add_hline(y=60, line_dash="dash", line_color="orange", row=2, col=1)
    fig.add_hline(y=40, line_dash="dash", line_color="navy", row=2, col=1)

    # Layout Setting
    min_date = df.index[0]
    max_date = df.index[-1]

    fig.update_layout(height=400, margin=dict(t=30, b=0, l=10, r=10), showlegend=False)
    
    # X축 고정 + Y축 범위 강제 적용
    fig.update_xaxes(range=[min_date, max_date], rangeslider_visible=False)
    
    # 여기서 계산해둔 y_range를 적용합니다.
    fig.update_layout(yaxis=dict(range=y_range)) 
    
    return fig

# -----------------------------------------------------------
# 4. 메인 UI
# -----------------------------------------------------------
def main():
    c1, c2 = st.columns([8,2])
    c1.title("📊 ETF Pro : Smart Dashboard")
    c1.caption("필터: 시총 200억↑ | 조건: **AND (교집합)** 적용 | **체크박스 선택 시 차트 자동 생성**")
    if c2.button("🔄 데이터 업데이트"):
        st.session_state['loaded'] = False
        st.rerun()

    if 'loaded' not in st.session_state: st.session_state['loaded'] = False
    
    if not st.session_state['loaded']:
        with st.spinner("시장 데이터 스캔 중..."):
            tickers = get_etf_list()
            df_res = analyze_data(tickers)
            st.session_state['df_res'] = df_res
            st.session_state['loaded'] = True
    
    if st.session_state['loaded']:
        df = st.session_state['df_res']
        if df.empty:
            st.error("데이터 없음")
            return

        st.divider()
        
        # --- Filters ---
        f1, f2, f3 = st.columns(3)
        with f1: sel_ma = st.selectbox("1. MA 시그널", ["All", "Golden Cross", "Dead Cross", "Golden Zone", "Dead Zone"])
        with f2: sel_rsi = st.selectbox("2. RSI 상태", ["All", "Overbought (60↑)", "Oversold (40↓)"])
        with f3: sel_bb = st.selectbox("3. 볼린저밴드", ["All", "Near Upper", "Near Lower"])

        mask = pd.Series([True]*len(df))
        if sel_ma != "All":
            if "Golden Cross" in sel_ma: mask &= (df['MA_Signal'] == "✅ Golden Cross")
            elif "Dead Cross" in sel_ma: mask &= (df['MA_Signal'] == "✅ Dead Cross")
            elif "Golden Zone" in sel_ma: mask &= (df['MA_Signal'].str.contains("Golden"))
            elif "Dead Zone" in sel_ma: mask &= (df['MA_Signal'].str.contains("Dead"))
        if sel_rsi != "All":
            if "Overbought" in sel_rsi: mask &= (df['RSI_Signal'] == "✅ Overbought")
            elif "Oversold" in sel_rsi: mask &= (df['RSI_Signal'] == "✅ Oversold")
        if sel_bb != "All":
            if "Upper" in sel_bb: mask &= (df['BB_Signal'] == "✅ Near Upper")
            elif "Lower" in sel_bb: mask &= (df['BB_Signal'] == "✅ Near Lower")
            
        filtered_df = df[mask].copy()
        
        # --- Table ---
        st.success(f"검색된 ETF: **{len(filtered_df)}** 종목")
        
        filtered_df.insert(0, "선택", False)
        
        display_cols = ['선택', 'Ticker', 'Name', 'Close', 'MA_Signal', 'MA_Gap', 'RSI_Signal', 'RSI_Value', 'BB_Signal', 'BB_PctB']
        
        edited_df = st.data_editor(
            filtered_df[display_cols],
            column_config={
                "선택": st.column_config.CheckboxColumn("View", width="small", default=False),
                "Ticker": "코드",
                "Name": "종목명",
                "Close": st.column_config.NumberColumn("현재가", format="%d원"),
                "MA_Gap": st.column_config.NumberColumn("이격도", format="%.2f%%"),
                "RSI_Value": st.column_config.NumberColumn("RSI", format="%.1f"),
                "BB_PctB": st.column_config.NumberColumn("BB위치", format="%.2f")
            },
            disabled=["Ticker", "Name", "Close", "MA_Signal", "MA_Gap", "RSI_Signal", "RSI_Value", "BB_Signal", "BB_PctB"],
            hide_index=True,
            use_container_width=True
        )
        
        # --- Charts ---
        selected_rows = edited_df[edited_df['선택'] == True]
        
        if not selected_rows.empty:
            st.divider()
            st.markdown(f"### 📈 차트 모아보기 ({len(selected_rows)}개)")
            
            cols = st.columns(2)
            
            for idx, row in enumerate(selected_rows.itertuples()):
                target_ticker = row.Ticker
                original_data = filtered_df[filtered_df['Ticker'] == target_ticker]
                
                if not original_data.empty:
                    chart_fig = plot_chart(original_data.iloc[0])
                    with cols[idx % 2]:
                        st.plotly_chart(chart_fig, use_container_width=True)
        else:
            st.info("👆 목록 왼쪽의 체크박스를 클릭하면 차트가 표시됩니다.")

if __name__ == "__main__":
    main()