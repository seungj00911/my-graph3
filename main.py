import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import pearsonr

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="서울 기온 예측기", layout="centered")
st.title("🌡️ 서울 기온 예측기")

# 2. 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_preprocess_data():
    # 데이터 읽기 (인코딩 UTF-8)
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    
    # '날짜' 열을 datetime 형태로 변환하여 '연도' 추출
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    # 평균기온 결측치 제거
    df = df.dropna(subset=['평균기온'])
    
    # 연도별 관측일 수 및 평균기온 계산
    yearly_stats = df.groupby('연도').agg(
        관측일수=('평균기온', 'count'),
        연평균기온=('평균기온', 'mean')
    ).reset_index()
    
    # 조건 필터링: 2025년까지의 데이터 && 관측일수가 300일 이상인 해만 포함
    filtered_df = yearly_stats[(yearly_stats['연도'] <= 2025) & (yearly_stats['관측일수'] >= 300)].copy()
    
    # 회귀분석을 위한 독립변수 계산 (1908년부터 지난 연수)
    filtered_df['경과연수'] = filtered_df['연도'] - 1908
    
    return filtered_df

try:
    data = load_and_preprocess_data()
    
    # 3. 데이터 요약 정보 계산
    num_years = len(data)
    start_year = int(data['연도'].min())
    end_year = int(data['연도'].max())
    
    # 4. 회귀 직선 및 상관계수 계산
    X = data['경과연수'].values
    Y = data['연평균기온'].values
    
    # numpy.polyfit을 이용한 1차 선형 회귀 (기존 변수 활용)
    slope, intercept = np.polyfit(X, Y, 1)
    
    # 상관계수 계산
    corr, _ = pearsonr(data['연도'], Y)
    
    # 회귀 예측 함수
    def predict_temp(target_year):
        elapsed = target_year - 1908
        return slope * elapsed + intercept

    # 5. 상단 정보 출력
    st.subheader("📊 데이터 분석 요약")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("분석된 해의 개수", f"{num_years}개")
    col2.metric("시작 연도", f"{start_year}년")
    col3.metric("끝 연도", f"{end_year}년")
    col4.metric("상관계수 (r)", f"{corr:.3f}")
    
    # 6. 예측 슬라이더 및 큰 화면 표시
    st.markdown("---")
    st.subheader("🔮 미래 기온 예측하기")
    selected_year = st.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1)
    
    predicted_val = predict_temp(selected_year)
    st.markdown(f"<div style='text-align: center; margin: 20px 0;'><h4>{selected_year}년 서울 예상 연평균 기온</h4>"
                f"<h1 style='color: #FF4B4B; font-size: 4rem;'>{predicted_val:.2f} °C</h1></div>", unsafe_allow_data=True)

    # 7. Plotly 시각화 (산점도 + 회귀 직선)
    st.markdown("---")
    st.subheader("📈 연도별 기온 변화 및 회귀선")
    
    # 시각화를 위한 전체 연도 범위 설정 (1908년부터 데이터 끝 혹은 슬라이더 선택 연도까지 커버)
    plot_years = np.arange(start_year, max(end_year, selected_year) + 1)
    plot_elapsed = plot_years - 1908
    line_y = slope * plot_elapsed + intercept
    
    fig = go.Figure()
    
    # 실제 데이터 산점도
    fig.add_trace(go.Scatter(
        x=data['연도'], y=Y,
        mode='markers',
        name='실제 연평균기온',
        marker=dict(color='#1f77b4', size=8)
    ))
    
    # 회귀선 추세선
    fig.add_trace(go.Scatter(
        x=plot_years, y=line_y,
        mode='lines',
        name='회귀 직선 (추세선)',
        line=dict(color='#ff7f0e', width=3)
    ))
    
    # 선택된 예측 지점 표시
    fig.add_trace(go.Scatter(
        x=[selected_year], y=[predicted_val],
        mode='markers+text',
        name=f'선택된 해 ({selected_year}년)',
        marker=dict(color='#FF4B4B', size=14, symbol='star'),
        text=[f"{predicted_val:.2f}°C"],
        textposition="top center"
    ))
    
    fig.update_layout(
        xaxis_title="연도",
        yaxis_title="평균 기온 (°C)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"데이터를 불러오거나 처리하는 중 오류가 발생했습니다: {e}")
