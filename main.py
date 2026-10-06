import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="기온 선형회귀 평가", layout="wide")

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year

    # 연도별 평균기온과 관측일수
    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 2025년 이후 + 관측일 300일 미만 제외
    annual = annual[
        (annual["연도"] <= 2025) &
        (annual["관측일수"] >= 300)
    ].copy()

    return annual.sort_values("연도")


annual = load_data()

# --------------------------------------------------
# 학습/테스트 데이터
# --------------------------------------------------

train_50 = annual[
    (annual["연도"] >= 1956) &
    (annual["연도"] <= 2005)
].copy()

train_100 = annual[
    (annual["연도"] >= 1906) &
    (annual["연도"] <= 2005)
].copy()

test = annual[
    (annual["연도"] >= 2006) &
    (annual["연도"] <= 2025)
].copy()


# --------------------------------------------------
# 모델 학습 및 평가 함수
# --------------------------------------------------

def train_and_evaluate(train_df, test_df):
    X_train = train_df[["연도"]]
    y_train = train_df["연평균기온"]

    X_test = test_df[["연도"]]
    y_test = test_df["연평균기온"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mae, mse, r2, y_pred


model_50, mae_50, mse_50, r2_50, pred_50 = train_and_evaluate(
    train_50, test
)

model_100, mae_100, mse_100, r2_100, pred_100 = train_and_evaluate(
    train_100, test
)


# --------------------------------------------------
# 화면
# --------------------------------------------------

st.title("기온 예측기: 선형회귀 모델 평가")

st.write(
    "서울의 연평균기온을 이용하여 과거 데이터를 학습하고 "
    "2006~2025년의 기온을 얼마나 잘 예측하는지 비교합니다."
)

st.subheader("데이터 분할")

st.write(
    "훈련 데이터: 1956~2005년(최근 50년), "
    "1906~2005년(최근 100년)"
)

st.write("공통 테스트 데이터: 2006~2025년")

col1, col2, col3 = st.columns(3)

col1.metric("50년 훈련 데이터", f"{len(train_50)}개 연도")
col2.metric("100년 훈련 데이터", f"{len(train_100)}개 연도")
col3.metric("공통 테스트 데이터", f"{len(test)}개 연도")


# --------------------------------------------------
# 평가 결과
# --------------------------------------------------

st.subheader("모델 성능 비교")

result = pd.DataFrame({
    "모델": [
        "최근 50년 학습 (1956~2005)",
        "최근 100년 학습 (1906~2005)"
    ],
    "기울기 (℃/년)": [
        model_50.coef_[0],
        model_100.coef_[0]
    ],
    "MAE (℃)": [
        mae_50,
        mae_100
    ],
    "MSE (℃²)": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})

st.dataframe(
    result.style.format({
        "기울기 (℃/년)": "{:.5f}",
        "MAE (℃)": "{:.3f}",
        "MSE (℃²)": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True
)


# --------------------------------------------------
# 모델별 그래프
# --------------------------------------------------

st.subheader("훈련 데이터와 테스트 데이터")

fig = go.Figure()

# 전체 실제 데이터
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        hovertemplate="%{x}년<br>%{y:.2f} ℃<extra></extra>"
    )
)

# 50년 회귀선
years_50 = np.arange(1956, 2026)
pred_line_50 = model_50.predict(
    pd.DataFrame({"연도": years_50})
)

fig.add_trace(
    go.Scatter(
        x=years_50,
        y=pred_line_50,
        mode="lines",
        name="50년 학습 회귀선"
    )
)

# 100년 회귀선
years_100 = np.arange(1906, 2026)
pred_line_100 = model_100.predict(
    pd.DataFrame({"연도": years_100})
)

fig.add_trace(
    go.Scatter(
        x=years_100,
        y=pred_line_100,
        mode="lines",
        name="100년 학습 회귀선"
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest"
)

fig.update_xaxes(tickformat="d")

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 테스트 데이터 예측 비교
# --------------------------------------------------

st.subheader("2006~2025년 실제값과 예측값")

prediction_df = test[["연도", "연평균기온"]].copy()

prediction_df["50년 모델 예측"] = pred_50
prediction_df["100년 모델 예측"] = pred_100

prediction_df = prediction_df.rename(
    columns={"연평균기온": "실제 평균기온"}
)

st.dataframe(
    prediction_df.style.format({
        "실제 평균기온": "{:.2f}",
        "50년 모델 예측": "{:.2f}",
        "100년 모델 예측": "{:.2f}"
    }),
    use_container_width=True
)


# --------------------------------------------------
# 비교 해석
# --------------------------------------------------

st.subheader("결과 비교")

if mae_50 < mae_100:
    mae_result = "50년 모델의 MAE가 더 작아 테스트 데이터의 평균적인 오차가 더 적었습니다."
else:
    mae_result = "100년 모델의 MAE가 더 작아 테스트 데이터의 평균적인 오차가 더 적었습니다."

if r2_50 > r2_100:
    r2_result = "50년 모델의 R²가 더 높아 테스트 데이터의 변동을 더 잘 설명했습니다."
else:
    r2_result = "100년 모델의 R²가 더 높아 테스트 데이터의 변동을 더 잘 설명했습니다."

st.write(
    f"""
    - **50년 모델 기울기:** {model_50.coef_[0]:.5f} ℃/년
    - **100년 모델 기울기:** {model_100.coef_[0]:.5f} ℃/년
    - **기울기 차이:** {abs(model_50.coef_[0] - model_100.coef_[0]):.5f} ℃/년

    - **MAE:** {mae_result}
    - **MSE:** 두 모델 중 값이 작은 모델이 큰 오차의 영향을 덜 받았습니다.
    - **R²:** {r2_result}
    """
)

st.caption(
    "MAE는 실제값과 예측값의 절대 오차 평균이고, "
    "MSE는 오차를 제곱하여 평균낸 값입니다. "
    "R²는 모델이 종속변수의 변동을 얼마나 설명하는지를 나타냅니다."
)
