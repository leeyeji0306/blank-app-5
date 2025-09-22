import pandas as pd
import numpy as np
from datetime import datetime

# ------------------------------
# 1️⃣ CSV 파일 로드 (없으면 예시 데이터 생성)
# ------------------------------
def load_nasa_power_from_csv(path="nasa_power.csv"):
    try:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        # 파일이 없으면 fallback 데이터 생성
        return fallback_data_generator()

def fallback_data_generator():
    today = pd.Timestamp.today()
    dates = pd.date_range(end=today, periods=60)
    return pd.DataFrame({
        "date": dates,
        "일 평균기온(℃)": np.random.uniform(20, 30, len(dates)),
        "일 최고기온(℃)": np.random.uniform(25, 35, len(dates))
    })

# ------------------------------
# 2️⃣ 폭염 flag 생성 (빈 데이터 대비)
# ------------------------------
def make_heatwave_flags(df):
    if df.empty:
        # 빈 데이터면 빈 DataFrame 반환
        return pd.DataFrame(columns=["date", "value", "group"])
    
    # '일 최고기온(℃)'만 필터링
    w = df[df['group'] == "일 최고기온(℃)"].copy()
    if w.empty or 'value' not in w.columns:
        return pd.DataFrame(columns=["date", "value", "group"])
    
    # 폭염일 정의
    w["폭염_flag"] = w["value"] >= 33
    return w

# ------------------------------
# 3️⃣ 사용자 데이터 처리 (melt 안전화)
# ------------------------------
def load_user_table(df):
    if df.empty:
        return pd.DataFrame(columns=["연도","월","폭염일수"])
    
    month_cols = [str(i)+"월" for i in range(1,13)]
    keep_cols = ["연도"]
    
    # melt 시 id_vars 중복 방지
    m = df.melt(id_vars=keep_cols, value_vars=month_cols,
                var_name="월", value_name="폭염일수")
    
    # 숫자형으로 변환, 결측값 0 처리
    m["폭염일수"] = pd.to_numeric(m["폭염일수"], errors="coerce").fillna(0)
    return m

# ------------------------------
# 4️⃣ 이동평균 적용 (빈 데이터, 그룹 없음 대비)
# ------------------------------
def smooth_data(df, smooth_win=3):
    if df.empty:
        return df
    
    # 필요한 그룹만 선택
    gtemp = df["group"].isin(["일 평균기온(℃)","일 최고기온(℃)"])
    if not gtemp.any():
        return df
    
    # rolling mean 적용
    df.loc[gtemp, "value"] = (
        df[gtemp]
        .sort_values("date")
        .groupby("group")["value"]
        .transform(lambda s: s.rolling(smooth_win, min_periods=1).mean())
    )
    return df

# ------------------------------
# 5️⃣ 날짜 제한 처리 (오늘 이후 제거)
# ------------------------------
def clamp_to_today(df):
    if "date" in df.columns:
        today = pd.Timestamp.today()
        df = df[df["date"] <= today]
    return df

# ------------------------------
# 6️⃣ 전체 데이터 처리 파이프라인 예시
# ------------------------------
def process_all(df_nasa, df_user):
    df_nasa = clamp_to_today(df_nasa)
    df_nasa = smooth_data(df_nasa)
    
    user_table = load_user_table(df_user)
    
    heatwave_flags = make_heatwave_flags(df_nasa.melt(id_vars=["date"], 
                                                      value_vars=["일 평균기온(℃)","일 최고기온(℃)"], 
                                                      var_name="group", value_name="value"))
    return df_nasa, user_table, heatwave_flags

