
# 라이브러리

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as spst
import datetime
import joblib
from keras.models import load_model
from urllib.parse import quote
import streamlit as st
import folium
from streamlit_folium import st_folium
import branca
from geopy.geocoders import Nominatim
import ssl
from urllib.request import urlopen
import plotly.express as px
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *

# warnings.filterwarnings(action='ignore')
# %config InlineBackend.figure_format='retina'

plt.rc('font', family='Malgun Gothic')


# 구현 할 기능
# 1. 항공권 가격 예측 V
# 2. folium 지도에 출발 ~ 도착(경로) 공항 표시  
# 3. 인도 항공사 별 평균 가격


# -------------------------------------- 함수부 --------------------------------------

# data 전처리 함수 생성

def preprocessing(df):
    
    journey_day_list = {'Monday' : 1 , 'Tuesday' : 5, 'Wednesday' : 6, 'Thurday' : 4, 'Friday' : 0, 'Saturday' : 2, 'Sunday' : 3}

    airline_list = {'Vistara' : 8, 'Air India' : 0, 'Indigo' : 5, 'AirAsia' : 1, 'GO FIRST' : 4, 'SpiceJet' : 6, 'AkasaAir' : 2, 'AllianceAir' : 3, 'StarAir' : 7}

    class_list = {'Economy' : 1, 'Business' : 0, 'Premium Economy' : 3, 'First' : 2}

    source_list = {'Delhi' : 3, 'Mumbai' : 6, 'Bangalore' : 1, 'Hyderabad' : 4, 'Chennai' : 2, 'Ahmedabed' : 0, 'Kolkata' : 5}

    departure_list = {'Before 6 AM' : 3, '6 AM - 12 PM' : 1, '12 PM - 6 PM' : 0, 'After 6 PM' : 2}

    total_stops_list = {'1-stop' : 0, 'non-stop' : 2, '2+-stop' : 1}

    arrival_list = {'Before 6 AM' : 3, '6 AM - 12 PM' : 1, '12 PM - 6 PM' : 0, 'After 6 PM' : 2}

    destination_list = {'Delhi' : 3, 'Mumbai' : 6, 'Bangalore' : 1, 'Hyderabad' : 4, 'Chennai' : 2, 'Ahmedabed' : 0, 'Kolkata' : 5}


    cols = df.select_dtypes('object').columns

    df['Journey_day'] = journey_day_list[Journey_day]

    df['Airline'] = airline_list[Airline]

    df['Class'] = class_list[Class]

    df['Source'] = source_list[Source]

    df['Departure'] = departure_list[Departure]

    df['Total_stops'] = total_stops_list[Total_stops]

    df['Arrival'] = arrival_list[Arrival]

    df['Destination'] = destination_list[Destination]

#     x = data.drop('Fare', axis=1)
#     y = data['Fare']
    
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    
#     scaler = StandardScaler()
    
#     scaler.fit(x_train)
    
#     x_train = scaler.transform(x_train)
#     x_test = scaler.transform(x_test)

    return df



# 항공권 가격 예측 함수 생성

def fare_predict(fare_data):
    
    test_df = pd.DataFrame(fare_data) # df 생성
    
    test_df = preprocessing(test_df)
    
    model_XGC = joblib.load('fare_model_xgb.pkl')  # model load
    
    pred_y_XGC = model_XGC.predict(test_df)  # predict
    
    return pred_y_XGC[0]


# -------------------------------------- 인터페이스부 --------------------------------------

# 레이아웃 구성하기 
st.set_page_config(layout="wide")

# tabs 만들기 
t1, t2 = st.tabs(['가격예측', '대시보드'])

# tab1 내용물 구성하기 
with t1:

    # 제목 넣기
    st.markdown("## 항공권 가격 예측 서비스 : India")

    st.image('plane.png')
        
    # 시간 정보 가져오기 
    now_date = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=9)

    
    # 환자정보 널기
    st.markdown("#### 예약 날짜 입력")

    ## -------------------- ▼ 1-1그룹 날짜/시간 입력 cols 구성(출동일/날짜정보(input_date)/출동시간/시간정보(input_time)) ▼ --------------------
     
    c1, c2, c3, c4 = st.columns([0.1, 0.3, 0.1, 0.3])
    with c1:
        st.info('출발일')
    with c2:
        Date_of_journey = st.date_input('출발일')
    with c3:
        st.info('출발요일')
    with c4:
        Journey_day = st.radio('출발요일', ['Monday', 'Tuesday', 'Wednesday', 'Thurday', 'Friday', 'Saturday', 'Sunday'], horizontal = True)

    c1, c2, c3, c4 = st.columns([0.1, 0.3, 0.1, 0.3])
    with c1:
        st.info('출발시간')
    with c2:
        Departure = st.radio('출발시간', ['Before 6 AM', '6 AM - 12 PM', '12 PM - 6 PM', 'After 6 PM'], horizontal = True)
    with c3:
        st.info('도착시간')
    with c4:
        Arrival = st.radio('도착공항', ['Before 6 AM', '6 AM - 12 PM', '12 PM - 6 PM', 'After 6 PM'], horizontal = True)
        
    c1, c2, c3, c4 = st.columns([0.1, 0.3, 0.1, 0.3])
    with c1:
        st.info('출발공항')
    with c2:
        Source = st.radio('출발공항', ['Delhi', 'Mumbai', 'Bangalore', 'Hyderabad', 'Chennai', 'Ahmedabed', 'Kolkata'], horizontal = True)
    with c3:
        st.info('도착공항')
    with c4:
        Destination = st.radio('도착공항', ['Delhi', 'Mumbai', 'Bangalore', 'Hyderabad', 'Chennai', 'Ahmedabed', 'Kolkata'], horizontal = True)
        
    c1, c2, c3, c4 = st.columns([0.1, 0.3, 0.1, 0.3])
    with c1:
        st.info('항공사')
    with c2:
        Airline = st.radio('항공사', ['Vistara', 'Air India', 'Indigo', 'AirAsia', 'GO FIRST', 'SpiceJet', 'AkasaAir', 'AllianceAir', 'StarAir'], horizontal = True)
    with c3:
        st.info('클래스')
    with c4:
        Class = st.radio('클래스', ['Economy', 'Business', 'Premium Economy', 'First'], horizontal = True)    
    
    c1, c2, c3, c4 = st.columns([0.1, 0.3, 0.1, 0.3])
    with c1:
        st.info('경유횟수')
    with c2:
        Total_stops = st.radio('경유횟수', ['1-stop', 'non-stop', '2+-stop'], horizontal = True)



    
    c1, c2 = st.columns([0.2, 0.7]) # col 나누기
    with c1:
        st.error("항공권 가격 예측 시스템")

    with c2: 
        if st.button('예상금액 조회하기'):

            
#  target인 'Fare' 을 제외한 나머지 features 정보 입력
                    
            fare_data = {
                'Journey_day' : [Journey_day],
                'Airline' : [Airline],
                'Class' : [Class],
                'Source' : [Source],
                'Departure' : [Departure],
                'Total_stops' : [Total_stops],
                'Arrival' : [Arrival],
                'Destination' : [Destination],
                'Duration_in_hours' : [12],
                'Days_left' : [25]
            }
        
            airline_mean_list = {'Vistara' : 27240.7, 'Air India' : 26914.5, 'StarAir' : 9792.7, 'Indigo' : 8198.7,
                         'SpiceJet' : 8109.7, 'GO FIRST': 8015.1, 'AirAsia' : 7092.2, 'AllianceAir' : 4077.5, 'AkasaAir' : 3570.0}
        
            special_m = round(fare_predict(fare_data), 0)
            
            my_diff = round(special_m - airline_mean_list[Airline], 1)
            
            if my_diff > 0:
                my_diff = '+' + str(my_diff)
                
            st.markdown(f"### 예측된 항공권 가격은 {special_m}$ 입니다.")
            st.markdown(f"#### (같은 항공사의 평균 항공권 가격 대비 : {my_diff}$)")


# -------------------------------------- 대시보드부 --------------------------------------
            
with t2:
    st.markdown("## Flight Analysis : India")
    
    st.error("통계 분석")
    
    data_1 = pd.read_csv('Cleaned_dataset.csv')
    
    df_1 = pd.DataFrame({"Airline": data_1['Airline'].value_counts().index, "cnt" : data_1['Airline'].value_counts().values})
    
    df_2 = data_1.groupby(by = 'Airline', as_index = False)['Fare'].mean().sort_values(by = 'Fare', ascending = False)
    df_2['Fare'] = round(df_2['Fare'],1) 
    
    df_3 = data_1.groupby(by = 'Class', as_index = False)['Fare'].mean().sort_values(by = 'Fare', ascending = False)
    df_3['Fare'] = round(df_3['Fare'],1) 
    
    c1, c2, c3 = st.columns([0.3, 0.3, 0.3])
    
    with c1:
        fig = px.pie(df_1, names='Airline', values='cnt', title = '항공사 점유율' , hole = 0.3)
        fig.update_traces(textposition = 'inside', textinfo = 'percent + label')
        fig.update_layout(font=dict(size=18))
        st.plotly_chart(fig)

    with c2:

        fig = px.pie(df_2, names='Airline', values='Fare', title = '항공사 별 평균 가격' , hole = 0.3)
        fig.update_traces(textposition = 'inside', textinfo = 'value + label')
        fig.update_layout(font=dict(size=18))
        st.plotly_chart(fig)

    with c3:

        fig = px.pie(df_3, names='Class', values='Fare', title = '클래스 별 평균 가격' , hole = 0.3)
        fig.update_traces(textposition = 'inside', textinfo = 'value + label')
        fig.update_layout(font=dict(size=18))
        st.plotly_chart(fig)
        
