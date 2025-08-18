import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from youtube_transcript_api import YouTubeTranscriptApi
from textblob import TextBlob
import requests
from datetime import datetime, timedelta
import time
import warnings
import re
from typing import Dict, List, Tuple
import json
import openai
from bs4 import BeautifulSoup
from urllib.parse import quote
from pytrends.request import TrendReq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="서학개미의 투자 탐구생활 🐜🔍",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .buy-signal {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .sell-signal {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
    .hold-signal {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)


class SectorTreemapAnalyzer:
    """섹터별 트리맵 분석 클래스"""
    
    @staticmethod
    @st.cache_data(ttl=3600)  # 1시간 캐싱
    def get_sp500_stocks() -> pd.DataFrame:
        """S&P 500 종목 리스트 가져오기"""
        try:
            # Wikipedia에서 S&P 500 종목 리스트 가져오기
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_df = tables[0]
            
            # 필요한 컬럼만 선택하고 정리
            sp500_df = sp500_df[['Symbol', 'Security', 'GICS Sector']]
            sp500_df.columns = ['symbol', 'company_name', 'sector']
            
            # 섹터명 정리 (공백 제거 및 간소화)
            sector_mapping = {
                'Information Technology': 'Technology',
                'Health Care': 'Healthcare', 
                'Financials': 'Financial',
                'Consumer Discretionary': 'Consumer_Discretionary',
                'Communication Services': 'Communication',
                'Industrials': 'Industrial',
                'Consumer Staples': 'Consumer_Staples',
                'Energy': 'Energy',
                'Materials': 'Materials',
                'Real Estate': 'Real_Estate',
                'Utilities': 'Utilities'
            }
            
            sp500_df['sector'] = sp500_df['sector'].map(sector_mapping).fillna(sp500_df['sector'])
            
            return sp500_df
            
        except Exception as e:
            st.error(f"S&P 500 종목 리스트 가져오기 실패: {str(e)}")
            # 백업용 주요 종목들
            return pd.DataFrame({
                'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ'],
                'company_name': ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.', 'Amazon.com Inc.', 'NVIDIA Corp.', 
                               'Meta Platforms Inc.', 'Tesla Inc.', 'Berkshire Hathaway Inc.', 'UnitedHealth Group Inc.', 'Johnson & Johnson'],
                'sector': ['Technology', 'Technology', 'Technology', 'Technology', 'Technology', 
                          'Technology', 'Technology', 'Financial', 'Healthcare', 'Healthcare']
            })
    
    @staticmethod
    @st.cache_data(ttl=1800)  # 30분 캐싱
    def get_sector_market_data(sp500_df: pd.DataFrame, sector_name: str, top_n: int = 20) -> pd.DataFrame:
        """특정 섹터의 시가총액 상위 N개 종목 데이터 가져오기"""
        try:
            # 해당 섹터 종목들 필터링
            sector_stocks = sp500_df[sp500_df['sector'] == sector_name]['symbol'].tolist()
            
            if not sector_stocks:
                st.warning(f"{sector_name} 섹터에 해당하는 종목이 없습니다.")
                return pd.DataFrame()
            
            sector_data = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, symbol in enumerate(sector_stocks):
                try:
                    status_text.text(f"데이터 수집 중: {symbol} ({i+1}/{len(sector_stocks)})")
                    progress_bar.progress((i + 1) / len(sector_stocks))
                    
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="5d")  # 최근 5일 데이터
                    
                    if hist.empty or len(hist) < 2:
                        continue
                    
                    # 시가총액이 없거나 0인 경우 스킵
                    market_cap = info.get('marketCap', 0)
                    if not market_cap or market_cap <= 0:
                        continue
                    
                    # 가격 정보
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    
                    # 회사명 정리
                    company_name = info.get('shortName', info.get('longName', symbol))
                    if len(company_name) > 30:
                        company_name = company_name[:27] + "..."
                    
                    sector_data.append({
                        'symbol': symbol,
                        'company_name': company_name,
                        'sector': sector_name,
                        'market_cap': market_cap,
                        'market_cap_b': market_cap / 1e9,  # 10억 달러 단위
                        'current_price': current_price,
                        'prev_price': prev_price,
                        'change_pct': change_pct,
                        'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0,
                        'pe_ratio': info.get('trailingPE', 0),
                        'dividend_yield': info.get('dividendYield', 0) or 0
                    })
                    
                    # API 호출 제한을 위한 지연
                    time.sleep(0.1)
                    
                except Exception as e:
                    continue
            
            progress_bar.empty()
            status_text.empty()
            
            if not sector_data:
                st.warning(f"{sector_name} 섹터의 데이터를 가져올 수 없습니다.")
                return pd.DataFrame()
            
            # DataFrame으로 변환하고 시가총액 기준으로 정렬
            df = pd.DataFrame(sector_data)
            df = df.sort_values('market_cap', ascending=False).head(top_n)
            
            return df
            
        except Exception as e:
            st.error(f"섹터 데이터 조회 실패: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def create_sector_treemap(sector_data: pd.DataFrame, sector_name: str) -> go.Figure:
        """섹터별 트리맵 생성 (스마트 텍스트 색상)"""
        try:
            if sector_data.empty:
                return go.Figure()
            
            # 색상과 텍스트 색상을 함께 결정
            colors = []
            
            for change in sector_data['change_pct']:
                if change > 2:
                    colors.append('#00AA00')  # 진한 초록 -> 밝기를 조금 낮춰서 검은 텍스트가 보이도록
                elif change > 0:
                    colors.append('#90EE90')  # 연한 초록
                elif change > -2:
                    colors.append('#FFB6C1')  # 연한 빨강
                else:
                    colors.append('#CC0000')  # 진한 빨강 -> 밝기를 조금 낮춰서 검은 텍스트가 보이도록
            
            # 호버 텍스트 생성
            hover_text = []
            for _, row in sector_data.iterrows():
                hover_text.append(
                    f"<b>{row['company_name']}</b><br>" +
                    f"종목코드: {row['symbol']}<br>" +
                    f"시가총액: ${row['market_cap_b']:.1f}B<br>" +
                    f"현재가: ${row['current_price']:.2f}<br>" +
                    f"변동률: {row['change_pct']:+.2f}%<br>" +
                    f"P/E 비율: {row['pe_ratio']:.1f}<br>" +
                    f"배당수익률: {row['dividend_yield']:.2f}%"
                )
            
            # 트리맵 생성
            fig = go.Figure(go.Treemap(
                labels=[f"<b>{row['symbol']}</b><br>{row['company_name']}<br>${row['market_cap_b']:.1f}B<br><b>{row['change_pct']:+.1f}%</b>" 
                    for _, row in sector_data.iterrows()],
                values=sector_data['market_cap'],
                parents=[""] * len(sector_data),
                marker=dict(
                    colors=colors,
                    line=dict(width=3, color='white')  # 경계선을 더 두껍게
                ),
                textfont=dict(
                    size=11,  # 크기를 살짝 줄여서 가독성 향상
                    color='black',  # 검은색 텍스트
                    family='Arial Black, Arial, sans-serif'  # 더 굵은 폰트로 가독성 향상
                ),
                hovertext=hover_text,
                hovertemplate='%{hovertext}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'{sector_name} 섹터 시가총액 TOP {len(sector_data)}',
                title_x=0.5,
                font_size=12,
                height=600,
                margin=dict(t=50, l=25, r=25, b=25)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"트리맵 생성 실패: {str(e)}")
            return go.Figure()
    
    @staticmethod
    def create_sector_summary_chart(all_sectors_data: Dict) -> go.Figure:
        """전체 섹터 요약 차트 생성"""
        try:
            if not all_sectors_data:
                return go.Figure()
            
            sector_summary = []
            
            for sector, data in all_sectors_data.items():
                if data.empty:
                    continue
                
                total_market_cap = data['market_cap'].sum()
                avg_change = data['change_pct'].mean()
                top_stock = data.iloc[0]
                
                sector_summary.append({
                    'sector': sector,
                    'total_market_cap_b': total_market_cap / 1e9,
                    'avg_change_pct': avg_change,
                    'stock_count': len(data),
                    'top_stock': f"{top_stock['symbol']} (${top_stock['market_cap_b']:.1f}B)"
                })
            
            if not sector_summary:
                return go.Figure()
            
            summary_df = pd.DataFrame(sector_summary)
            summary_df = summary_df.sort_values('total_market_cap_b', ascending=True)
            
            # 색상 결정 (평균 변동률 기준)
            colors = ['green' if x > 0 else 'red' for x in summary_df['avg_change_pct']]
            
            fig = go.Figure(go.Bar(
                y=summary_df['sector'],
                x=summary_df['total_market_cap_b'],
                orientation='h',
                marker_color=colors,
                text=[f"{x:.1f}% ({summary_df.iloc[i]['stock_count']}개)" 
                      for i, x in enumerate(summary_df['avg_change_pct'])],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>' +
                             '총 시가총액: $%{x:.1f}B<br>' +
                             '평균 변동률: %{text}<br>' +
                             '<extra></extra>'
            ))
            
            fig.update_layout(
                title='섹터별 시가총액 및 평균 변동률',
                xaxis_title='총 시가총액 (Billions $)',
                yaxis_title='섹터',
                height=500,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"섹터 요약 차트 생성 실패: {str(e)}")
            return go.Figure()


class DataManager:
    """데이터 관리 클래스"""
    
    @staticmethod
    @st.cache_data(ttl=3600)  # 1시간 캐싱
    def get_stock_data(symbol: str, period: str = "2y") -> pd.DataFrame:
        """주식 데이터 가져오기"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            st.error(f"데이터 조회 실패: {symbol} - {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_financial_data(symbol: str) -> Dict:
        """재무 데이터 가져오기"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            return {
                'info': info,
                'financials': financials,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow
            }
        except Exception as e:
            st.error(f"재무 데이터 조회 실패: {symbol} - {str(e)}")
            return {}


class PredictionModel:
    """예측 모델 클래스"""
    
    @staticmethod
    def prophet_forecast(data: pd.DataFrame, days: int = 30) -> Tuple[pd.DataFrame, Dict]:
        """Prophet을 사용한 주가 예측"""
        try:
            # Prophet용 데이터 준비
            df = data.reset_index()
            df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            
            # 시간대 정보 제거 (Prophet은 timezone-naive datetime을 요구)
            df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
            
            # 모델 생성 및 학습
            model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
            model.fit(df)
            
            # 예측
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            
            # 성능 평가 (마지막 30일)
            if len(df) > 30:
                train_size = len(df) - 30
                train_df = df[:train_size]
                test_df = df[train_size:]
                
                model_eval = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
                model_eval.fit(train_df)
                
                future_eval = model_eval.make_future_dataframe(periods=30)
                forecast_eval = model_eval.predict(future_eval)
                
                # MAPE 계산
                actual = test_df['y'].values
                predicted = forecast_eval.tail(30)['yhat'].values[:len(actual)]
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            else:
                # 데이터가 부족한 경우 더미 메트릭
                mape = 0.0
                rmse = 0.0
            
            metrics = {'MAPE': mape, 'RMSE': rmse}
            
            return forecast, metrics
            
        except Exception as e:
            st.error(f"Prophet 예측 실패: {str(e)}")
            return pd.DataFrame(), {}
    
    @staticmethod
    def arima_forecast(data: pd.DataFrame, days: int = 30) -> Tuple[pd.DataFrame, Dict]:
        """ARIMA를 사용한 주가 예측"""
        try:
            prices = data['Close'].values
            
            # ARIMA 모델 (자동 파라미터는 시간이 오래 걸리므로 고정값 사용)
            model = ARIMA(prices, order=(5,1,2))
            fitted_model = model.fit()
            
            # 예측
            forecast_result = fitted_model.forecast(steps=days)
            
            # 예측 결과를 DataFrame으로 변환
            last_date = data.index[-1]
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
            
            forecast_df = pd.DataFrame({
                'ds': forecast_dates,
                'yhat': forecast_result,
                'yhat_lower': forecast_result * 0.95,  # 간단한 신뢰구간
                'yhat_upper': forecast_result * 1.05
            })
            
            # 성능 평가
            if len(prices) > 30:
                train_size = len(prices) - 30
                train_data = prices[:train_size]
                test_data = prices[train_size:]
                
                model_eval = ARIMA(train_data, order=(5,1,2))
                fitted_eval = model_eval.fit()
                forecast_eval = fitted_eval.forecast(steps=len(test_data))
                
                mape = np.mean(np.abs((test_data - forecast_eval) / test_data)) * 100
                rmse = np.sqrt(np.mean((test_data - forecast_eval) ** 2))
            else:
                mape = 0.0
                rmse = 0.0
            
            metrics = {'MAPE': mape, 'RMSE': rmse}
            
            return forecast_df, metrics
            
        except Exception as e:
            st.error(f"ARIMA 예측 실패: {str(e)}")
            return pd.DataFrame(), {}
        
    @staticmethod
    def prophet_volume_forecast(data: pd.DataFrame, days: int = 30) -> Tuple[pd.DataFrame, Dict]:
        """Prophet을 사용한 거래량 예측"""
        try:
            # Prophet용 데이터 준비 (거래량 기준)
            df = data.reset_index()
            
            # 거래량이 0인 데이터 제거
            df = df[df['Volume'] > 0]
            
            if len(df) < 10:  # 데이터가 너무 적으면 예측 불가
                return pd.DataFrame(), {'error': '거래량 데이터가 부족합니다'}
            
            df = df[['Date', 'Volume']].rename(columns={'Date': 'ds', 'Volume': 'y'})
            
            # 시간대 정보 제거
            df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
            
            # 거래량은 로그 변환하여 예측 (변동성 완화)
            df['y'] = np.log(df['y'] + 1)
            
            # 모델 생성 및 학습
            model = Prophet(
                daily_seasonality=True, 
                weekly_seasonality=True, 
                yearly_seasonality=False,  # 거래량은 연간 계절성이 약함
                seasonality_mode='additive'
            )
            model.fit(df)
            
            # 예측
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            
            # 로그 변환 역변환
            forecast['yhat'] = np.exp(forecast['yhat']) - 1
            forecast['yhat_lower'] = np.exp(forecast['yhat_lower']) - 1
            forecast['yhat_upper'] = np.exp(forecast['yhat_upper']) - 1
            
            # 음수 값 제거
            forecast['yhat'] = np.maximum(forecast['yhat'], 0)
            forecast['yhat_lower'] = np.maximum(forecast['yhat_lower'], 0)
            forecast['yhat_upper'] = np.maximum(forecast['yhat_upper'], 0)
            
            # 성능 평가
            if len(df) > 15:
                train_size = len(df) - 15
                train_df = df[:train_size]
                test_df = df[train_size:]
                
                model_eval = Prophet(
                    daily_seasonality=True, 
                    weekly_seasonality=True, 
                    yearly_seasonality=False,
                    seasonality_mode='additive'
                )
                model_eval.fit(train_df)
                
                future_eval = model_eval.make_future_dataframe(periods=15)
                forecast_eval = model_eval.predict(future_eval)
                
                # 실제 값과 예측 값 비교 (로그 역변환)
                actual = np.exp(test_df['y'].values) - 1
                predicted = np.exp(forecast_eval.tail(15)['yhat'].values[:len(actual)]) - 1
                
                # MAPE 계산 (거래량이 0에 가까운 경우 처리)
                mask = actual > 1000  # 거래량이 1000 이상인 경우만 계산
                if mask.sum() > 0:
                    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
                    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                else:
                    mape = 0.0
                    rmse = 0.0
            else:
                mape = 0.0
                rmse = 0.0
            
            metrics = {'MAPE': mape, 'RMSE': rmse}
            
            return forecast, metrics
            
        except Exception as e:
            return pd.DataFrame(), {'error': f"Prophet 거래량 예측 실패: {str(e)}"}
    
    @staticmethod
    def arima_volume_forecast(data: pd.DataFrame, days: int = 30) -> Tuple[pd.DataFrame, Dict]:
        """ARIMA를 사용한 거래량 예측"""
        try:
            # 거래량이 0인 데이터 제거
            volume_data = data[data['Volume'] > 0]['Volume']
            
            if len(volume_data) < 20:
                return pd.DataFrame(), {'error': '거래량 데이터가 부족합니다'}
            
            # 로그 변환으로 안정화
            log_volume = np.log(volume_data + 1)
            
            # ARIMA 모델 (거래량은 차분이 필요할 수 있음)
            model = ARIMA(log_volume, order=(3,1,2))
            fitted_model = model.fit()
            
            # 예측
            forecast_result = fitted_model.forecast(steps=days)
            
            # 신뢰구간 계산
            forecast_ci = fitted_model.get_forecast(steps=days).conf_int()
            
            # 로그 역변환
            forecast_values = np.exp(forecast_result) - 1
            forecast_lower = np.exp(forecast_ci.iloc[:, 0]) - 1
            forecast_upper = np.exp(forecast_ci.iloc[:, 1]) - 1
            
            # 예측 결과를 DataFrame으로 변환
            last_date = data.index[-1]
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
            
            forecast_df = pd.DataFrame({
                'ds': forecast_dates,
                'yhat': forecast_values,
                'yhat_lower': forecast_lower,
                'yhat_upper': forecast_upper
            })
            
            # 성능 평가
            if len(log_volume) > 15:
                train_size = len(log_volume) - 15
                train_data = log_volume[:train_size]
                test_data = log_volume[train_size:]
                
                model_eval = ARIMA(train_data, order=(3,1,2))
                fitted_eval = model_eval.fit()
                forecast_eval = fitted_eval.forecast(steps=len(test_data))
                
                # 로그 역변환하여 비교
                actual = np.exp(test_data) - 1
                predicted = np.exp(forecast_eval) - 1
                
                # MAPE 계산
                mask = actual > 1000
                if mask.sum() > 0:
                    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
                    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                else:
                    mape = 0.0
                    rmse = 0.0
            else:
                mape = 0.0
                rmse = 0.0
            
            metrics = {'MAPE': mape, 'RMSE': rmse}
            
            return forecast_df, metrics
            
        except Exception as e:
            return pd.DataFrame(), {'error': f"ARIMA 거래량 예측 실패: {str(e)}"}


# 차트 생성 함수 추가

def create_volume_chart(data: pd.DataFrame, forecast: pd.DataFrame, symbol: str):
    """거래량 차트 생성 (선 그래프)"""
    fig = go.Figure()
    
    # 실제 거래량 (선 그래프)
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Volume'],
        mode='lines',
        name='실제 거래량',
        line=dict(color='lightblue', width=2),
        fill=None
    ))
    
    if not forecast.empty:
        # 예측 거래량 (선 그래프)
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='예측 거래량',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        # 신뢰구간 (영역 차트)
        if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='신뢰구간',
                fillcolor='rgba(255,165,0,0.2)'
            ))
    
    fig.update_layout(
        title=f'{symbol} 거래량 예측',
        xaxis_title='날짜',
        yaxis_title='거래량',
        hovermode='x unified',
        height=400,
        yaxis=dict(type='linear')  # 로그 스케일이 필요하면 'log'로 변경
    )
    
    return fig


def create_combined_chart(price_data: pd.DataFrame, volume_data: pd.DataFrame, 
                         price_forecast: pd.DataFrame, volume_forecast: pd.DataFrame, symbol: str):
    """주가와 거래량을 함께 보여주는 복합 차트"""
    from plotly.subplots import make_subplots
    
    # 2개 행의 서브플롯 생성
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} 주가 예측', f'{symbol} 거래량 예측'),
        row_width=[0.7, 0.3]
    )
    
    # 주가 차트 (첫 번째 행)
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['Close'],
            mode='lines',
            name='실제 주가',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    if not price_forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=price_forecast['ds'],
                y=price_forecast['yhat'],
                mode='lines',
                name='예측 주가',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
    
    # 거래량 차트 (두 번째 행) - 선 그래프로 변경
    fig.add_trace(
        go.Scatter(
            x=volume_data.index,
            y=volume_data['Volume'],
            mode='lines',
            name='실제 거래량',
            line=dict(color='lightblue', width=2),
            fill='tozeroy',  # 0까지 채우기
            fillcolor='rgba(173,216,230,0.3)'
        ),
        row=2, col=1
    )
    
    if not volume_forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=volume_forecast['ds'],
                y=volume_forecast['yhat'],
                mode='lines',
                name='예측 거래량',
                line=dict(color='orange', width=2, dash='dash'),
                fill='tozeroy',  # 0까지 채우기
                fillcolor='rgba(255,165,0,0.3)'
            ),
            row=2, col=1
        )
    
    # 레이아웃 업데이트
    fig.update_layout(
        title=f'{symbol} 주가 및 거래량 종합 예측',
        height=700,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="날짜", row=2, col=1)
    fig.update_yaxes(title_text="주가 ($)", row=1, col=1)
    fig.update_yaxes(title_text="거래량", row=2, col=1)
    
    return fig


# 거래량 분석 함수
def analyze_volume_trends(data: pd.DataFrame, forecast: pd.DataFrame) -> Dict:
    """거래량 트렌드 분석"""
    try:
        analysis = {}
        
        if data.empty or 'Volume' not in data.columns:
            return {'error': '거래량 데이터가 없습니다'}
        
        # 최근 거래량 통계
        recent_volume = data['Volume'][-30:]  # 최근 30일
        avg_volume_30d = recent_volume.mean()
        
        # 전체 평균과 비교
        total_avg_volume = data['Volume'].mean()
        volume_ratio = avg_volume_30d / total_avg_volume if total_avg_volume > 0 else 0
        
        # 거래량 트렌드 (최근 10일 vs 이전 10일)
        if len(data) >= 20:
            recent_10d = data['Volume'][-10:].mean()
            previous_10d = data['Volume'][-20:-10].mean()
            trend_change = ((recent_10d - previous_10d) / previous_10d * 100) if previous_10d > 0 else 0
        else:
            trend_change = 0
        
        # 거래량 변동성
        volume_std = recent_volume.std()
        volume_cv = (volume_std / avg_volume_30d * 100) if avg_volume_30d > 0 else 0
        
        # 예측 거래량 분석
        if not forecast.empty and 'yhat' in forecast.columns:
            predicted_avg = forecast['yhat'].mean()
            volume_forecast_change = ((predicted_avg - avg_volume_30d) / avg_volume_30d * 100) if avg_volume_30d > 0 else 0
        else:
            volume_forecast_change = 0
            predicted_avg = 0
        
        analysis = {
            'avg_volume_30d': avg_volume_30d,
            'total_avg_volume': total_avg_volume,
            'volume_ratio': volume_ratio,
            'trend_change': trend_change,
            'volume_cv': volume_cv,
            'volume_forecast_change': volume_forecast_change,
            'predicted_avg': predicted_avg
        }
        
        return analysis
        
    except Exception as e:
        return {'error': f'거래량 분석 실패: {str(e)}'}


class SentimentAnalyzer:
    """감성 분석 클래스"""
    
    @staticmethod
    @st.cache_data(ttl=1800)  # 30분 캐싱
    def search_web_news(query: str, num_results: int = 10) -> List[Dict]:
        """웹에서 뉴스 검색"""
        try:
            # Google 검색 (뉴스 필터)
            search_url = f"https://www.google.com/search?q={quote(query)}&tbm=nws&num={num_results}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            news_results = []
            
            # 뉴스 항목 파싱
            news_items = soup.find_all('div', class_='SoaBEf')
            
            for item in news_items[:num_results]:
                try:
                    # 제목 추출
                    title_elem = item.find('div', class_='MBeuO')
                    title = title_elem.get_text() if title_elem else "제목 없음"
                    
                    # 링크 추출
                    link_elem = item.find('a')
                    link = link_elem.get('href') if link_elem else "#"
                    
                    # 출처 추출
                    source_elem = item.find('div', class_='CEMjEf')
                    source = source_elem.get_text() if source_elem else "출처 불명"
                    
                    # 시간 추출
                    time_elem = item.find('span', class_='r0bn4c')
                    pub_time = time_elem.get_text() if time_elem else "시간 불명"
                    
                    # 요약 추출
                    snippet_elem = item.find('div', class_='GI74Re')
                    snippet = snippet_elem.get_text() if snippet_elem else ""
                    
                    news_results.append({
                        'title': title,
                        'link': link,
                        'source': source,
                        'published_time': pub_time,
                        'snippet': snippet
                    })
                    
                except Exception as e:
                    continue
            
            return news_results
            
        except Exception as e:
            st.warning(f"웹 검색 실패: {str(e)}")
            return []
    
    @staticmethod
    def analyze_with_gpt(news_data: List[Dict], symbol: str, openai_api_key: str) -> Dict:
        """GPT를 이용한 뉴스 분석"""
        try:
            if not openai_api_key:
                return {"error": "OpenAI API 키가 필요합니다."}
            
            # 최신 OpenAI 라이브러리 사용
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)
            
            # 뉴스 데이터를 텍스트로 변환
            news_text = ""
            for i, news in enumerate(news_data[:10], 1):  # 최대 10개 뉴스
                news_text += f"\n{i}. 제목: {news['title']}\n"
                news_text += f"   출처: {news['source']} ({news['published_time']})\n"
                news_text += f"   내용: {news['snippet']}\n"
            
            prompt = f"""
다음은 {symbol} 기업과 관련된 최근 뉴스들입니다. 이를 분석하여 다음 형식으로 답변해주세요:

뉴스 데이터:
{news_text}

분석 요청:
1. 전체적인 시장 감성 (매우 긍정적/긍정적/중립적/부정적/매우 부정적)
2. 주요 이슈 3가지 (각각 한 줄로 요약)
3. 투자자 관점에서의 시사점 (3-4줄)
4. 예상되는 주가 영향 (상승 요인/하락 요인)

응답은 한국어로, 투자 전문가 관점에서 객관적이고 분석적으로 작성해주세요.
"""

            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "당신은 전문 투자 분석가입니다. 뉴스를 분석하여 객관적이고 통찰력 있는 투자 분석을 제공합니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content.strip()
            
            # 감성 점수 추출 (간단한 키워드 기반)
            if "매우 긍정적" in analysis:
                sentiment_score = 0.8
                sentiment_label = "매우 긍정적"
            elif "긍정적" in analysis:
                sentiment_score = 0.4
                sentiment_label = "긍정적"
            elif "매우 부정적" in analysis:
                sentiment_score = -0.8
                sentiment_label = "매우 부정적"
            elif "부정적" in analysis:
                sentiment_score = -0.4
                sentiment_label = "부정적"
            else:
                sentiment_score = 0.0
                sentiment_label = "중립적"
            
            return {
                "analysis": analysis,
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label,
                "news_count": len(news_data)
            }
            
        except Exception as e:
            return {"error": f"GPT 분석 실패: {str(e)}"}
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_news_sentiment(symbol: str, days: int = 7) -> Dict:
        """뉴스 감성 분석 (기존 더미 데이터 방식 유지)"""
        try:
            # 기존 더미 데이터 (GPT 분석이 실패할 경우 대비)
            news_data = {
                'articles': [
                    {'title': f'{symbol} shows strong quarterly results', 'description': 'The company reported better than expected earnings.'},
                    {'title': f'{symbol} faces market challenges', 'description': 'Industry headwinds affecting performance.'},
                    {'title': f'{symbol} announces new product launch', 'description': 'Innovation driving future growth prospects.'}
                ]
            }
            
            sentiments = []
            for article in news_data['articles']:
                text = f"{article['title']} {article['description']}"
                blob = TextBlob(text)
                sentiment = blob.sentiment.polarity
                
                if sentiment > 0.1:
                    category = 'Positive'
                elif sentiment < -0.1:
                    category = 'Negative'
                else:
                    category = 'Neutral'
                
                sentiments.append({
                    'title': article['title'],
                    'sentiment_score': sentiment,
                    'sentiment_category': category
                })
            
            # 감성 집계
            positive = sum(1 for s in sentiments if s['sentiment_category'] == 'Positive')
            negative = sum(1 for s in sentiments if s['sentiment_category'] == 'Negative')
            neutral = sum(1 for s in sentiments if s['sentiment_category'] == 'Neutral')
            
            return {
                'sentiments': sentiments,
                'summary': {
                    'positive': positive,
                    'negative': negative,
                    'neutral': neutral,
                    'total': len(sentiments)
                }
            }
        except Exception as e:
            st.error(f"감성 분석 실패: {str(e)}")
            return {}
    
    @staticmethod
    def get_enhanced_news_sentiment(symbol: str, openai_api_key: str = None) -> Dict:
        """GPT 기반 향상된 뉴스 감성 분석"""
        try:
            # 웹 검색으로 실제 뉴스 수집
            search_query = f"{symbol} stock news latest"
            news_results = SentimentAnalyzer.search_web_news(search_query, num_results=15)
            
            if not news_results:
                # 웹 검색 실패시 기존 방식 사용
                return SentimentAnalyzer.get_news_sentiment(symbol)
            
            result = {
                'news_articles': news_results,
                'basic_sentiment': None,
                'gpt_analysis': None
            }
            
            # 기본 TextBlob 감성 분석
            basic_sentiments = []
            for news in news_results:
                text = f"{news['title']} {news['snippet']}"
                blob = TextBlob(text)
                sentiment = blob.sentiment.polarity
                
                if sentiment > 0.1:
                    category = 'Positive'
                elif sentiment < -0.1:
                    category = 'Negative'
                else:
                    category = 'Neutral'
                
                basic_sentiments.append({
                    'title': news['title'],
                    'sentiment_score': sentiment,
                    'sentiment_category': category,
                    'link': news['link'],
                    'source': news['source'],
                    'time': news['published_time']
                })
            
            # 기본 감성 집계
            positive = sum(1 for s in basic_sentiments if s['sentiment_category'] == 'Positive')
            negative = sum(1 for s in basic_sentiments if s['sentiment_category'] == 'Negative')
            neutral = sum(1 for s in basic_sentiments if s['sentiment_category'] == 'Neutral')
            
            result['basic_sentiment'] = {
                'sentiments': basic_sentiments,
                'summary': {
                    'positive': positive,
                    'negative': negative,
                    'neutral': neutral,
                    'total': len(basic_sentiments)
                }
            }
            
            # GPT 분석 (API 키가 있는 경우)
            if openai_api_key:
                gpt_result = SentimentAnalyzer.analyze_with_gpt(news_results, symbol, openai_api_key)
                result['gpt_analysis'] = gpt_result
            
            return result
            
        except Exception as e:
            st.error(f"향상된 감성 분석 실패: {str(e)}")
            return SentimentAnalyzer.get_news_sentiment(symbol)


class FinancialAnalyzer:
    """재무 분석 클래스"""
    
    @staticmethod
    def calculate_financial_metrics(financial_data: Dict) -> Dict:
        """재무 지표 계산"""
        try:
            info = financial_data.get('info', {})
            
            metrics = {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'roe': info.get('returnOnEquity', 0),
                'profit_margin': info.get('profitMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'free_cash_flow': info.get('freeCashflow', 0),
                'current_ratio': info.get('currentRatio', 0),
                'dividend_yield': info.get('dividendYield', 0) or 0
            }
            
            # None 값들을 0으로 변환
            for key, value in metrics.items():
                if value is None:
                    metrics[key] = 0
            
            return metrics
            
        except Exception as e:
            st.error(f"재무 지표 계산 실패: {str(e)}")
            return {}
    
    @staticmethod
    def analyze_financial_metrics_with_gpt(metrics: Dict, financial_history: Dict, symbol: str, openai_api_key: str) -> Dict:
        """GPT를 이용한 재무 지표 분석"""
        try:
            if not openai_api_key:
                return {"error": "OpenAI API 키가 필요합니다."}
            
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)
            
            # 재무 데이터를 텍스트로 변환
            financial_text = f"""
{symbol} 기업의 재무 지표 분석:

📊 현재 재무 지표:
- 시가총액: ${metrics.get('market_cap', 0)/1e9:.1f}B
- P/E 비율: {metrics.get('pe_ratio', 0):.2f}
- P/B 비율: {metrics.get('pb_ratio', 0):.2f}
- 부채비율 (D/E): {metrics.get('debt_to_equity', 0):.2f}
- ROE: {metrics.get('roe', 0)*100:.1f}%
- 순이익률: {metrics.get('profit_margin', 0)*100:.1f}%
- 매출 성장률: {metrics.get('revenue_growth', 0)*100:.1f}%
- 자유현금흐름: ${metrics.get('free_cash_flow', 0)/1e9:.1f}B
- 유동비율: {metrics.get('current_ratio', 0):.2f}
- 배당수익률: {metrics.get('dividend_yield', 0):.2f}%
"""

            # 3개년 실적 데이터 추가
            if financial_history and financial_history.get('years'):
                financial_text += f"""

📈 최근 3개년 실적 추이:
"""
                for i, year in enumerate(financial_history['years']):
                    revenue = financial_history['revenue'][i]
                    op_income = financial_history['operating_income'][i]
                    net_income = financial_history['net_income'][i]
                    
                    financial_text += f"""
- {year}년: 매출 ${revenue:.1f}B, 영업이익 ${op_income:.1f}B, 순이익 ${net_income:.1f}B"""

            prompt = f"""
다음은 {symbol} 기업의 재무 지표입니다. 전문 재무 분석가 관점에서 분석해주세요:

{financial_text}

다음 형식으로 분석해주세요:

1. **재무 건전성 종합 평가** (5점 만점으로 점수 부여)
   - 점수: X/5점
   - 한 줄 요약

2. **주요 강점** (3가지)
   - 강점 1: 설명
   - 강점 2: 설명  
   - 강점 3: 설명

3. **주요 약점 또는 주의사항** (2-3가지)
   - 약점 1: 설명
   - 약점 2: 설명

4. **업종 대비 경쟁력**
   - 업종 평균 대비 우수한 지표
   - 업종 평균 대비 부족한 지표

5. **투자자 관점 분석**
   - 장기 투자자 관점
   - 단기 투자자 관점
   - 리스크 요인

6. **재무 개선 과제**
   - 개선이 필요한 영역
   - 모니터링 필요 지표

7. **결론 및 투자 의견**
   - 매수/보유/매도 추천
   - 추천 이유 (2-3줄)

각 항목을 구체적이고 실용적으로 분석해주세요. 수치가 0이거나 없는 지표는 "데이터 없음"으로 처리하고 다른 지표로 보완 분석해주세요.
"""

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "당신은 전문 재무 분석가입니다. 기업의 재무 지표를 종합적으로 분석하여 투자자에게 실용적인 인사이트를 제공합니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content.strip()
            
            # 점수 추출 (간단한 정규식 사용)
            import re
            score_match = re.search(r'점수[:\s]*(\d+(?:\.\d+)?)[/점]', analysis)
            score = float(score_match.group(1)) if score_match else 3.0
            
            return {
                "analysis": analysis,
                "score": score,
                "max_score": 5.0,
                "success": True
            }
            
        except Exception as e:
            return {"error": f"GPT 재무 분석 실패: {str(e)}"}

    @staticmethod
    def get_financial_grade(score: float, max_score: float = 5.0) -> Dict:
        """재무 점수를 등급으로 변환"""
        percentage = (score / max_score) * 100
        
        if percentage >= 90:
            return {"grade": "A+", "color": "success", "description": "매우 우수"}
        elif percentage >= 80:
            return {"grade": "A", "color": "success", "description": "우수"}
        elif percentage >= 70:
            return {"grade": "B+", "color": "success", "description": "양호"}
        elif percentage >= 60:
            return {"grade": "B", "color": "warning", "description": "보통"}
        elif percentage >= 50:
            return {"grade": "C+", "color": "warning", "description": "미흡"}
        elif percentage >= 40:
            return {"grade": "C", "color": "error", "description": "부족"}
        else:
            return {"grade": "D", "color": "error", "description": "매우 부족"}


class EconomicIndicatorAnalyzer:
    """대외 경제지표 분석 클래스"""
    
    @staticmethod
    def interpret_vix(vix_value: float) -> Dict:
        """VIX 수치 해석"""
        if vix_value >= 50:
            return {
                'level': '극도 공포',
                'color': 'error',
                'emoji': '🚨',
                'message': '패닉 상태 - 역설적 매수 기회 가능',
                'advice': '극도의 공포로 인한 과매도 상황. 장기 투자자에게는 기회가 될 수 있음'
            }
        elif vix_value >= 30:
            return {
                'level': '높은 공포',
                'color': 'error',
                'emoji': '😰',
                'message': '시장 불안정 - 주의 깊은 매수 검토',
                'advice': '변동성이 높은 상태. 분할 매수나 방어적 투자 전략 고려'
            }
        elif vix_value >= 20:
            return {
                'level': '보통 긴장',
                'color': 'warning',
                'emoji': '⚠️',
                'message': '정상적인 변동성 수준',
                'advice': '일반적인 시장 조정 범위. 정상적인 투자 전략 유지'
            }
        elif vix_value >= 12:
            return {
                'level': '안정',
                'color': 'success',
                'emoji': '😌',
                'message': '시장 안정 - 정상적인 투자 환경',
                'advice': '안정적인 시장 상황. 성장주 투자에 적합한 환경'
            }
        else:
            return {
                'level': '극도 안정',
                'color': 'warning',
                'emoji': '😴',
                'message': '과도한 낙관론 - 주의 필요',
                'advice': '너무 안일한 상황. 시장 과열 가능성을 염두에 두고 신중한 투자 필요'
            }
    
    @staticmethod
    @st.cache_data(ttl=1800)  # 30분 캐싱
    def get_economic_indicators() -> Dict:
        """주요 경제지표 데이터 가져오기"""
        try:
            indicators = {}
            
            # 주요 경제지표 심볼들
            symbols = {
                'gold': 'GC=F',  # 금 선물
                'dollar_index': 'DX-Y.NYB',  # 달러 인덱스
                'sp500': '^GSPC',  # S&P 500
                'nasdaq': '^IXIC',  # 나스닥
                'dow': '^DJI',  # 다우존스
                'us_10yr': '^TNX',  # 미국 10년 국채
                'us_2yr': '^IRX',  # 미국 2년 국채
                'vix': '^VIX',  # 변동성 지수
                'oil': 'CL=F',  # 원유 선물
                'btc': 'BTC-USD'  # 비트코인
            }
            
            for name, symbol in symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1y")
                    info = ticker.info
                    
                    if not data.empty:
                        current_price = data['Close'].iloc[-1]
                        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                        change_pct = ((current_price - prev_price) / prev_price) * 100
                        
                        indicators[name] = {
                            'symbol': symbol,
                            'data': data,
                            'current_price': current_price,
                            'change_pct': change_pct,
                            'name': info.get('longName', name.replace('_', ' ').title())
                        }
                except Exception as e:
                    st.warning(f"{name} 데이터 조회 실패: {str(e)}")
                    continue
                    
            return indicators
            
        except Exception as e:
            st.error(f"경제지표 조회 실패: {str(e)}")
            return {}
    
    @staticmethod
    def analyze_economic_indicators_with_gpt(indicators: Dict, openai_api_key: str) -> Dict:
        """GPT를 이용한 종합 경제지표 분석"""
        try:
            if not openai_api_key:
                return {"error": "OpenAI API 키가 필요합니다."}
            
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)
            
            # 경제지표 데이터를 텍스트로 변환
            economic_text = "📊 현재 주요 경제지표 현황:\n\n"
            
            # 주식 지수
            if 'sp500' in indicators:
                sp500 = indicators['sp500']
                economic_text += f"• S&P 500: {sp500['current_price']:.2f} ({sp500['change_pct']:+.2f}%)\n"
            
            if 'nasdaq' in indicators:
                nasdaq = indicators['nasdaq']
                economic_text += f"• 나스닥: {nasdaq['current_price']:.2f} ({nasdaq['change_pct']:+.2f}%)\n"
            
            if 'dow' in indicators:
                dow = indicators['dow']
                economic_text += f"• 다우존스: {dow['current_price']:.2f} ({dow['change_pct']:+.2f}%)\n"
            
            economic_text += "\n"
            
            # 채권 수익률
            if 'us_10yr' in indicators:
                us_10yr = indicators['us_10yr']
                economic_text += f"• 미국 10년 국채 수익률: {us_10yr['current_price']:.3f}% ({us_10yr['change_pct']:+.2f}%)\n"
            
            if 'us_2yr' in indicators:
                us_2yr = indicators['us_2yr']
                economic_text += f"• 미국 2년 국채 수익률: {us_2yr['current_price']:.3f}% ({us_2yr['change_pct']:+.2f}%)\n"
            
            economic_text += "\n"
            
            # 원자재
            if 'gold' in indicators:
                gold = indicators['gold']
                economic_text += f"• 금 가격: ${gold['current_price']:.2f} ({gold['change_pct']:+.2f}%)\n"
            
            if 'oil' in indicators:
                oil = indicators['oil']
                economic_text += f"• 원유 가격: ${oil['current_price']:.2f} ({oil['change_pct']:+.2f}%)\n"
            
            economic_text += "\n"
            
            # 기타 지표
            if 'dollar_index' in indicators:
                dollar = indicators['dollar_index']
                economic_text += f"• 달러 인덱스: {dollar['current_price']:.2f} ({dollar['change_pct']:+.2f}%)\n"
            
            if 'vix' in indicators:
                vix = indicators['vix']
                economic_text += f"• VIX 공포지수: {vix['current_price']:.2f} ({vix['change_pct']:+.2f}%)\n"
            
            if 'btc' in indicators:
                btc = indicators['btc']
                economic_text += f"• 비트코인: ${btc['current_price']:,.0f} ({btc['change_pct']:+.2f}%)\n"

            prompt = f"""
    다음은 현재 주요 경제지표들의 실시간 데이터입니다. 전문 경제 분석가 관점에서 종합 분석해주세요:

    {economic_text}

    다음 형식으로 분석해주세요:

    ## 🌍 대외 경제환경 종합 분석

    ### 1. **현재 경제 상황 진단** (5점 만점으로 점수 부여)
    - 종합 점수: X/5점
    - 한 줄 요약

    ### 2. **주식시장 분석**
    - 주요 지수 동향 및 의미
    - 시장 모멘텀 평가
    - 섹터별 영향 전망

    ### 3. **금리 환경 분석**
    - 수익률 곡선 상태 및 의미
    - 연준 정책 방향성 추론
    - 장단기 금리 스프레드 분석

    ### 4. **인플레이션 및 원자재**
    - 금, 원유 가격 동향 분석
    - 인플레이션 압력 진단
    - 원자재 순환 사이클 위치

    ### 5. **달러 및 외환 환경**
    - 달러 강세/약세 배경
    - 글로벌 유동성 상황
    - 신흥국 영향 평가

    ### 6. **리스크 지표 분석**
    - VIX를 통한 시장 불안감 측정
    - 안전자산 선호도 변화
    - 시스템 리스크 수준 평가

    ### 7. **투자자를 위한 시사점**
    - 현재 상황에서의 자산배분 방향
    - 주의해야 할 리스크 요인
    - 기회 요인 및 투자 테마

    ### 8. **향후 전망** (1-3개월)
    - 예상되는 시장 시나리오
    - 주요 변곡점 및 모니터링 지표
    - 정책 변화 가능성

    ### 9. **결론 및 투자 가이드**
    - 현재 경제환경 한 줄 요약
    - Risk-On / Risk-Off 중 어느 상황인지
    - 추천 투자 전략 (공격적/중립적/방어적)

    각 섹션을 구체적이고 실용적으로 분석해주시고, 투자자가 실제로 활용할 수 있는 인사이트를 제공해주세요.
    """

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "당신은 전문 거시경제 분석가입니다. 다양한 경제지표를 종합하여 현재 경제환경을 정확히 진단하고, 투자자에게 실용적인 가이드를 제공합니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content.strip()
            
            # 점수 추출
            import re
            score_match = re.search(r'점수[:\s]*(\d+(?:\.\d+)?)[/점]', analysis)
            score = float(score_match.group(1)) if score_match else 3.0
            
            # 투자 환경 분류
            if score >= 4.0:
                environment = "Risk-On (위험자산 선호)"
                environment_color = "success"
            elif score >= 3.0:
                environment = "중립적 (혼재된 신호)"
                environment_color = "warning"
            else:
                environment = "Risk-Off (안전자산 선호)"
                environment_color = "error"
            
            return {
                "analysis": analysis,
                "score": score,
                "max_score": 5.0,
                "environment": environment,
                "environment_color": environment_color,
                "success": True
            }
            
        except Exception as e:
            return {"error": f"GPT 경제분석 실패: {str(e)}"}
    
    @staticmethod
    def get_financial_history(symbol: str) -> Dict:
        """3개년 재무 실적 데이터 가져오기"""
        try:
            stock = yf.Ticker(symbol)
            financials = stock.financials
            
            if financials.empty:
                return {}
            
            # 최근 3년 데이터 추출
            years = financials.columns[:3]  # 최신 3개년
            
            financial_history = {
                'years': [year.strftime('%Y') for year in years],
                'revenue': [],
                'operating_income': [],
                'net_income': []
            }
            
            for year in years:
                # 매출 (Total Revenue)
                revenue = financials.loc['Total Revenue', year] if 'Total Revenue' in financials.index else 0
                
                # 영업이익 (Operating Income)
                operating_income = financials.loc['Operating Income', year] if 'Operating Income' in financials.index else 0
                
                # 순이익 (Net Income)
                net_income = financials.loc['Net Income', year] if 'Net Income' in financials.index else 0
                
                financial_history['revenue'].append(revenue / 1e9 if revenue else 0)  # 10억 단위
                financial_history['operating_income'].append(operating_income / 1e9 if operating_income else 0)
                financial_history['net_income'].append(net_income / 1e9 if net_income else 0)
            
            return financial_history
            
        except Exception as e:
            st.error(f"재무 실적 데이터 조회 실패: {str(e)}")
            return {}
    

class BuffettAnalyzer:
    """버핏 스타일 투자 분석 클래스"""
    
    @staticmethod
    def buffett_analysis(financial_metrics: Dict, sentiment_data: Dict) -> Dict:
        """버핏 스타일 투자 분석"""
        try:
            score = 0
            max_score = 100
            reasons = []
            
            # ROE 평가 (25점)
            roe = financial_metrics.get('roe', 0) * 100 if financial_metrics.get('roe') else 0
            if roe > 15:
                score += 25
                reasons.append(f"✅ 우수한 ROE: {roe:.1f}% (기준: 15% 이상)")
            elif roe > 10:
                score += 15
                reasons.append(f"⚠️ 적정 ROE: {roe:.1f}% (기준: 15% 이상)")
            else:
                reasons.append(f"❌ 낮은 ROE: {roe:.1f}% (기준: 15% 이상)")
            
            # P/E 비율 평가 (20점)
            pe_ratio = financial_metrics.get('pe_ratio', 0)
            if 10 <= pe_ratio <= 20:
                score += 20
                reasons.append(f"✅ 적정 P/E 비율: {pe_ratio:.1f} (선호: 10-20)")
            elif pe_ratio > 0:
                score += 10
                reasons.append(f"⚠️ P/E 비율: {pe_ratio:.1f} (선호: 10-20)")
            else:
                reasons.append("❌ P/E 비율 데이터 없음")
            
            # 부채비율 평가 (20점)
            debt_to_equity = financial_metrics.get('debt_to_equity', 0)
            if debt_to_equity < 0.3:
                score += 20
                reasons.append(f"✅ 낮은 부채비율: {debt_to_equity:.2f} (선호: 0.3 미만)")
            elif debt_to_equity < 0.5:
                score += 10
                reasons.append(f"⚠️ 부채비율: {debt_to_equity:.2f} (선호: 0.3 미만)")
            else:
                reasons.append(f"❌ 높은 부채비율: {debt_to_equity:.2f} (선호: 0.3 미만)")
            
            # 자유현금흐름 평가 (15점)
            fcf = financial_metrics.get('free_cash_flow', 0)
            if fcf > 0:
                score += 15
                reasons.append(f"✅ 양의 자유현금흐름: ${fcf:,.0f}")
            else:
                reasons.append("❌ 음의 자유현금흐름")
            
            # 수익성 평가 (10점)
            profit_margin = financial_metrics.get('profit_margin', 0) * 100 if financial_metrics.get('profit_margin') else 0
            if profit_margin > 10:
                score += 10
                reasons.append(f"✅ 우수한 수익성: {profit_margin:.1f}% (선호: 10% 이상)")
            elif profit_margin > 5:
                score += 5
                reasons.append(f"⚠️ 적정 수익성: {profit_margin:.1f}% (선호: 10% 이상)")
            else:
                reasons.append(f"❌ 낮은 수익성: {profit_margin:.1f}% (선호: 10% 이상)")
            
            # 감성 분석 반영 (10점)
            if sentiment_data:
                sentiment_summary = sentiment_data.get('summary', {})
                positive_ratio = sentiment_summary.get('positive', 0) / max(sentiment_summary.get('total', 1), 1)
                if positive_ratio > 0.6:
                    score += 10
                    reasons.append(f"✅ 긍정적 시장 감성: {positive_ratio:.1%}")
                elif positive_ratio > 0.4:
                    score += 5
                    reasons.append(f"⚠️ 중립적 시장 감성: {positive_ratio:.1%}")
                else:
                    reasons.append(f"❌ 부정적 시장 감성: {positive_ratio:.1%}")
            
            # 투자 등급 결정
            if score >= 80:
                grade = "BUY"
                grade_color = "buy-signal"
                recommendation = "🚀 강력 매수 추천: 버핏의 투자 기준을 대부분 충족합니다."
            elif score >= 60:
                grade = "HOLD"
                grade_color = "hold-signal"
                recommendation = "📊 보유 권장: 일부 기준을 충족하나 신중한 검토가 필요합니다."
            else:
                grade = "SELL"
                grade_color = "sell-signal"
                recommendation = "⚠️ 투자 비추천: 버핏의 투자 기준에 미달합니다."
            
            return {
                'score': score,
                'max_score': max_score,
                'grade': grade,
                'grade_color': grade_color,
                'recommendation': recommendation,
                'reasons': reasons
            }
            
        except Exception as e:
            st.error(f"버핏 분석 실패: {str(e)}")
            return {}

class YouTubeAnalyzer:
    """유튜브 채널 검색 및 영상 요약 클래스 (향상된 버전)"""
    
    @staticmethod
    @st.cache_data(ttl=3600)  # 1시간 캐싱
    def search_youtube_videos(query: str, max_results: int = 50, sort_order: str = 'relevance') -> List[Dict]:
        """유튜브 영상 검색 (향상된 버전)"""
        try:
            # 정렬 옵션에 따른 URL 파라미터 설정
            sort_params = {
                'relevance': '',  # 기본값
                'upload_date': '&sp=CAI%253D',  # 최신순
                'view_count': '&sp=CAM%253D',   # 조회수순  
                'rating': '&sp=CAE%253D'        # 평점순 (좋아요 기준)
            }
            
            sort_param = sort_params.get(sort_order, '')
            search_url = f"https://www.youtube.com/results?search_query={quote(query)}{sort_param}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            video_results = []
            
            # JavaScript로 렌더링된 데이터 추출
            pattern = r'var ytInitialData = ({.*?});'
            match = re.search(pattern, response.text)
            
            if match:
                try:
                    data = json.loads(match.group(1))
                    
                    # 검색 결과에서 비디오 정보 추출
                    contents = data.get('contents', {}).get('twoColumnSearchResultsRenderer', {}).get('primaryContents', {}).get('sectionListRenderer', {}).get('contents', [])
                    
                    for content in contents:
                        items = content.get('itemSectionRenderer', {}).get('contents', [])
                        
                        for item in items:
                            if 'videoRenderer' in item:
                                video = item['videoRenderer']
                                
                                # 비디오 정보 추출
                                video_id = video.get('videoId', '')
                                title = video.get('title', {}).get('runs', [{}])[0].get('text', '')
                                
                                # 채널 정보
                                channel_info = video.get('ownerText', {}).get('runs', [{}])[0]
                                channel_name = channel_info.get('text', '')
                                
                                # 조회수 정보 및 숫자 추출
                                view_count_text = ''
                                view_count_num = 0
                                if 'viewCountText' in video:
                                    view_count_text = video['viewCountText'].get('simpleText', '')
                                    # 조회수 숫자 추출 (예: "1.2M views" -> 1200000)
                                    view_count_num = YouTubeAnalyzer._parse_view_count(view_count_text)
                                
                                # 업로드 시간 및 파싱
                                published_text = ''
                                published_days_ago = 0
                                if 'publishedTimeText' in video:
                                    published_text = video['publishedTimeText'].get('simpleText', '')
                                    published_days_ago = YouTubeAnalyzer._parse_published_time(published_text)
                                
                                # 재생 시간 및 초 단위 변환
                                duration = ''
                                duration_seconds = 0
                                if 'lengthText' in video:
                                    duration = video['lengthText'].get('simpleText', '')
                                    duration_seconds = YouTubeAnalyzer._parse_duration(duration)
                                
                                # 썸네일 URL
                                thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
                                
                                # 좋아요/싫어요 정보 (YouTube API 제한으로 정확하지 않을 수 있음)
                                likes_count = 0  # 실제로는 추출하기 어려움
                                
                                video_results.append({
                                    'video_id': video_id,
                                    'title': title,
                                    'channel_name': channel_name,
                                    'view_count': view_count_text,
                                    'view_count_num': view_count_num,  # 필터링용 숫자값
                                    'published_time': published_text,
                                    'published_days_ago': published_days_ago,  # 필터링용 숫자값
                                    'duration': duration,
                                    'duration_seconds': duration_seconds,  # 필터링용 숫자값
                                    'likes_count': likes_count,  # 추후 구현 가능
                                    'thumbnail_url': thumbnail_url,
                                    'video_url': f"https://www.youtube.com/watch?v={video_id}"
                                })
                                
                                if len(video_results) >= max_results:
                                    break
                        
                        if len(video_results) >= max_results:
                            break
                    
                except json.JSONDecodeError:
                    pass
            
            # 결과가 부족하면 더미 데이터로 보완 (데모용)
            if len(video_results) < 10:
                demo_data = YouTubeAnalyzer._generate_demo_videos(query, max_results - len(video_results))
                video_results.extend(demo_data)
            
            return video_results[:max_results]
            
        except Exception as e:
            st.error(f"유튜브 검색 실패: {str(e)}")
            # 전체 실패시 더미 데이터 반환
            return YouTubeAnalyzer._generate_demo_videos(query, min(max_results, 20))
    
    @staticmethod
    def _parse_view_count(view_text: str) -> int:
        """조회수 텍스트를 숫자로 변환"""
        try:
            if not view_text:
                return 0
            
            # "1.2M views" -> 1200000
            import re
            numbers = re.findall(r'[\d.]+', view_text.lower())
            if not numbers:
                return 0
            
            num_str = numbers[0]
            multiplier = 1
            
            if 'k' in view_text.lower():
                multiplier = 1000
            elif 'm' in view_text.lower():
                multiplier = 1000000
            elif 'b' in view_text.lower():
                multiplier = 1000000000
            
            return int(float(num_str) * multiplier)
        except:
            return 0
    
    @staticmethod
    def _parse_published_time(time_text: str) -> int:
        """업로드 시간을 일 단위로 변환"""
        try:
            if not time_text:
                return 999999  # 매우 오래된 것으로 처리
            
            import re
            
            # "2 days ago", "1 week ago", "3 months ago" 등 파싱
            if 'hour' in time_text or '시간' in time_text:
                hours = re.findall(r'\d+', time_text)
                return float(hours[0]) / 24 if hours else 0
            elif 'day' in time_text or '일' in time_text:
                days = re.findall(r'\d+', time_text)
                return int(days[0]) if days else 1
            elif 'week' in time_text or '주' in time_text:
                weeks = re.findall(r'\d+', time_text)
                return int(weeks[0]) * 7 if weeks else 7
            elif 'month' in time_text or '개월' in time_text or '달' in time_text:
                months = re.findall(r'\d+', time_text)
                return int(months[0]) * 30 if months else 30
            elif 'year' in time_text or '년' in time_text:
                years = re.findall(r'\d+', time_text)
                return int(years[0]) * 365 if years else 365
            else:
                return 1  # 기본값
        except:
            return 999999
    
    @staticmethod
    def _parse_duration(duration_text: str) -> int:
        """재생시간을 초 단위로 변환"""
        try:
            if not duration_text:
                return 0
            
            # "12:34" -> 754초, "1:23:45" -> 5025초
            parts = duration_text.split(':')
            total_seconds = 0
            
            if len(parts) == 2:  # MM:SS
                total_seconds = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:  # HH:MM:SS
                total_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                return 0
            
            return total_seconds
        except:
            return 0
    
    @staticmethod
    def _generate_demo_videos(query: str, count: int) -> List[Dict]:
        """데모용 영상 데이터 생성"""
        import random
        
        demo_videos = []
        base_channels = [
            "투자왕", "주식연구소", "경제분석가", "재테크TV", "투자의신",
            "Financial Wisdom", "Stock Analysis Pro", "Investment Guru", "Market Insider", "Trading Master"
        ]
        
        for i in range(count):
            random_views = random.randint(1000, 5000000)
            random_days = random.randint(0, 365)
            random_duration = random.randint(300, 3600)  # 5분~1시간
            
            demo_videos.append({
                'video_id': f'demo_{i}_{hash(query) % 10000}',
                'title': f'{query} 관련 투자 분석 영상 {i+1}',
                'channel_name': random.choice(base_channels),
                'view_count': f'{random_views:,}회 조회',
                'view_count_num': random_views,
                'published_time': f'{random_days}일 전',
                'published_days_ago': random_days,
                'duration': f'{random_duration//60}:{random_duration%60:02d}',
                'duration_seconds': random_duration,
                'likes_count': random.randint(10, random_views//100),
                'thumbnail_url': 'https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg',
                'video_url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
            })
        
        return demo_videos
    
    @staticmethod
    def filter_videos(videos: List[Dict], 
                     min_views: int = 0, 
                     max_days_ago: int = 999999,
                     min_duration: int = 0,
                     max_duration: int = 999999,
                     sort_by: str = 'relevance') -> List[Dict]:
        """영상 필터링 및 정렬"""
        try:
            # 필터링
            filtered = []
            for video in videos:
                if (video['view_count_num'] >= min_views and
                    video['published_days_ago'] <= max_days_ago and
                    min_duration <= video['duration_seconds'] <= max_duration):
                    filtered.append(video)
            
            # 정렬
            if sort_by == 'view_count':
                filtered.sort(key=lambda x: x['view_count_num'], reverse=True)
            elif sort_by == 'upload_date':
                filtered.sort(key=lambda x: x['published_days_ago'])
            elif sort_by == 'duration':
                filtered.sort(key=lambda x: x['duration_seconds'], reverse=True)
            elif sort_by == 'likes':
                filtered.sort(key=lambda x: x['likes_count'], reverse=True)
            # 'relevance'는 기본 순서 유지
            
            return filtered
        except Exception as e:
            st.error(f"필터링 중 오류: {str(e)}")
            return videos

    # 기존 메서드들은 그대로 유지
    @staticmethod
    def get_video_transcript(video_id: str) -> str:
        """유튜브 영상 자막 추출 (실제 구현)"""
        try:
            # youtube-transcript-api 설치 확인 및 실제 자막 추출
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                from youtube_transcript_api.formatters import TextFormatter
                
                # 자막 언어 우선순위: 한국어 → 영어 → 자동생성 자막
                languages_to_try = [
                    ['ko'],           # 한국어
                    ['en'],           # 영어
                    ['ko', 'en'],     # 한국어 또는 영어
                    None              # 자동 생성 자막 포함
                ]
                
                transcript_text = None
                
                for languages in languages_to_try:
                    try:
                        if languages is None:
                            # 자동 생성 자막 포함해서 시도
                            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                            
                            # 수동 자막 우선 시도
                            for transcript in transcript_list:
                                if not transcript.is_generated:
                                    transcript_data = transcript.fetch()
                                    break
                            else:
                                # 수동 자막이 없으면 자동 생성 자막 사용
                                for transcript in transcript_list:
                                    if transcript.is_generated:
                                        transcript_data = transcript.fetch()
                                        break
                                else:
                                    continue
                        else:
                            # 특정 언어로 시도
                            transcript_data = YouTubeTranscriptApi.get_transcript(
                                video_id, 
                                languages=languages
                            )
                        
                        # 텍스트 포맷터로 정리
                        formatter = TextFormatter()
                        transcript_text = formatter.format_transcript(transcript_data)
                        
                        # 성공하면 루프 종료
                        break
                        
                    except Exception:
                        # 현재 언어/방법으로 실패하면 다음 시도
                        continue
                
                if transcript_text:
                    # 자막 텍스트 정리
                    cleaned_text = transcript_text.strip()
                    
                    # 너무 짧은 자막은 유효하지 않다고 판단
                    if len(cleaned_text) < 50:
                        raise Exception("자막이 너무 짧습니다")
                    
                    # 자막 품질 정보 추가
                    quality_info = ""
                    if len(cleaned_text) > 5000:
                        quality_info = "[고품질 자막] "
                    elif len(cleaned_text) > 1000:
                        quality_info = "[표준 자막] "
                    else:
                        quality_info = "[간단한 자막] "
                    
                    return quality_info + cleaned_text
                else:
                    raise Exception("모든 언어에서 자막 추출 실패")
                    
            except ImportError:
                # youtube-transcript-api가 설치되지 않은 경우
                return f"""
[라이브러리 미설치] youtube-transcript-api가 설치되지 않았습니다.

실제 자막을 가져오려면 다음 명령어로 설치하세요:
pip install youtube-transcript-api

현재는 데모 데이터로 분석됩니다:

이 영상에서는 {video_id}에 대한 투자 분석을 다룹니다. 

주요 내용:
1. 현재 시장 상황 및 트렌드 분석
2. 기업의 재무 성과 및 건전성 검토  
3. 향후 성장 전망 및 잠재적 리스크 요인
4. 포트폴리오 구성 및 투자 전략 제안

전문가 의견:
- 장기적 관점에서 긍정적인 성장 전망
- 단기적으로는 시장 변동성에 주의 필요
- 분산 투자를 통한 리스크 관리 권장
- 정기적인 포트폴리오 리뷰 및 조정 필요

투자 권고사항:
- 개인의 투자 성향과 목표를 고려한 신중한 판단
- 충분한 자료 조사 및 전문가 상담 권장
- 투자 원금 손실 가능성에 대한 인지 필요

결론:
체계적인 분석과 신중한 접근을 통해 현명한 투자 결정을 내리시기 바랍니다.
"""
                
        except Exception as e:
            # 자막 추출 완전 실패 시 더미 데이터
            error_msg = str(e)
            
            # 일반적인 오류 메시지들에 대한 사용자 친화적 설명
            if "TranscriptsDisabled" in error_msg:
                reason = "이 영상은 자막이 비활성화되어 있습니다."
            elif "NoTranscriptFound" in error_msg:
                reason = "이 영상에는 자막이 제공되지 않습니다."
            elif "VideoUnavailable" in error_msg:
                reason = "영상에 접근할 수 없습니다. (비공개 또는 삭제됨)"
            elif "TooManyRequests" in error_msg:
                reason = "YouTube API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
            else:
                reason = f"자막 추출 중 오류 발생: {error_msg}"
            
            return f"""
[자막 추출 실패] {reason}

대신 영상 정보 기반 분석을 진행합니다:

영상 ID: {video_id}

분석 방향성:
1. 영상 제목과 채널 정보를 바탕으로 한 내용 추정
2. 유사한 투자 분석 영상들의 일반적인 패턴 분석
3. 현재 시장 상황을 고려한 투자 시사점 도출

주의사항:
- 실제 영상 내용과 다를 수 있음
- 정확한 분석을 위해서는 직접 영상 시청 권장
- 투자 결정은 여러 소스를 종합하여 판단 필요

권장사항:
영상 링크를 통해 직접 시청하시거나, 자막이 제공되는 다른 영상을 선택해보세요.
"""
    
    @staticmethod
    def summarize_video_with_gpt(transcript: str, video_title: str, openai_api_key: str) -> Dict:
        """GPT를 이용한 영상 요약"""
        try:
            if not openai_api_key:
                return {"error": "OpenAI API 키가 필요합니다."}
            
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)
            
            prompt = f"""
다음은 "{video_title}" 유튜브 영상의 전체 대화 내용입니다. 이를 투자자 관점에서 요약해주세요.

영상 내용:
{transcript[:3000]}  # 토큰 제한을 위해 앞부분만 사용

다음 형식으로 요약해주세요:

1. **핵심 내용 요약** (3-4줄)
2. **주요 투자 포인트** (3개)
3. **언급된 리스크** (있다면)
4. **투자자를 위한 핵심 시사점** (2-3줄)
5. **추천 여부** (추천/보류/비추천 중 하나와 이유)

투자 전문가 관점에서 객관적이고 실용적으로 분석해주세요.
"""

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "당신은 전문 투자 분석가입니다. 유튜브 영상 내용을 투자자 관점에서 요약하고 분석합니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            
            return {
                "summary": summary,
                "success": True
            }
            
        except Exception as e:
            return {"error": f"GPT 요약 실패: {str(e)}"}

    
def create_stock_chart(data: pd.DataFrame, forecast: pd.DataFrame, symbol: str):
    """주가 차트 생성"""
    fig = go.Figure()
    
    # 실제 주가
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='실제 주가',
        line=dict(color='blue', width=2)
    ))
    
    if not forecast.empty:
        # 예측 주가
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='예측 주가',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # 신뢰구간
        if 'yhat_lower' in forecast.columns:
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='신뢰구간',
                fillcolor='rgba(255,0,0,0.2)'
            ))
    
    fig.update_layout(
        title=f'{symbol} 주가 예측',
        xaxis_title='날짜',
        yaxis_title='주가 ($)',
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_sentiment_chart(sentiment_data: Dict):
    """감성 분석 차트 생성"""
    if not sentiment_data:
        return go.Figure()
    
    summary = sentiment_data.get('summary', {})
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Positive', 'Neutral', 'Negative'],
            y=[summary.get('positive', 0), summary.get('neutral', 0), summary.get('negative', 0)],
            marker_color=['green', 'gray', 'red']
        )
    ])
    
    fig.update_layout(
        title='뉴스 감성 분석',
        xaxis_title='감성',
        yaxis_title='기사 수',
        height=400
    )
    
    return fig


def create_financial_metrics_chart(metrics: Dict):
    """재무 지표 차트 생성"""
    if not metrics:
        return go.Figure()
    
    # 주요 지표만 선택
    key_metrics = {
        'ROE (%)': metrics.get('roe', 0) * 100,
        'P/E Ratio': metrics.get('pe_ratio', 0),
        'Debt/Equity': metrics.get('debt_to_equity', 0),
        'Profit Margin (%)': metrics.get('profit_margin', 0) * 100,
        'Current Ratio': metrics.get('current_ratio', 0)
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(key_metrics.keys()),
            y=list(key_metrics.values()),
            marker_color='lightblue'
        )
    ])
    
    fig.update_layout(
        title='주요 재무 지표',
        xaxis_title='지표',
        yaxis_title='값',
        height=400
    )
    
    return fig


def create_financial_history_chart(financial_history: Dict):
    """3개년 재무 실적 차트 생성"""
    if not financial_history or not financial_history.get('years'):
        return go.Figure()
    
    years = financial_history['years']
    
    fig = go.Figure()
    
    # 매출
    fig.add_trace(go.Bar(
        name='매출',
        x=years,
        y=financial_history['revenue'],
        marker_color='lightblue'
    ))
    
    # 영업이익
    fig.add_trace(go.Bar(
        name='영업이익',
        x=years,
        y=financial_history['operating_income'],
        marker_color='lightgreen'
    ))
    
    # 순이익
    fig.add_trace(go.Bar(
        name='순이익',
        x=years,
        y=financial_history['net_income'],
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title='최근 3개년 재무 실적 (단위: 10억 달러)',
        xaxis_title='연도',
        yaxis_title='금액 (Billions $)',
        barmode='group',
        height=400
    )
    
    return fig


def create_economic_indicators_dashboard(indicators: Dict):
    """경제지표 대시보드 생성"""
    if not indicators:
        return {}
    
    charts = {}
    
    # 주요 지수들 차트
    major_indices = ['sp500', 'nasdaq', 'dow']
    if any(idx in indicators for idx in major_indices):
        fig_indices = go.Figure()
        
        for idx in major_indices:
            if idx in indicators:
                data = indicators[idx]['data']
                name = indicators[idx]['name']
                
                # 정규화 (첫 번째 값을 100으로)
                normalized_data = (data['Close'] / data['Close'].iloc[0]) * 100
                
                fig_indices.add_trace(go.Scatter(
                    x=data.index,
                    y=normalized_data,
                    mode='lines',
                    name=name,
                    line=dict(width=2)
                ))
        
        fig_indices.update_layout(
            title='주요 주식 지수 추이 (정규화, 기준점=100)',
            xaxis_title='날짜',
            yaxis_title='정규화된 지수',
            height=400
        )
        
        charts['indices'] = fig_indices
    
    # 금리 차트
    bond_yields = ['us_10yr', 'us_2yr']
    if any(bond in indicators for bond in bond_yields):
        fig_bonds = go.Figure()
        
        for bond in bond_yields:
            if bond in indicators:
                data = indicators[bond]['data']
                name = '10년 국채' if bond == 'us_10yr' else '2년 국채'
                
                fig_bonds.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name=name,
                    line=dict(width=2)
                ))
        
        fig_bonds.update_layout(
            title='미국 국채 수익률 추이',
            xaxis_title='날짜',
            yaxis_title='수익률 (%)',
            height=400
        )
        
        charts['bonds'] = fig_bonds
    
    # 원자재 차트
    commodities = ['gold', 'oil']
    if any(comm in indicators for comm in commodities):
        fig_commodities = make_subplots(
            rows=1, cols=2,
            subplot_titles=('금 가격', '원유 가격'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        if 'gold' in indicators:
            gold_data = indicators['gold']['data']
            fig_commodities.add_trace(
                go.Scatter(x=gold_data.index, y=gold_data['Close'],
                          mode='lines', name='금', line=dict(color='gold')),
                row=1, col=1
            )
        
        if 'oil' in indicators:
            oil_data = indicators['oil']['data']
            fig_commodities.add_trace(
                go.Scatter(x=oil_data.index, y=oil_data['Close'],
                          mode='lines', name='원유', line=dict(color='black')),
                row=1, col=2
            )
        
        fig_commodities.update_layout(height=400, title_text="원자재 가격 추이")
        charts['commodities'] = fig_commodities
    
    return charts

class GoogleTrendsAnalyzer:
    """구글 트렌드 분석 클래스"""

    @staticmethod
    def _safe_create_pytrends():
        """안전한 pytrends 객체 생성"""
        try:
            # 기본 설정으로 시도
            pytrends = TrendReq()
            return pytrends, None
        except Exception as e1:
            try:
                # 최소 설정으로 시도
                pytrends = TrendReq(hl='en-US', tz=360)
                return pytrends, None
            except Exception as e2:
                try:
                    # 더 간단한 설정으로 시도
                    pytrends = TrendReq(hl='en', tz=0)
                    return pytrends, None
                except Exception as e3:
                    return None, f"모든 초기화 방법 실패: {str(e3)}"
    
    @staticmethod
    @st.cache_data(ttl=1800)  # 30분 캐싱 (더 짧게)
    def get_trends_data(keywords: List[str], timeframe: str = '12-m', geo: str = '') -> Dict:
        """구글 트렌드 데이터 가져오기 - 강력한 오류 처리"""
        
        # 입력 검증
        if not keywords:
            return {"error": "검색할 키워드가 없습니다."}
        
        # 키워드 정리
        clean_keywords = []
        for k in keywords:
            if k and isinstance(k, str) and k.strip():
                clean_keywords.append(k.strip())
        
        if not clean_keywords:
            return {"error": "유효한 키워드가 없습니다."}
        
        clean_keywords = clean_keywords[:3]  # 최대 3개로 제한 (안정성)
        
        st.info(f"🔍 검색 키워드: {', '.join(clean_keywords)}")
        st.info(f"📅 기간: {timeframe}, 🌍 지역: {geo if geo else '전 세계'}")
        
        # pytrends 라이브러리 설치 확인
        try:
            from pytrends.request import TrendReq
        except ImportError:
            return {
                "error": "pytrends 라이브러리가 설치되지 않았습니다.",
                "suggestions": [
                    "터미널에서 다음 명령어 실행:",
                    "pip install pytrends",
                    "또는",
                    "pip install --upgrade pytrends"
                ]
            }
        
        # pytrends 객체 생성 시도
        st.info("🔧 pytrends 초기화 중...")
        pytrends, error = GoogleTrendsAnalyzer._safe_create_pytrends()
        
        if pytrends is None:
            return {
                "error": f"pytrends 초기화 실패: {error}",
                "suggestions": [
                    "pytrends 라이브러리 재설치:",
                    "pip uninstall pytrends",
                    "pip install pytrends",
                    "또는 잠시 후 다시 시도"
                ]
            }
        
        st.success("✅ pytrends 초기화 성공!")
        
        # 결과 저장용
        result = {
            'interest_over_time': None,
            'interest_by_region': None,
            'related_queries': None,
            'related_topics': None,
            'keywords': clean_keywords,
            'timeframe': timeframe
        }
        
        # 데이터 수집 시도
        success_count = 0
        
        # 1. 시간별 관심도 데이터 (가장 중요)
        st.info("📊 시간별 트렌드 데이터 수집 중...")
        try:
            # 페이로드 빌드
            pytrends.build_payload(
                clean_keywords, 
                cat=0, 
                timeframe=timeframe, 
                geo=geo, 
                gprop=''
            )
            
            # 시간별 데이터 가져오기
            interest_over_time = pytrends.interest_over_time()
            
            if (interest_over_time is not None and 
                hasattr(interest_over_time, 'empty') and 
                not interest_over_time.empty):
                
                # 'isPartial' 컬럼 제거
                if 'isPartial' in interest_over_time.columns:
                    interest_over_time = interest_over_time.drop('isPartial', axis=1)
                
                # 실제 데이터 있는지 확인
                data_sum = 0
                for col in interest_over_time.columns:
                    if col in clean_keywords:
                        data_sum += interest_over_time[col].sum()
                
                if data_sum > 0:
                    result['interest_over_time'] = interest_over_time
                    success_count += 1
                    st.success(f"✅ 시간별 트렌드 데이터 수집 완료 ({len(interest_over_time)}개 포인트)")
                else:
                    st.warning("⚠️ 시간별 트렌드 데이터에 검색량이 없습니다.")
            else:
                st.warning("⚠️ 시간별 트렌드 데이터가 비어있습니다.")
                
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg:
                st.error("❌ 요청 한도 초과 - 5분 후 다시 시도하세요.")
            elif "timeout" in error_msg:
                st.error("❌ 요청 시간 초과 - 네트워크 확인 후 재시도하세요.")
            else:
                st.error(f"❌ 시간별 데이터 수집 실패: {str(e)}")
        
        # 2. 지역별 관심도 데이터 (선택사항)
        if success_count > 0:  # 시간별 데이터가 있을 때만 시도
            st.info("🌍 지역별 관심도 데이터 수집 중...")
            try:
                interest_by_region = pytrends.interest_by_region(
                    resolution='COUNTRY', 
                    inc_low_vol=True, 
                    inc_geo_code=False
                )
                
                if (interest_by_region is not None and 
                    hasattr(interest_by_region, 'empty') and 
                    not interest_by_region.empty):
                    
                    result['interest_by_region'] = interest_by_region
                    success_count += 1
                    st.success(f"✅ 지역별 관심도 데이터 수집 완료 ({len(interest_by_region)}개 국가)")
                else:
                    st.info("ℹ️ 지역별 관심도 데이터가 없습니다.")
                    
            except Exception as e:
                st.info(f"ℹ️ 지역별 데이터 수집 실패: {str(e)}")
        
        # 3. 관련 검색어 (선택사항)
        if success_count > 0:  # 기본 데이터가 있을 때만 시도
            st.info("🔍 관련 검색어 수집 중...")
            try:
                related_queries = pytrends.related_queries()
                
                if (related_queries is not None and 
                    isinstance(related_queries, dict) and 
                    len(related_queries) > 0):
                    
                    result['related_queries'] = related_queries
                    success_count += 1
                    st.success("✅ 관련 검색어 데이터 수집 완료")
                else:
                    st.info("ℹ️ 관련 검색어 데이터가 없습니다.")
                    
            except Exception as e:
                st.info(f"ℹ️ 관련 검색어 수집 실패: {str(e)}")
        
        # 결과 검증
        if success_count == 0:
            return {
                "error": "수집된 데이터가 없습니다.",
                "suggestions": [
                    "🔄 다른 키워드 시도:",
                    f"  • '{clean_keywords[0]}' 대신 '{clean_keywords[0]} stock' 사용",
                    "📅 분석 기간 변경:",
                    "  • '지난 12개월' 또는 '지난 5년' 선택",
                    "🌍 지역 설정 변경:",
                    "  • '전 세계' 또는 '미국' 선택",
                    "⏰ 잠시 후 재시도:",
                    "  • 5-10분 후 다시 시도",
                    "🔤 영어 키워드 사용:",
                    "  • 한글보다 영어 키워드가 더 정확"
                ]
            }
        
        st.success(f"🎉 총 {success_count}개 데이터 유형 수집 완료!")
        return result
    
    @staticmethod
    def create_trends_chart(trends_data: Dict) -> go.Figure:
        """트렌드 차트 생성 - 안전성 강화"""
        try:
            if not trends_data or trends_data.get('error'):
                return go.Figure()
            
            interest_df = trends_data.get('interest_over_time')
            keywords = trends_data.get('keywords', [])
            
            if (interest_df is None or 
                not hasattr(interest_df, 'empty') or 
                interest_df.empty):
                return go.Figure()
            
            fig = go.Figure()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            added_traces = 0
            
            for i, keyword in enumerate(keywords):
                try:
                    if keyword in interest_df.columns:
                        y_data = interest_df[keyword]
                        if y_data.sum() > 0:  # 데이터가 있는 경우만
                            fig.add_trace(go.Scatter(
                                x=interest_df.index,
                                y=y_data,
                                mode='lines+markers',
                                name=keyword,
                                line=dict(color=colors[i % len(colors)], width=2),
                                marker=dict(size=4),
                                hovertemplate=f'<b>{keyword}</b><br>' +
                                            'Date: %{x}<br>' +
                                            'Interest: %{y}<br>' +
                                            '<extra></extra>'
                            ))
                            added_traces += 1
                except Exception:
                    continue
            
            if added_traces == 0:
                return go.Figure()
            
            fig.update_layout(
                title=f'구글 검색 트렌드 ({trends_data.get("timeframe", "12개월")})',
                xaxis_title='날짜',
                yaxis_title='검색 관심도 (0-100)',
                height=500,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"차트 생성 실패: {str(e)}")
            return go.Figure()
    
    @staticmethod
    def create_regional_chart(trends_data: Dict) -> go.Figure:
        """지역별 차트 생성 - 안전성 강화"""
        try:
            if not trends_data or trends_data.get('error'):
                return go.Figure()
            
            regional_df = trends_data.get('interest_by_region')
            keywords = trends_data.get('keywords', [])
            
            if (regional_df is None or 
                not hasattr(regional_df, 'empty') or 
                regional_df.empty or 
                not keywords):
                return go.Figure()
            
            first_keyword = keywords[0]
            if first_keyword not in regional_df.columns:
                return go.Figure()
            
            # 0보다 큰 값만 필터링
            valid_data = regional_df[regional_df[first_keyword] > 0]
            if valid_data.empty:
                return go.Figure()
            
            top_regions = valid_data[first_keyword].sort_values(ascending=False).head(20)
            if top_regions.empty:
                return go.Figure()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_regions.values,
                y=top_regions.index,
                orientation='h',
                marker=dict(
                    color=top_regions.values,
                    colorscale='Blues',
                    showscale=True,
                    colorbar=dict(title="관심도")
                ),
                text=[f"{val:.0f}" for val in top_regions.values],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>' +
                            first_keyword + ' 관심도: %{x}<br>' +
                            '<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'국가별 검색 관심도 - {first_keyword}',
                xaxis_title='검색 관심도',
                yaxis_title='국가',
                height=600,
                yaxis=dict(autorange="reversed")
            )
            
            return fig
            
        except Exception as e:
            st.error(f"지역별 차트 생성 실패: {str(e)}")
            return go.Figure()
    
    @staticmethod
    def analyze_trends_with_gpt(trends_data: Dict, openai_api_key: str) -> Dict:
        """GPT를 이용한 트렌드 분석"""
        try:
            if not openai_api_key or not trends_data:
                return {"error": "API 키 또는 트렌드 데이터가 없습니다."}
            
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)
            
            # 트렌드 데이터 텍스트로 변환
            keywords = trends_data.get('keywords', [])
            interest_df = trends_data.get('interest_over_time')
            
            if interest_df is None or interest_df.empty:
                return {"error": "트렌드 데이터가 비어있습니다."}
            
            # 최근 트렌드 분석
            analysis_text = f"📊 구글 트렌드 분석 데이터:\n\n"
            analysis_text += f"검색어: {', '.join(keywords)}\n"
            analysis_text += f"분석 기간: {trends_data.get('timeframe', '12개월')}\n\n"
            
            # 각 키워드별 최근 트렌드
            for keyword in keywords:
                if keyword in interest_df.columns:
                    recent_values = interest_df[keyword].tail(10)
                    avg_recent = recent_values.mean()
                    trend_direction = "상승" if recent_values.iloc[-1] > recent_values.iloc[0] else "하락"
                    max_val = interest_df[keyword].max()
                    max_date = interest_df[keyword].idxmax()
                    
                    analysis_text += f"• {keyword}:\n"
                    analysis_text += f"  - 최근 평균 관심도: {avg_recent:.1f}\n"
                    analysis_text += f"  - 최근 트렌드: {trend_direction}\n"
                    analysis_text += f"  - 최고 관심도: {max_val} (날짜: {max_date.strftime('%Y-%m-%d')})\n\n"
            
            # 관련 검색어 정보 추가
            related_queries = trends_data.get('related_queries', {})
            if related_queries:
                analysis_text += "🔍 관련 상승 검색어:\n"
                for keyword in keywords:
                    if keyword in related_queries and related_queries[keyword].get('rising') is not None:
                        rising_queries = related_queries[keyword]['rising'].head(3)
                        for _, row in rising_queries.iterrows():
                            analysis_text += f"  - {row['query']} (+{row['value']}%)\n"
                analysis_text += "\n"

            prompt = f"""
다음은 구글 트렌드 데이터 분석 결과입니다. 투자자 관점에서 분석해주세요:

{analysis_text}

다음 형식으로 분석해주세요:

### 🔍 구글 트렌드 종합 분석

#### 1. **검색 트렌드 요약** (5점 만점으로 점수 부여)
- 종합 점수: X/5점
- 한 줄 요약

#### 2. **주요 트렌드 패턴**
- 검색량 변화의 특징적 패턴
- 계절성 또는 주기성 분석
- 급상승/급하락 구간 분석

#### 3. **투자자 관심도 분석**
- 검색량과 투자 심리의 상관관계
- 대중의 관심도 변화 의미
- 미디어 노출과의 연관성

#### 4. **지역별 관심도 인사이트**
- 주요 관심 지역 분석
- 글로벌 vs 로컬 관심도
- 지역별 투자 트렌드 시사점

#### 5. **관련 검색어 분석**
- 상승 검색어의 의미
- 투자자들의 주요 관심사
- 시장 심리 반영 키워드

#### 6. **투자 시사점**
- 검색 트렌드와 주가의 일반적 상관관계
- 현재 트렌드가 시사하는 바
- 주의해야 할 신호들

#### 7. **향후 전망 및 모니터링 포인트**
- 검색 트렌드 방향성 예측
- 주시해야 할 변곡점
- 추가 모니터링 키워드

#### 8. **결론**
- 현재 대중 관심도 한 줄 요약
- 투자자 관점에서의 종합 의견

각 섹션을 구체적이고 실용적으로 분석해주시고, 구글 트렌드를 투자 분석에 활용하는 실용적인 인사이트를 제공해주세요.
"""

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "당신은 전문 투자 분석가입니다. 구글 트렌드 데이터를 활용하여 투자자에게 유용한 시장 심리 분석을 제공합니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content.strip()
            
            # 점수 추출
            import re
            score_match = re.search(r'점수[:\s]*(\d+(?:\.\d+)?)[/점]', analysis)
            score = float(score_match.group(1)) if score_match else 3.0
            
            return {
                "analysis": analysis,
                "score": score,
                "max_score": 5.0,
                "success": True
            }
            
        except Exception as e:
            return {"error": f"GPT 트렌드 분석 실패: {str(e)}"}

    @staticmethod
    def get_stock_related_keywords(symbol: str, company_name: str = None) -> List[str]:
        """주식 관련 키워드 생성 - 더 안전하게"""
        keywords = []
        
        if symbol and isinstance(symbol, str):
            symbol = symbol.strip().upper()
            keywords.append(symbol)
            keywords.append(f"{symbol} stock")
        
        if company_name and isinstance(company_name, str):
            clean_name = company_name.strip()
            # 회사 suffix 제거
            for suffix in [' Inc.', ' Corp.', ' Ltd.', ' Co.', ' LLC', ' Plc']:
                clean_name = clean_name.replace(suffix, '')
            
            if clean_name and clean_name != symbol:
                keywords.append(clean_name)
        
        # 중복 제거 및 유효성 검사
        unique_keywords = []
        for kw in keywords:
            if kw and kw.strip() and kw.strip() not in unique_keywords:
                unique_keywords.append(kw.strip())
        
        return unique_keywords[:3]  # 최대 3개로 제한
    
    # 더미 데이터 생성 함수 (백업용)
@staticmethod
def create_demo_trends_data(keywords: List[str]) -> Dict:
    """데모/테스트용 트렌드 데이터 생성"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # 날짜 생성 (최근 12개월)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='W')
    
    # 더미 트렌드 데이터 생성
    demo_data = {}
    for keyword in keywords[:3]:
        # 랜덤하지만 현실적인 트렌드 패턴 생성
        base_trend = 30 + 20 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
        noise = np.random.normal(0, 5, len(dates))
        values = np.maximum(0, base_trend + noise)
        demo_data[keyword] = values
    
    demo_df = pd.DataFrame(demo_data, index=dates)
    
    return {
        'interest_over_time': demo_df,
        'interest_by_region': None,
        'related_queries': None,
        'related_topics': None,
        'keywords': keywords[:3],
        'timeframe': '12-m',
        'is_demo': True
    }

def main():
    """메인 애플리케이션"""
    
    st.markdown('<h1 class="main-header"> 서학개미의 투자 탐구생활 🐜🔍</h1>', unsafe_allow_html=True)
    st.markdown('### AI 기반 주가 예측 및 투자 분석 플랫폼')
    
    # 면책 고지
    st.warning('⚠️ **투자 면책 고지**: 본 분석은 정보 제공 목적이며 투자 자문이 아닙니다. 모든 투자 결정은 본인 책임하에 하시기 바랍니다.')
    
    # 사이드바 - 입력 설정
    st.sidebar.header('📊 분석 설정')
    
    # 종목 입력
    symbols_input = st.sidebar.text_input(
        '종목 코드 입력 (쉼표로 구분)',
        value='AAPL',
        help='예: AAPL,MSFT,GOOGL'
    )
    
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    
    # 분석 옵션
    prediction_model = st.sidebar.selectbox(
        '예측 모델 선택',
        ['Prophet', 'ARIMA']
    )
    
    prediction_days = st.sidebar.slider(
        '예측 기간 (일)',
        min_value=7,
        max_value=180,
        value=30
    )
    
    # 탭 생성 (섹터 트리맵 탭 추가)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(['🏠 홈', '📊 섹터 트리맵', '📈 주가 예측', '📰 뉴스 분석', '💰 재무 건전성', '🌍 경제지표', '🧠 버핏 조언', '📺 유튜브 분석', '📈 구글 트렌드'])
    
    with tab1:
        st.header('🏠 서비스 개요')
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.subheader('📊 섹터 트리맵')
            st.write('실시간 섹터별 시가총액 시각화')
            st.write('• S&P 500 기반 섹터 분류')
            st.write('• 시가총액 상위 20개 종목')
            st.write('• 일일 변동률 색상 표시')
            st.write('• 트리맵 인터랙티브 차트')
        
        with col2:
            st.subheader('📈 주가 및 거래량 예측')
            st.write('AI 모델을 활용한 주가 및 거래량 예측')
            st.write('• Prophet: Facebook의 시계열 예측')
            st.write('• ARIMA: 전통적인 통계 모델')
            st.write('• 백테스트 성능 지표 제공')
            st.write('• 시각화 차트')
        
        with col3:
            st.subheader('📰 뉴스 분석')
            st.write('구글 뉴스 분석')
            st.write('• 실시간 감성 분석')
            st.write('• 긍정/부정/중립 분류')
            st.write('• 트렌드 시각화')
            st.write('• GPT 기반 뉴스 요약')

        with col4:
            st.subheader('🧠 워렌 버핏 스타일 조언')
            st.write('가치 투자 관점의 종합 분석')
            st.write('• ROE, P/E 비율 등 핵심 지표')
            st.write('• 부채비율 및 재무 건전성')
            st.write('• 자유현금흐름 분석')
            st.write('• 매수/보유/매도 등급 제공')
            st.write('• 버핏 철학 기반 점수 시스템')

        # 추가 섹션
        st.markdown('---')
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.subheader('💰 재무 지표 분석')
            st.write('과거 실적 기반 트렌드 분석')
            st.write('• 매출, 영업이익, 순이익 추이')
            st.write('• 연도별 성장률 분석')
            st.write('• 수익성 지표 변화')
            st.write('• GPT 기반 재무 지표 분석')
            
        with col2:
            st.subheader('🌍 대외 경제지표')
            st.write('거시경제 환경 분석')
            st.write('• 주요 주식 지수 추이')
            st.write('• 금리, 금시세, 달러 동향')
            st.write('• 시장 변동성 지표')
            st.write('• 원자재 가격 추이')
            st.write('• GPT 기반 거시경제 분석')

        with col3:
            st.subheader('📺 유튜브 영상 분석')
            st.write('주식 관련 유튜브 콘텐츠 AI 분석')
            st.write('• 실시간 영상 검색')
            st.write('• GPT 기반 영상 요약')
            st.write('• 투자 포인트 자동 추출')
            st.write('• 전문가 의견 종합 분석')
            st.write('• 리스크 요인 식별')

        with col4:
            st.subheader('📈 구글 트렌드 분석')
            st.write('검색량 기반 시장 심리 분석')
            st.write('• 실시간 검색 관심도 추이')
            st.write('• 지역별 관심도 분포')
            st.write('• 관련 검색어 분석')
            st.write('• GPT 기반 트렌드 해석')
            st.write('• 투자 심리 지표로 활용')

        # 플랫폼 특징 섹션 추가
        st.markdown('---')
        st.header('🚀 플랫폼 핵심 특징')

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('🤖 AI 기반 통합 분석')
            st.success("""
            **GPT-4 활용 전문가급 분석**
            • 뉴스 감성 분석 및 요약
            • 재무지표 전문가 해석  
            • 경제환경 종합 진단
            • 구글 트렌드 투자 심리 분석
            • 유튜브 영상 핵심 요약
            
            **실시간 데이터 연동**
            • Yahoo Finance 실시간 주가
            • Google News 최신 뉴스
            • Google Trends 검색량
            • S&P 500 섹터별 데이터
            • 경제지표 실시간 업데이트
            """)
        
        with col2:
            st.subheader('📊 차별화된 분석 기능')
            st.info("""
            **독창적인 분석 도구**
            • 섹터별 트리맵 시각화
            • 워렌 버핏 스타일 평가 시스템
            • 주가 + 거래량 동시 예측
            • 3개년 재무실적 트렌드 분석
            • 거시경제 vs 개별종목 연관분석
            
            **사용자 친화적 설계**
            • 직관적인 탭 구조
            • 대화형 차트 (Plotly)
            • 단계별 가이드 제공
            • 모바일/데스크톱 반응형
            • 한 번에 여러 종목 분석 가능
            """)

        # 투자 프로세스 가이드
        st.markdown('---')
        st.header('📋 추천 투자 분석 프로세스')

        process_steps = [
            ("1️⃣ **섹터 트리맵**", "관심 섹터의 전반적인 시장 상황 파악"),
            ("2️⃣ **주가 예측**", "AI 모델로 향후 주가 및 거래량 전망"),
            ("3️⃣ **뉴스 분석**", "최신 뉴스의 감성과 시장 영향 평가"),
            ("4️⃣ **재무 건전성**", "기업의 재무 상태와 성장성 점검"),
            ("5️⃣ **경제지표**", "거시경제 환경이 투자에 미치는 영향"),
            ("6️⃣ **버핏 조언**", "가치 투자 관점에서의 종합 평가"),
            ("7️⃣ **유튜브 분석**", "전문가들의 의견과 시장 분위기 확인"),
            ("8️⃣ **구글 트렌드**", "대중의 관심도와 투자 심리 측정"),
            ("9️⃣ **종합 판단**", "모든 분석을 종합하여 최종 투자 결정")
        ]
        
        for i in range(0, len(process_steps), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(process_steps):
                    step, description = process_steps[i + j]
                    with cols[j]:
                        st.markdown(f"**{step}**")
                        st.write(description)

        # 성공 사례 및 활용 팁
        st.markdown('---')
        st.header('💡 효과적인 활용 방법')
        
        with st.expander('🎯 초보 투자자를 위한 가이드'):
            st.markdown("""
            **📚 투자 학습 단계별 활용법**
            
            **🥉 초급자 (투자 경험 6개월 미만)**
            1. 워렌 버핏 조언 탭부터 시작 → 기본적인 재무지표 학습
            2. 뉴스 분석으로 시장 감성 이해
            3. 섹터 트리맵으로 시장 전체 흐름 파악
            
            **🥈 중급자 (투자 경험 6개월~2년)**
            1. 재무 건전성 + 경제지표 조합 분석
            2. 주가 예측 모델로 타이밍 연구
            3. 구글 트렌드로 대중 심리 활용
            
            **🥇 고급자 (투자 경험 2년 이상)**
            1. 모든 탭을 종합하여 다각도 분석
            2. GPT 분석 결과와 본인 분석 비교
            3. 포트폴리오 차원에서 종목 간 상관관계 분석
            """)
        
        with st.expander('⚡ 시간대별 효율적 사용법'):
            st.markdown("""
            **🕐 시간대별 최적 활용 전략**
            
            **🌅 장 시작 전 (9:00 AM 이전)**
            • 경제지표 → 오늘의 시장 환경 파악
            • 뉴스 분석 → 밤사이 발생한 이슈 확인
            • 섹터 트리맵 → 프리마켓 동향 체크
            
            **📈 장중 (9:30 AM - 4:00 PM)**
            • 실시간 트리맵으로 섹터 로테이션 모니터링
            • 구글 트렌드로 급등/급락 종목의 관심도 확인
            • 뉴스 분석으로 돌발 이슈 대응
            
            **🌆 장 마감 후 (4:00 PM 이후)**
            • 재무 건전성으로 신규 종목 발굴
            • 버핏 조언으로 장기 투자 종목 선별
            • 유튜브 분석으로 전문가 의견 수집
            • 주가 예측으로 다음 주 전략 수립
            
            **📅 주말 (토-일)**
            • 모든 탭 종합 분석으로 다음 주 계획 수립
            • GPT 분석 결과들을 정리하여 투자 일지 작성
            • 새로운 종목 후보군 발굴 및 사전 조사
            """)
        
        with st.expander('🔥 고급 활용 팁'):
            st.markdown("""
            **🚀 전문가 수준 활용 전략**
            
            **📊 다중 종목 비교 분석**
            • 동일 섹터 내 3-5개 종목 동시 분석
            • 워렌 버핏 점수로 종목 순위 매기기
            • 구글 트렌드로 관심도 상대 비교
            
            **⏰ 타이밍 최적화**
            • VIX 공포지수 + 뉴스 감성 조합으로 시장 타이밍 포착
            • 구글 트렌드 급상승 + 주가 예측 조합으로 모멘텀 투자
            • 경제지표 변화 + 섹터 트리맵으로 섹터 로테이션 전략
            
            **🎯 리스크 관리**
            • 버핏 조언 낮은 점수 + 부정적 뉴스 = 투자 제외
            • 경제지표 악화 + VIX 상승 = 방어적 포지션
            • 트렌드 관심도 급락 + 거래량 감소 = 청산 신호
            
            **💎 숨은 보석 발굴**
            • 재무 건전성 우수 + 구글 트렌드 낮음 = 저평가 후보
            • 유튜브 전문가 긍정 + 뉴스 일시적 부정 = 단기 기회
            • 섹터 전체 하락 + 개별 기업 실적 양호 = 역발상 투자
            """)
    
    with tab2:
        st.header('📊 섹터별 시가총액 트리맵')
        
        # S&P 500 종목 리스트 가져오기
        with st.spinner('S&P 500 종목 리스트를 가져오는 중...'):
            sp500_df = SectorTreemapAnalyzer.get_sp500_stocks()
        
        if sp500_df.empty:
            st.error('S&P 500 데이터를 가져올 수 없습니다.')
        else:
            # 섹터 선택
            available_sectors = sorted(sp500_df['sector'].unique())
            selected_sectors = st.multiselect(
                '분석할 섹터를 선택하세요 (최대 4개 권장)',
                available_sectors,
                default=[]
            )
            
            if not selected_sectors:
                st.warning('분석할 섹터를 선택해주세요.')
            else:
                # 섹터별 분석 실행
                all_sectors_data = {}
                
                for sector in selected_sectors:
                    st.subheader(f'🏢 {sector} 섹터')
                    
                    with st.spinner(f'{sector} 섹터 데이터를 수집하는 중...'):
                        sector_data = SectorTreemapAnalyzer.get_sector_market_data(sp500_df, sector, top_n=20)
                        all_sectors_data[sector] = sector_data
                    
                    if not sector_data.empty:
                        # 트리맵 생성 및 표시
                        fig = SectorTreemapAnalyzer.create_sector_treemap(sector_data, sector)
                        st.plotly_chart(fig, use_container_width=True, key=f"treemap_{sector}")
                        
                        # 섹터 요약 정보
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_market_cap = sector_data['market_cap'].sum() / 1e12
                            st.metric('총 시가총액', f'${total_market_cap:.2f}T')
                        
                        with col2:
                            avg_change = sector_data['change_pct'].mean()
                            color = 'normal' if avg_change > 0 else 'inverse'
                            st.metric('평균 변동률', f'{avg_change:+.2f}%', delta_color=color)
                        
                        with col3:
                            positive_count = len(sector_data[sector_data['change_pct'] > 0])
                            st.metric('상승 종목', f'{positive_count}/{len(sector_data)}')
                        
                        with col4:
                            top_stock = sector_data.iloc[0]
                            st.metric('최대 종목', f"{top_stock['symbol']}")
                        
                        # 상위 5개 종목 테이블
                        st.subheader(f'📈 {sector} 섹터 TOP 5')
                        top_5 = sector_data.head(5)[['symbol', 'company_name', 'market_cap_b', 'current_price', 'change_pct']]
                        top_5.columns = ['종목코드', '회사명', '시가총액($B)', '현재가($)', '변동률(%)']
                        
                        # 변동률에 따른 색상 적용
                        def color_negative_red(val):
                            if isinstance(val, (int, float)):
                                color = 'color: red' if val < 0 else 'color: green'
                                return color
                            return ''
                        
                        styled_df = top_5.style.applymap(color_negative_red, subset=['변동률(%)'])
                        st.dataframe(styled_df, use_container_width=True)
                        
                        st.markdown('---')
                    else:
                        st.warning(f'{sector} 섹터의 데이터를 가져올 수 없습니다.')
                
                # 전체 섹터 요약 차트
                if all_sectors_data:
                    st.header('📊 선택된 섹터 종합 비교')
                    summary_fig = SectorTreemapAnalyzer.create_sector_summary_chart(all_sectors_data)
                    if summary_fig.data:
                        st.plotly_chart(summary_fig, use_container_width=True, key="sector_summary")
                
                # 색상 범례 설명
                with st.expander('🎨 트리맵 색상 가이드'):
                    st.markdown("""
                    **📊 트리맵 해석 방법:**
                    
                    **박스 크기**: 시가총액 (클수록 시가총액이 큼)
                    
                    **색상 의미**:
                    - 🟢 **진한 초록**: +2% 이상 상승 (강한 상승)
                    - 🍃 **연한 초록**: 0% ~ +2% 상승 (약한 상승)
                    - 🌸 **연한 빨강**: 0% ~ -2% 하락 (약한 하락)
                    - 🔴 **진한 빨강**: -2% 이하 하락 (강한 하락)
                    
                    **정보 표시**:
                    - 첫 번째 줄: 종목 코드
                    - 두 번째 줄: 회사명
                    - 세 번째 줄: 시가총액 (Billions)
                    - 네 번째 줄: 일일 변동률
                    
                    **💡 투자 활용 팁**:
                    - 큰 박스면서 초록색 → 대형주 상승 (시장 강세 신호)
                    - 작은 박스들이 대부분 빨간색 → 소형주 약세
                    - 섹터 내 색상이 고르게 분포 → 개별 종목 이슈
                    - 섹터 내 색상이 한쪽으로 치우침 → 섹터 전반 이슈
                    """)
    
    with tab3:
        st.header('📈 주가 예측 분석')
        
        if not symbols:
            st.warning('종목 코드를 입력해주세요.')
        else:
            # 거래량 예측 옵션 추가
            st.subheader('⚙️ 예측 설정')
            col1, col2 = st.columns(2)
            
            with col1:
                include_volume = st.checkbox('거래량 예측 포함', value=True, help='주가 예측과 함께 거래량도 예측합니다')
            
            with col2:
                chart_type = st.selectbox('차트 형태', ['개별 차트', '통합 차트'], help='개별 차트: 주가와 거래량을 따로 표시\n통합 차트: 주가와 거래량을 함께 표시')
            
            for symbol in symbols:
                st.subheader(f'{symbol} 분석')
                
                with st.spinner(f'{symbol} 데이터를 가져오는 중...'):
                    # 주가 데이터 가져오기
                    stock_data = DataManager.get_stock_data(symbol)
                    
                    if stock_data.empty:
                        st.error(f'{symbol} 데이터를 가져올 수 없습니다.')
                        continue
                    
                    # 주가 예측 모델 실행
                    if prediction_model == 'Prophet':
                        price_forecast, price_metrics = PredictionModel.prophet_forecast(stock_data, prediction_days)
                        
                        # 거래량 예측 (옵션이 선택된 경우)
                        if include_volume:
                            volume_forecast, volume_metrics = PredictionModel.prophet_volume_forecast(stock_data, prediction_days)
                        else:
                            volume_forecast, volume_metrics = pd.DataFrame(), {}
                    else:  # ARIMA
                        price_forecast, price_metrics = PredictionModel.arima_forecast(stock_data, prediction_days)
                        
                        # 거래량 예측 (옵션이 선택된 경우)
                        if include_volume:
                            volume_forecast, volume_metrics = PredictionModel.arima_volume_forecast(stock_data, prediction_days)
                        else:
                            volume_forecast, volume_metrics = pd.DataFrame(), {}
                    
                    # 차트 표시
                    if include_volume and chart_type == '통합 차트' and not volume_forecast.empty:
                        # 통합 차트
                        combined_fig = create_combined_chart(stock_data, stock_data, price_forecast, volume_forecast, symbol)
                        st.plotly_chart(combined_fig, use_container_width=True, key=f"combined_chart_{symbol}")
                    else:
                        # 개별 차트들
                        # 주가 차트
                        price_fig = create_stock_chart(stock_data, price_forecast, symbol)
                        st.plotly_chart(price_fig, use_container_width=True, key=f"price_chart_{symbol}")
                        
                        # 거래량 차트 (옵션이 선택된 경우)
                        if include_volume:
                            if not volume_forecast.empty:
                                volume_fig = create_volume_chart(stock_data, volume_forecast, symbol)
                                st.plotly_chart(volume_fig, use_container_width=True, key=f"volume_chart_{symbol}")
                            else:
                                st.warning('거래량 예측 데이터를 생성할 수 없습니다.')
                    
                    # 성능 지표 표시
                    if include_volume and not volume_forecast.empty:
                        # 주가와 거래량 성능 지표를 함께 표시
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader('📈 주가 예측 성능')
                            if price_metrics:
                                subcol1, subcol2, subcol3 = st.columns(3)
                                with subcol1:
                                    st.metric('예측 모델', prediction_model)
                                with subcol2:
                                    st.metric('MAPE', f"{price_metrics.get('MAPE', 0):.2f}%")
                                with subcol3:
                                    st.metric('RMSE', f"{price_metrics.get('RMSE', 0):.2f}")
                        
                        with col2:
                            st.subheader('📊 거래량 예측 성능')
                            if volume_metrics and not volume_metrics.get('error'):
                                subcol1, subcol2, subcol3 = st.columns(3)
                                with subcol1:
                                    st.metric('예측 모델', prediction_model)
                                with subcol2:
                                    st.metric('MAPE', f"{volume_metrics.get('MAPE', 0):.2f}%")
                                with subcol3:
                                    st.metric('RMSE', f"{volume_metrics.get('RMSE', 0):.0f}")
                            elif volume_metrics.get('error'):
                                st.error(volume_metrics['error'])
                    else:
                        # 주가 성능 지표만 표시
                        if price_metrics:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric('예측 모델', prediction_model)
                            with col2:
                                st.metric('MAPE', f"{price_metrics.get('MAPE', 0):.2f}%")
                            with col3:
                                st.metric('RMSE', f"{price_metrics.get('RMSE', 0):.2f}")
                    
                    # 거래량 트렌드 분석
                    if include_volume:
                        volume_analysis = analyze_volume_trends(stock_data, volume_forecast)
                        
                        if not volume_analysis.get('error'):
                            st.subheader('📊 거래량 트렌드 분석')
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                avg_vol = volume_analysis.get('avg_volume_30d', 0)
                                st.metric('최근 30일 평균 거래량', f"{avg_vol:,.0f}")
                            
                            with col2:
                                ratio = volume_analysis.get('volume_ratio', 0)
                                st.metric(
                                    '전체 평균 대비',
                                    f"{ratio:.2f}배",
                                    delta="활발" if ratio > 1.2 else "보통" if ratio > 0.8 else "저조"
                                )
                            
                            with col3:
                                trend = volume_analysis.get('trend_change', 0)
                                st.metric(
                                    '최근 트렌드',
                                    f"{trend:+.1f}%",
                                    delta="증가" if trend > 10 else "감소" if trend < -10 else "안정"
                                )
                            
                            with col4:
                                forecast_change = volume_analysis.get('volume_forecast_change', 0)
                                st.metric(
                                    '예측 변화율',
                                    f"{forecast_change:+.1f}%",
                                    delta="증가 예상" if forecast_change > 5 else "감소 예상" if forecast_change < -5 else "유지 예상"
                                )
                            
                            # 거래량 해석
                            with st.expander('📊 거래량 분석 해석'):
                                st.markdown(f'''
                                **거래량 분석 리포트:**
                                
                                **현재 상황:**
                                - 최근 30일 평균 거래량: {avg_vol:,.0f}
                                - 전체 평균 대비: {ratio:.2f}배 ({'활발' if ratio > 1.2 else '보통' if ratio > 0.8 else '저조'})
                                - 최근 트렌드: {trend:+.1f}% ({'증가' if trend > 10 else '감소' if trend < -10 else '안정'})
                                
                                **예측 전망:**
                                - 향후 {prediction_days}일 예상 변화: {forecast_change:+.1f}%
                                - 예상 평균 거래량: {volume_analysis.get('predicted_avg', 0):,.0f}
                                
                                **투자 시사점:**
                                - 거래량 증가: 관심도 상승, 변동성 확대 가능성
                                - 거래량 감소: 관심도 하락, 횡보 가능성
                                - 급격한 거래량 변화: 중요한 뉴스나 이벤트 발생 신호
                                ''')
                        else:
                            st.warning(f"거래량 분석 실패: {volume_analysis.get('error', '알 수 없는 오류')}")
                    
                    st.markdown('---')
    
    with tab4:
        st.header('📰 최신 뉴스 분석')
        
        if not symbols:
            st.warning('종목 코드를 입력해주세요.')
            return
        
        # OpenAI API 키 입력
        st.subheader('🔧 GPT 분석 설정 (선택사항)')
        openai_api_key = st.text_input(
            "OpenAI API 키 (GPT 기반 뉴스 분석을 위해 필요)",
            type="password",
            help="GPT를 이용한 실시간 뉴스 분석을 원하시면 OpenAI API 키를 입력하세요. 입력하지 않으면 기본 감성 분석만 진행됩니다."
        )
        
        # API 키 검증
        if openai_api_key:
            if openai_api_key.startswith('sk-') and len(openai_api_key) > 20:
                st.success("✅ API 키 형식이 올바릅니다.")
            else:
                st.error("❌ API 키 형식이 잘못되었습니다. 'sk-'로 시작하는 올바른 키를 입력해주세요.")
        else:
            st.info("ℹ️ API 키를 입력하지 않으면 기본 감성 분석만 제공됩니다.")
        
        for symbol in symbols:
            st.subheader(f'{symbol} 뉴스 감성 분석')
            
            with st.spinner(f'{symbol} 뉴스 데이터를 분석하는 중...'):
                # GPT 기반 향상된 분석
                enhanced_sentiment_data = SentimentAnalyzer.get_enhanced_news_sentiment(symbol, openai_api_key)
                
                # 디버깅 정보 표시
                if openai_api_key:
                    st.info(f"🔑 API 키 설정됨 - GPT 분석 진행 중...")
                    
                    # 뉴스 수집 상태 확인
                    news_count = len(enhanced_sentiment_data.get('news_articles', []))
                    st.info(f"📰 수집된 뉴스: {news_count}개")
                else:
                    st.warning("🔑 API 키가 설정되지 않음 - 기본 분석만 진행")
                
                if enhanced_sentiment_data.get('gpt_analysis'):
                    if enhanced_sentiment_data['gpt_analysis'].get('error'):
                        # GPT 분석 오류 표시
                        st.error(f"🤖 GPT 분석 오류: {enhanced_sentiment_data['gpt_analysis']['error']}")
                        st.info("💡 기본 감성 분석으로 대체합니다.")
                    else:
                        # GPT 분석 성공
                        st.success("🤖 GPT 분석 완료!")
                        gpt_analysis = enhanced_sentiment_data['gpt_analysis']
                        
                        # 감성 상태 표시
                        sentiment_label = gpt_analysis.get('sentiment_label', '중립적')
                        if '긍정' in sentiment_label:
                            st.success(f"📈 전체 감성: {sentiment_label}")
                        elif '부정' in sentiment_label:
                            st.error(f"📉 전체 감성: {sentiment_label}")
                        else:
                            st.info(f"📊 전체 감성: {sentiment_label}")
                        
                        # GPT 분석 내용 표시
                        st.markdown("### 📋 전문가 분석 요약")
                        st.markdown(gpt_analysis['analysis'])
                        
                        st.markdown("---")
                
                elif openai_api_key:
                    # API 키는 있지만 GPT 분석이 없는 경우
                    st.warning("🤖 GPT 분석을 실행할 수 없습니다. API 키를 확인해주세요.")
                
                # 실제 뉴스 기사들 표시
                if enhanced_sentiment_data.get('news_articles'):
                    st.subheader('📰 최신 뉴스 (실시간 수집)')
                    
                    # 뉴스 개수와 기본 감성 요약
                    if enhanced_sentiment_data.get('basic_sentiment'):
                        summary = enhanced_sentiment_data['basic_sentiment']['summary']
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric('전체 기사', summary.get('total', 0))
                        
                        with col2:
                            st.metric('긍정', summary.get('positive', 0), delta_color='normal')
                        
                        with col3:
                            st.metric('중립', summary.get('neutral', 0), delta_color='off')
                        
                        with col4:
                            st.metric('부정', summary.get('negative', 0), delta_color='inverse')
                    
                    # 감성 분석 차트 (기본 분석 기준)
                    if enhanced_sentiment_data.get('basic_sentiment'):
                        fig = create_sentiment_chart(enhanced_sentiment_data['basic_sentiment'])
                        st.plotly_chart(fig, use_container_width=True, key=f"sentiment_chart_{symbol}")
                    
                    # 개별 뉴스 기사 표시
                    st.subheader('📄 개별 뉴스 기사')
                    
                    if enhanced_sentiment_data.get('basic_sentiment'):
                        sentiments = enhanced_sentiment_data['basic_sentiment']['sentiments']
                        
                        for i, item in enumerate(sentiments[:10]):  # 최대 10개 표시
                            sentiment_color = 'green' if item['sentiment_category'] == 'Positive' else 'red' if item['sentiment_category'] == 'Negative' else 'gray'
                            
                            with st.expander(f"📰 {item['title'][:80]}{'...' if len(item['title']) > 80 else ''}"):
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.markdown(f"**제목**: {item['title']}")
                                    st.markdown(f"**출처**: {item.get('source', 'Unknown')} | **시간**: {item.get('time', 'Unknown')}")
                                    if item.get('link') and item['link'] != '#':
                                        st.markdown(f"**링크**: [기사 읽기]({item['link']})")
                                    else:
                                        st.markdown("**링크**: 링크 없음")
                                
                                with col2:
                                    st.markdown(f"**감성**: <span style='color: {sentiment_color}'>{item['sentiment_category']}</span>", unsafe_allow_html=True)
                                    st.markdown(f"**점수**: {item['sentiment_score']:.3f}")
                    
                else:
                    # 웹 검색 실패시 기본 더미 데이터 사용
                    st.warning("실시간 뉴스 수집에 실패했습니다. 기본 감성 분석을 표시합니다.")
                    sentiment_data = SentimentAnalyzer.get_news_sentiment(symbol)
                    
                    if sentiment_data:
                        # 기본 감성 분석 차트
                        fig = create_sentiment_chart(sentiment_data)
                        st.plotly_chart(fig, use_container_width=True, key=f"sentiment_chart_{symbol}")
                        
                        # 감성 요약
                        summary = sentiment_data.get('summary', {})
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric('전체 기사', summary.get('total', 0))
                        
                        with col2:
                            st.metric('긍정', summary.get('positive', 0), delta_color='normal')
                        
                        with col3:
                            st.metric('중립', summary.get('neutral', 0), delta_color='off')
                        
                        with col4:
                            st.metric('부정', summary.get('negative', 0), delta_color='inverse')
                        
                        # 개별 기사 감성 (더미 데이터)
                        st.subheader('개별 뉴스 감성 (샘플 데이터)')
                        sentiments = sentiment_data.get('sentiments', [])
                        for i, item in enumerate(sentiments[:5]):
                            sentiment_color = 'green' if item['sentiment_category'] == 'Positive' else 'red' if item['sentiment_category'] == 'Negative' else 'gray'
                            st.markdown(f"**{item['title']}**")
                            st.markdown(f"감성: <span style='color: {sentiment_color}'>{item['sentiment_category']}</span> (점수: {item['sentiment_score']:.3f})", unsafe_allow_html=True)
                            st.markdown('---')
        
        # 사용법 안내
        with st.expander('💡 GPT 뉴스 분석 사용법'):
            st.markdown("""
            **🔑 OpenAI API 키 설정**
            1. [OpenAI 홈페이지](https://platform.openai.com/api-keys)에서 API 키 발급
            2. 위의 입력창에 API 키 입력
            3. 실시간 뉴스 수집 및 GPT 분석 자동 실행
            
            **📊 제공 기능**
            - **실시간 뉴스 수집**: Google 뉴스에서 최신 기사 수집
            - **GPT 전문 분석**: AI 기반 심층 감성 및 시장 분석
            - **기사 링크**: 원문 기사로 바로 이동 가능
            - **종합 투자 조언**: 뉴스 기반 투자 시사점 제공
            
            **⚠️ 주의사항**
            - API 키가 없어도 기본 감성 분석은 가능합니다
            - GPT 분석은 OpenAI 사용량에 따라 과금됩니다
            - 실시간 뉴스 수집은 네트워크 상황에 따라 다를 수 있습니다
            """)
        
    
    with tab5:
        st.header('💰 재무 건전성 분석')
        
        if not symbols:
            st.warning('종목 코드를 입력해주세요.')
            return
        
        # OpenAI API 키 입력 섹션 추가
        st.subheader('🔧 GPT 재무 분석 설정 (선택사항)')
        col1, col2 = st.columns([2, 1])
        
        with col1:
            financial_openai_api_key = st.text_input(
                "OpenAI API 키 (GPT 기반 재무 분석을 위해 필요)",
                type="password",
                help="GPT를 이용한 전문가 수준의 재무 분석을 원하시면 OpenAI API 키를 입력하세요.",
                key="financial_api_key"
            )
        
        with col2:
            if financial_openai_api_key:
                if financial_openai_api_key.startswith('sk-') and len(financial_openai_api_key) > 20:
                    st.success("✅ API 키 활성화")
                else:
                    st.error("❌ 잘못된 API 키")
        
        for symbol in symbols:
            st.subheader(f'{symbol} 재무 분석')
            
            with st.spinner(f'{symbol} 재무 데이터를 분석하는 중...'):
                financial_data = DataManager.get_financial_data(symbol)
                
                if financial_data:
                    metrics = FinancialAnalyzer.calculate_financial_metrics(financial_data)
                    
                    if metrics:
                        # 3개년 재무 실적 데이터 가져오기
                        financial_history = EconomicIndicatorAnalyzer.get_financial_history(symbol)
                        
                        # GPT 기반 재무 분석 (API 키가 있는 경우)
                        if financial_openai_api_key:
                            st.subheader('🤖 AI 재무 전문가 분석')
                            
                            with st.spinner('GPT가 재무 지표를 분석하는 중...'):
                                gpt_analysis = FinancialAnalyzer.analyze_financial_metrics_with_gpt(
                                    metrics, financial_history, symbol, financial_openai_api_key
                                )
                            
                            if gpt_analysis.get('error'):
                                st.error(f"🤖 GPT 분석 오류: {gpt_analysis['error']}")
                                st.info("💡 기본 재무 분석으로 진행합니다.")
                            else:
                                # GPT 분석 성공
                                score = gpt_analysis.get('score', 3.0)
                                grade_info = FinancialAnalyzer.get_financial_grade(score)
                                
                                # 점수와 등급 표시
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        "재무 건전성 점수",
                                        f"{score:.1f}/5.0",
                                        delta=f"{(score/5.0)*100:.0f}%"
                                    )
                                
                                with col2:
                                    if grade_info['color'] == 'success':
                                        st.success(f"등급: {grade_info['grade']} ({grade_info['description']})")
                                    elif grade_info['color'] == 'warning':
                                        st.warning(f"등급: {grade_info['grade']} ({grade_info['description']})")
                                    else:
                                        st.error(f"등급: {grade_info['grade']} ({grade_info['description']})")
                                
                                with col3:
                                    # 상태에 따른 이모지 표시
                                    if score >= 4.0:
                                        st.success("🏆 우수 기업")
                                    elif score >= 3.0:
                                        st.info("📊 양호한 기업")
                                    elif score >= 2.0:
                                        st.warning("⚠️ 주의 필요")
                                    else:
                                        st.error("🚨 위험 신호")
                                
                                # GPT 분석 내용 표시
                                st.markdown("### 📋 전문가 재무 분석 리포트")
                                st.markdown(gpt_analysis['analysis'])
                                
                                st.markdown("---")
                        
                        # 기존 재무 지표 차트 및 데이터 표시
                        st.subheader('📊 재무 지표 시각화')
                        fig = create_financial_metrics_chart(metrics)
                        st.plotly_chart(fig, use_container_width=True, key=f"financial_chart_{symbol}")
                        
                        # 3개년 재무 실적
                        if financial_history and financial_history.get('years'):
                            st.subheader('📈 최근 3개년 재무 실적')
                            fig_history = create_financial_history_chart(financial_history)
                            st.plotly_chart(fig_history, use_container_width=True, key=f"financial_history_{symbol}")
                            
                            # 실적 요약 테이블
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write('**📈 매출 추이**')
                                for i, year in enumerate(financial_history['years']):
                                    revenue = financial_history['revenue'][i]
                                    if i > 0:
                                        prev_revenue = financial_history['revenue'][i-1]
                                        growth = ((revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
                                        st.write(f"• {year}: ${revenue:.1f}B ({growth:+.1f}%)")
                                    else:
                                        st.write(f"• {year}: ${revenue:.1f}B")
                            
                            with col2:
                                st.write('**💼 영업이익 추이**')
                                for i, year in enumerate(financial_history['years']):
                                    op_income = financial_history['operating_income'][i]
                                    if i > 0:
                                        prev_op = financial_history['operating_income'][i-1]
                                        growth = ((op_income - prev_op) / prev_op * 100) if prev_op > 0 else 0
                                        st.write(f"• {year}: ${op_income:.1f}B ({growth:+.1f}%)")
                                    else:
                                        st.write(f"• {year}: ${op_income:.1f}B")
                            
                            with col3:
                                st.write('**💰 순이익 추이**')
                                for i, year in enumerate(financial_history['years']):
                                    net_income = financial_history['net_income'][i]
                                    if i > 0:
                                        prev_net = financial_history['net_income'][i-1]
                                        growth = ((net_income - prev_net) / prev_net * 100) if prev_net > 0 else 0
                                        st.write(f"• {year}: ${net_income:.1f}B ({growth:+.1f}%)")
                                    else:
                                        st.write(f"• {year}: ${net_income:.1f}B")
                        else:
                            st.info("3개년 재무 실적 데이터를 가져올 수 없습니다.")
                        
                        # 주요 재무 지표 요약
                        st.subheader('📋 주요 재무 지표 요약')
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            pe_ratio = metrics.get('pe_ratio', 0)
                            pe_color = 'normal' if 10 <= pe_ratio <= 25 else 'inverse' if pe_ratio > 25 else 'off'
                            st.metric('P/E 비율', f"{pe_ratio:.2f}", delta="적정" if pe_color == 'normal' else "높음" if pe_ratio > 25 else "낮음", delta_color=pe_color)
                            
                            roe = metrics.get('roe', 0)*100
                            roe_color = 'normal' if roe > 15 else 'inverse'
                            st.metric('ROE', f"{roe:.1f}%", delta="우수" if roe > 15 else "개선필요", delta_color=roe_color)
                        
                        with col2:
                            pb_ratio = metrics.get('pb_ratio', 0)
                            pb_color = 'normal' if pb_ratio < 3 else 'inverse'
                            st.metric('P/B 비율', f"{pb_ratio:.2f}", delta="적정" if pb_color == 'normal' else "높음", delta_color=pb_color)
                            
                            debt_ratio = metrics.get('debt_to_equity', 0)
                            debt_color = 'normal' if debt_ratio < 0.5 else 'inverse'
                            st.metric('부채비율', f"{debt_ratio:.2f}", delta="안전" if debt_color == 'normal' else "주의", delta_color=debt_color)
                        
                        with col3:
                            profit_margin = metrics.get('profit_margin', 0)*100
                            margin_color = 'normal' if profit_margin > 10 else 'inverse'
                            st.metric('순이익률', f"{profit_margin:.1f}%", delta="우수" if margin_color == 'normal' else "개선필요", delta_color=margin_color)
                            
                            current_ratio = metrics.get('current_ratio', 0)
                            current_color = 'normal' if current_ratio > 1.5 else 'inverse'
                            st.metric('유동비율', f"{current_ratio:.2f}", delta="안전" if current_color == 'normal' else "주의", delta_color=current_color)
                        
                        with col4:
                            market_cap = metrics.get('market_cap', 0)
                            st.metric('시가총액', f"${market_cap/1e9:.1f}B" if market_cap > 0 else "N/A")
                            
                            dividend_yield = metrics.get('dividend_yield', 0)
                            st.metric('배당수익률', f"{dividend_yield:.1f}%" if dividend_yield > 0 else "0.0%")
                        
                        # 상세 재무 정보 (기존 코드 유지)
                        with st.expander('📊 상세 재무 정보'):
                            info = financial_data.get('info', {})
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write('**🏢 기업 정보**')
                                st.write(f"• 기업명: {info.get('longName', 'N/A')}")
                                st.write(f"• 섹터: {info.get('sector', 'N/A')}")
                                st.write(f"• 산업: {info.get('industry', 'N/A')}")
                                st.write(f"• 직원 수: {info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else "• 직원 수: N/A")
                            
                            with col2:
                                st.write('**💰 상세 재무 지표**')
                                fcf = metrics.get('free_cash_flow', 0)
                                st.write(f"• 자유현금흐름: ${fcf/1e9:.2f}B" if fcf > 0 else "• 자유현금흐름: N/A")
                                revenue_growth = metrics.get('revenue_growth', 0)
                                st.write(f"• 매출 성장률: {revenue_growth*100:.1f}%" if revenue_growth else "• 매출 성장률: N/A")
                                st.write(f"• 총 매출: ${info.get('totalRevenue', 0)/1e9:.2f}B" if info.get('totalRevenue') else "• 총 매출: N/A")
                                st.write(f"• 총 현금: ${info.get('totalCash', 0)/1e9:.2f}B" if info.get('totalCash') else "• 총 현금: N/A")
                else:
                    st.error(f'{symbol}의 재무 데이터를 가져올 수 없습니다.')
        
        # 사용법 안내
        with st.expander('💡 GPT 재무 분석 사용법'):
            st.markdown("""
            **🔑 OpenAI API 키 설정**
            1. [OpenAI 홈페이지](https://platform.openai.com/api-keys)에서 API 키 발급
            2. 위의 입력창에 API 키 입력
            3. AI 기반 전문가 수준의 재무 분석 자동 실행
            
            **📊 제공하는 분석**
            - **재무 건전성 종합 평가**: 5점 만점 점수 및 등급
            - **주요 강점 및 약점**: 구체적인 분석과 개선점
            - **업종 대비 경쟁력**: 상대적 위치 평가
            - **투자자 관점 분석**: 장기/단기 투자 시각
            - **재무 개선 과제**: 모니터링 필요 지표
            - **투자 의견**: 매수/보유/매도 추천 및 근거
            
            **🎯 분석 특징**
            - 전문 재무 분석가 수준의 인사이트
            - 3개년 실적 트렌드 반영
            - 업종별 벤치마킹
            - 실용적인 투자 조언
            
            **⚠️ 주의사항**
            - API 키가 없어도 기본 재무 분석은 가능
            - GPT 분석은 OpenAI 사용량에 따라 과금
            - AI 분석은 참고용이며 최종 투자 결정은 본인 책임
            """)
    
    with tab6:
        st.header('🌍 대외 경제지표 분석')
        
        with st.spinner('경제지표 데이터를 가져오는 중...'):
            indicators = EconomicIndicatorAnalyzer.get_economic_indicators()
            
            if indicators:
                # 경제지표 요약 카드
                st.subheader('📊 주요 경제지표 현황')
                
                # 4개씩 2행으로 배치
                cols = st.columns(4)
                indicator_names = {
                    'sp500': 'S&P 500',
                    'nasdaq': '나스닥',
                    'dow': '다우존스',
                    'gold': '금 가격',
                    'oil': '원유 가격',
                    'dollar_index': '달러 인덱스',
                    'us_10yr': '미국 10년 국채',
                    'vix': 'VIX 공포지수'
                }
                
                for i, (key, display_name) in enumerate(indicator_names.items()):
                    if key in indicators:
                        with cols[i % 4]:
                            current_price = indicators[key]['current_price']
                            change_pct = indicators[key]['change_pct']
                            
                            # 단위 설정
                            if key in ['gold', 'oil']:
                                unit = ''
                            elif key in ['us_10yr', 'vix']:
                                unit = '%'
                            else:
                                unit = ''
                            
                            st.metric(
                                display_name,
                                f"{current_price:.2f}{unit}",
                                f"{change_pct:+.2f}%"
                            )
                
                st.markdown('---')
                
                # 차트 생성 및 표시
                charts = create_economic_indicators_dashboard(indicators)
                
                if 'indices' in charts:
                    st.subheader('📈 주요 주식 지수 추이')
                    st.plotly_chart(charts['indices'], use_container_width=True, key="indices_chart")
                
                if 'bonds' in charts:
                    st.subheader('💹 미국 국채 수익률')
                    st.plotly_chart(charts['bonds'], use_container_width=True, key="bonds_chart")
                
                if 'commodities' in charts:
                    st.subheader('🥇 원자재 가격')
                    st.plotly_chart(charts['commodities'], use_container_width=True, key="commodities_chart")
                
                # 추가 지표들
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'dollar_index' in indicators:
                        st.subheader('💵 달러 인덱스')
                        dollar_data = indicators['dollar_index']['data']
                        fig_dollar = go.Figure()
                        fig_dollar.add_trace(go.Scatter(
                            x=dollar_data.index,
                            y=dollar_data['Close'],
                            mode='lines',
                            name='달러 인덱스',
                            line=dict(color='green', width=2)
                        ))
                        fig_dollar.update_layout(
                            xaxis_title='날짜',
                            yaxis_title='지수',
                            height=300
                        )
                        st.plotly_chart(fig_dollar, use_container_width=True, key="dollar_chart")
                
                with col2:
                    if 'vix' in indicators:
                        st.subheader('😰 VIX 공포지수')
                        vix_data = indicators['vix']['data']
                        current_vix = indicators['vix']['current_price']
                        
                        fig_vix = go.Figure()
                        fig_vix.add_trace(go.Scatter(
                            x=vix_data.index,
                            y=vix_data['Close'],
                            mode='lines',
                            name='VIX',
                            line=dict(color='red', width=2)
                        ))
                        
                        # VIX 구간별 색상 표시
                        fig_vix.add_hline(y=30, line_dash="dash", line_color="red", 
                                         annotation_text="공포 구간 (30)")
                        fig_vix.add_hline(y=20, line_dash="dash", line_color="orange", 
                                         annotation_text="긴장 구간 (20)")
                        fig_vix.add_hline(y=12, line_dash="dash", line_color="green", 
                                         annotation_text="안정 구간 (12)")
                        
                        fig_vix.update_layout(
                            xaxis_title='날짜',
                            yaxis_title='변동성 (%)',
                            height=300
                        )
                        st.plotly_chart(fig_vix, use_container_width=True, key="vix_chart")
                        
                        # VIX 해석 표시
                        vix_interpretation = EconomicIndicatorAnalyzer.interpret_vix(current_vix)
                        
                        if vix_interpretation['color'] == 'error':
                            st.error(f"""
                            **{vix_interpretation['emoji']} {vix_interpretation['level']}**: {vix_interpretation['message']}
                            
                            💡 **투자 조언**: {vix_interpretation['advice']}
                            """)
                        elif vix_interpretation['color'] == 'warning':
                            st.warning(f"""
                            **{vix_interpretation['emoji']} {vix_interpretation['level']}**: {vix_interpretation['message']}
                            
                            💡 **투자 조언**: {vix_interpretation['advice']}
                            """)
                        else:
                            st.success(f"""
                            **{vix_interpretation['emoji']} {vix_interpretation['level']}**: {vix_interpretation['message']}
                            
                            💡 **투자 조언**: {vix_interpretation['advice']}
                            """)
                
                # 경제지표 해석 가이드
                with st.expander('📚 경제지표 해석 가이드'):
                    st.markdown("""
                    **주식 지수**
                    - **S&P 500**: 미국 대형주 500개 기업의 시가총액 가중 지수
                    - **나스닥**: 기술주 중심의 지수, 성장주 투자 기준
                    - **다우존스**: 미국 우량주 30개 기업의 주가 평균
                    
                    **채권 수익률**
                    - **10년 국채**: 장기 금리 기준, 경제 전망 반영
                    - **2년 국채**: 단기 금리 기준, 연준 정책 반영
                    
                    **원자재**
                    - **금**: 안전자산, 인플레이션 헤지 수단
                    - **원유**: 경기 선행지표, 에너지 비용 기준
                    
                    **기타 지표**
                    - **달러 인덱스**: 달러 강세/약세 측정
                    - **VIX**: 시장 변동성과 투자자 불안감 측정
                    
                    **📊 VIX 공포지수 상세 해석**
                    
                    - **🚨 50+ (극도 공포)**: 패닉 상태, 역설적 매수 기회
                    - **😰 30-50 (높은 공포)**: 시장 불안정, 주의 깊은 투자
                    - **⚠️ 20-30 (보통 긴장)**: 정상적인 변동성 수준
                    - **😌 12-20 (안정)**: 정상적인 투자 환경
                    - **😴 12 미만 (극도 안정)**: 과도한 낙관론, 주의 필요
                    
                    **워렌 버핏 관점에서 VIX 활용법**
                    - VIX 높음 = "남들이 두려워할 때" = 매수 기회
                    - VIX 낮음 = "남들이 욕심낼 때" = 신중해야 할 때
                    """)
            else:
                st.error('경제지표 데이터를 가져올 수 없습니다.')

        st.markdown('---')

        # GPT 기반 종합 경제 분석
        st.subheader('🤖 AI 기반 종합 경제환경 분석')

        # API 키 입력
        col1, col2 = st.columns([3, 1])

        with col1:
            economic_openai_api_key = st.text_input(
                "OpenAI API 키 (GPT 기반 종합 경제 분석을 위해 필요)",
                type="password",
                help="GPT를 이용한 전문가 수준의 경제환경 분석을 원하시면 OpenAI API 키를 입력하세요.",
                key="economic_analysis_api_key"
            )

        with col2:
            if economic_openai_api_key:
                if economic_openai_api_key.startswith('sk-') and len(economic_openai_api_key) > 20:
                    st.success("✅ API 키 활성화")
                else:
                    st.error("❌ 잘못된 API 키")

        # 분석 시작 버튼
        if economic_openai_api_key and indicators:
            if st.button("🚀 종합 경제환경 분석 시작", use_container_width=True, type="primary"):
                with st.spinner('📊 GPT가 모든 경제지표를 종합 분석하는 중...'):
                    gpt_economic_analysis = EconomicIndicatorAnalyzer.analyze_economic_indicators_with_gpt(
                        indicators, economic_openai_api_key
                    )
                
                if gpt_economic_analysis.get('error'):
                    st.error(f"🤖 분석 오류: {gpt_economic_analysis['error']}")
                else:
                    # 분석 결과 표시
                    score = gpt_economic_analysis.get('score', 3.0)
                    environment = gpt_economic_analysis.get('environment', '중립적')
                    env_color = gpt_economic_analysis.get('environment_color', 'warning')
                    
                    # 종합 점수 및 환경 표시
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "경제환경 점수",
                            f"{score:.1f}/5.0",
                            delta=f"{(score/5.0)*100:.0f}%"
                        )
                    
                    with col2:
                        if env_color == 'success':
                            st.success(f"📈 {environment}")
                        elif env_color == 'warning':
                            st.warning(f"📊 {environment}")
                        else:
                            st.error(f"📉 {environment}")
                    
                    with col3:
                        # 상태에 따른 투자 가이드
                        if score >= 4.0:
                            st.success("🚀 공격적 투자 환경")
                        elif score >= 3.0:
                            st.info("⚖️ 균형잡힌 투자")
                        else:
                            st.error("🛡️ 방어적 투자 환경")
                    
                    # GPT 분석 내용 표시
                    st.markdown("### 📋 종합 경제환경 분석 리포트")
                    st.markdown(gpt_economic_analysis['analysis'])

        elif not economic_openai_api_key:
            st.info("""
            🔑 **OpenAI API 키를 입력하시면 다음과 같은 고급 분석을 받으실 수 있습니다:**
            
            • **종합 경제 상황 진단**: 모든 지표를 종합한 5점 만점 점수
            • **투자 환경 분류**: Risk-On/Risk-Off 판단
            • **섹터별 영향 분석**: 어떤 섹터가 유리/불리한지
            • **자산배분 가이드**: 현재 상황에 맞는 투자 전략
            • **리스크 요인 진단**: 주의해야 할 위험 요소들
            • **단기 전망**: 1-3개월 시장 시나리오
            """)

        elif not indicators:
            st.warning("경제지표 데이터를 먼저 불러와야 분석이 가능합니다.")

        # 경제지표 활용 가이드
        with st.expander('💡 GPT 경제분석 활용법'):
            st.markdown("""
            **🎯 제공하는 분석**
            - **종합 점수**: 모든 지표를 고려한 5점 만점 점수
            - **투자 환경**: Risk-On/Risk-Off 분류
            - **자산배분 가이드**: 주식/채권/원자재/현금 비중 조언
            - **리스크 진단**: 주요 위험 요인 식별
            - **섹터 전망**: 유리한 섹터와 불리한 섹터
            - **단기 전망**: 1-3개월 시장 시나리오
            
            **📊 분석 기준**
            - 주식 지수: 시장 심리 및 위험선호도
            - 국채 수익률: 금리 환경 및 경기 전망
            - 원자재: 인플레이션 압력 및 실물경기
            - 달러/VIX: 글로벌 유동성 및 리스크
            
            **🚀 활용 방법**
            1. API 키 입력 후 분석 시작
            2. 종합 점수와 환경 분류 확인
            3. 자산배분 가이드 참고
            4. 리스크 요인 모니터링
            5. 개별 종목 분석과 결합하여 투자 결정
            
            **⚠️ 주의사항**
            - AI 분석은 참고용이며 투자 보장하지 않음
            - 급변하는 시장 상황 실시간 반영 한계
            - 다른 분석과 함께 종합적으로 판단 필요
            - 개인 투자 성향과 목표 고려 필수
            """)
    
    with tab7:
        st.header('🧠 워렌 버핏 스타일 투자 조언')
        
        if not symbols:
            st.warning('종목 코드를 입력해주세요.')
            return
        
        for symbol in symbols:
            st.subheader(f'{symbol} 버핏 스타일 분석')
            
            with st.spinner(f'{symbol}을 워렌 버핏의 관점에서 분석하는 중...'):
                # 데이터 수집
                financial_data = DataManager.get_financial_data(symbol)
                sentiment_data = SentimentAnalyzer.get_news_sentiment(symbol)
                
                if financial_data:
                    metrics = FinancialAnalyzer.calculate_financial_metrics(financial_data)
                    analysis = BuffettAnalyzer.buffett_analysis(metrics, sentiment_data)
                    
                    if analysis:
                        # 투자 등급 표시
                        grade_class = analysis.get('grade_color', 'hold-signal')
                        st.markdown(f"""
                        <div class="{grade_class}">
                            <h3>투자 등급: {analysis.get('grade', 'N/A')}</h3>
                            <p>{analysis.get('recommendation', '')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 점수 표시
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                '버핏 점수',
                                f"{analysis.get('score', 0)}/{analysis.get('max_score', 100)}",
                                delta=f"{analysis.get('score', 0)}점"
                            )
                        
                        with col2:
                            percentage = (analysis.get('score', 0) / analysis.get('max_score', 100)) * 100
                            st.metric(
                                '적합도',
                                f"{percentage:.1f}%"
                            )
                        
                        with col3:
                            grade = analysis.get('grade', 'HOLD')
                            if grade == 'BUY':
                                st.success('🚀 매수')
                            elif grade == 'SELL':
                                st.error('⚠️ 매도')
                            else:
                                st.warning('📊 보유')
                        
                        # 분석 근거
                        st.subheader('📋 분석 근거')
                        reasons = analysis.get('reasons', [])
                        
                        for reason in reasons:
                            if '✅' in reason:
                                st.success(reason)
                            elif '⚠️' in reason:
                                st.warning(reason)
                            elif '❌' in reason:
                                st.error(reason)
                            else:
                                st.info(reason)
                        
                        # 워렌 버핏의 투자 철학
                        with st.expander('📚 워렌 버핏의 투자 철학'):
                            st.markdown("""
                            **워렌 버핏의 핵심 투자 원칙:**
                            
                            1. **사업 이해하기**: 자신이 이해할 수 있는 사업에만 투자
                            2. **경제적 해자**: 경쟁우위를 가진 기업 선호
                            3. **우수한 경영진**: 주주 가치를 중시하는 경영진
                            4. **합리적 가격**: 내재가치 대비 저평가된 주식
                            5. **장기 투자**: "영원히 보유할 수 있는 주식"
                            
                            **주요 재무 지표:**
                            - ROE > 15%
                            - 꾸준한 자유현금흐름
                            - 낮은 부채비율
                            - 안정적인 수익성
                            - 적정한 P/E 비율 (10-20)
                            """)
                        
                        # 리스크 요인
                        st.subheader('⚠️ 투자 리스크')
                        st.warning("""
                        **주요 리스크 요인:**
                        - 시장 변동성에 따른 주가 하락 위험
                        - 기업 실적 악화 가능성
                        - 산업 전반의 구조적 변화
                        - 거시경제 악화 영향
                        - 예측 모델의 한계
                        
                        **투자 전 고려사항:**
                        - 개인 투자 목표 및 위험 성향 확인
                        - 충분한 추가 조사 및 분석 필요
                        - 분산 투자를 통한 리스크 관리
                        - 정기적인 포트폴리오 리뷰
                        """)
                else:
                    st.error(f'{symbol}의 재무 데이터를 가져올 수 없습니다.')

    # 유튜브 분석 탭 개선 코드 (tab8 부분만)
    with tab8:
        st.header('📺 유튜브 영상 분석 (향상된 버전)')
        
        # 세션 상태 초기화 - 더 안정적인 방법
        if 'selected_videos' not in st.session_state:
            st.session_state.selected_videos = {}
        if 'video_summaries' not in st.session_state:
            st.session_state.video_summaries = {}
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        if 'last_search_query' not in st.session_state:
            st.session_state.last_search_query = ""
        if 'filtered_results' not in st.session_state:
            st.session_state.filtered_results = []
        
        # OpenAI API 키 입력
        st.subheader('🔧 GPT 요약 설정 (선택사항)')
        openai_api_key = st.text_input(
            "OpenAI API 키 (영상 요약을 위해 필요)",
            type="password",
            help="GPT를 이용한 영상 요약을 원하시면 OpenAI API 키를 입력하세요.",
            key="youtube_api_key"
        )
        
        # 검색 섹션
        st.subheader('🔍 유튜브 검색')
        
        # 검색 설정
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 검색 폼 사용하여 자동 재로드 방지
            with st.form("youtube_search_form"):
                search_query = st.text_input(
                    "검색어를 입력하세요",
                    value=st.session_state.last_search_query,
                    placeholder="예: AAPL 주식 분석, 애플 투자 전망, 테슬라 실적 분석",
                    help="주식 종목명이나 투자 관련 키워드를 입력하세요"
                )
                
                # 검색 옵션
                search_col1, search_col2 = st.columns(2)
                
                with search_col1:
                    max_results = st.selectbox(
                        "검색 결과 수",
                        [20, 30, 50, 100],
                        index=1,
                        help="더 많은 결과를 원하시면 큰 숫자를 선택하세요"
                    )
                
                with search_col2:
                    initial_sort = st.selectbox(
                        "초기 정렬",
                        ["관련도", "최신순", "조회수순", "평점순"],
                        help="검색 시 적용할 기본 정렬 방식"
                    )
                
                search_button = st.form_submit_button("🔍 검색", use_container_width=True)
        
        with col2:
            st.info("""
            **🆕 새로운 기능:**
            • 검색 결과 최대 100개
            • 조회수/업로드일/재생시간 필터
            • 다양한 정렬 옵션
            • 실시간 필터링
            """)
        
        # 정렬 옵션 매핑
        sort_mapping = {
            "관련도": "relevance",
            "최신순": "upload_date", 
            "조회수순": "view_count",
            "평점순": "rating"
        }
        
        # 검색 실행
        if search_button and search_query:
            if search_query != st.session_state.last_search_query:
                with st.spinner('유튜브 영상을 검색하는 중...'):
                    videos = YouTubeAnalyzer.search_youtube_videos(
                        search_query, 
                        max_results=max_results,
                        sort_order=sort_mapping[initial_sort]
                    )
                    st.session_state.search_results = videos
                    st.session_state.filtered_results = videos  # 초기엔 필터링 안 함
                    st.session_state.last_search_query = search_query
                st.success(f"✅ '{search_query}' 검색 완료! {len(videos)}개 영상 발견")
        
        # 필터 및 정렬 섹션
        if st.session_state.search_results:
            st.markdown("---")
            st.subheader('🎛️ 고급 필터 및 정렬')
            
            # 필터 컨트롤
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
            
            with filter_col1:
                st.write("**📊 조회수 필터**")
                min_views = st.selectbox(
                    "최소 조회수",
                    [0, 1000, 10000, 100000, 1000000],
                    format_func=lambda x: f"{x:,}회" if x > 0 else "제한 없음"
                )
            
            with filter_col2:
                st.write("**📅 업로드 기간**")
                max_days = st.selectbox(
                    "최대 업로드 전",
                    [999999, 1, 7, 30, 90, 365],
                    index=0,
                    format_func=lambda x: "전체 기간" if x == 999999 else f"{x}일 전"
                )
            
            with filter_col3:
                st.write("**⏱️ 재생시간 범위**")
                duration_range = st.selectbox(
                    "영상 길이",
                    ["전체", "짧음 (4분 이하)", "보통 (4-20분)", "김 (20분 이상)"]
                )
                
                # 재생시간 범위를 초로 변환
                if duration_range == "짧음 (4분 이하)":
                    min_dur, max_dur = 0, 240
                elif duration_range == "보통 (4-20분)":
                    min_dur, max_dur = 240, 1200
                elif duration_range == "김 (20분 이상)":
                    min_dur, max_dur = 1200, 999999
                else:
                    min_dur, max_dur = 0, 999999
            
            with filter_col4:
                st.write("**🔄 정렬 방식**")
                sort_by = st.selectbox(
                    "정렬 기준",
                    ["관련도", "조회수순", "최신순", "재생시간순"],
                    help="필터링된 결과를 다시 정렬합니다"
                )
                
                sort_mapping_filter = {
                    "관련도": "relevance",
                    "조회수순": "view_count",
                    "최신순": "upload_date", 
                    "재생시간순": "duration"
                }
            
            # 필터 적용 버튼
            if st.button("🎯 필터 적용", use_container_width=True):
                with st.spinner("필터링 중..."):
                    filtered_videos = YouTubeAnalyzer.filter_videos(
                        st.session_state.search_results,
                        min_views=min_views,
                        max_days_ago=max_days,
                        min_duration=min_dur,
                        max_duration=max_dur,
                        sort_by=sort_mapping_filter[sort_by]
                    )
                    st.session_state.filtered_results = filtered_videos
                
                st.success(f"✅ 필터 적용 완료! {len(st.session_state.filtered_results)}개 영상")
            
            # 필터 상태 표시
            if st.session_state.filtered_results:
                original_count = len(st.session_state.search_results)
                filtered_count = len(st.session_state.filtered_results)
                
                st.info(f"📊 **필터 결과**: {filtered_count}개 영상 (전체 {original_count}개 중)")
                
                # 현재 필터 조건 표시
                active_filters = []
                if min_views > 0:
                    active_filters.append(f"조회수 {min_views:,}회 이상")
                if max_days < 999999:
                    active_filters.append(f"{max_days}일 이내")
                if duration_range != "전체":
                    active_filters.append(f"재생시간 {duration_range}")
                if sort_by != "관련도":
                    active_filters.append(f"{sort_by} 정렬")
                
                if active_filters:
                    st.caption(f"🎛️ 활성 필터: {' | '.join(active_filters)}")
            
            # 검색 결과 표시 (필터링된 결과 사용)
            videos = st.session_state.filtered_results if st.session_state.filtered_results else st.session_state.search_results
            
            st.markdown("---")
            st.subheader(f'📹 "{st.session_state.last_search_query}" 검색 결과 ({len(videos)}개)')
            
            if not videos:
                st.warning("😅 필터 조건에 맞는 영상이 없습니다. 필터를 조정해보세요.")
            else:
                # 페이지네이션 (한 페이지에 9개씩)
                videos_per_page = 9
                total_pages = (len(videos) - 1) // videos_per_page + 1
                
                if total_pages > 1:
                    page_col1, page_col2, page_col3 = st.columns([1, 1, 1])
                    
                    with page_col2:
                        current_page = st.selectbox(
                            f"페이지 ({total_pages}페이지 중)",
                            range(1, total_pages + 1),
                            key="video_page_selector"
                        )
                    
                    start_idx = (current_page - 1) * videos_per_page
                    end_idx = start_idx + videos_per_page
                    page_videos = videos[start_idx:end_idx]
                else:
                    page_videos = videos[:videos_per_page]
                
                # 그리드 형태로 영상 표시 (3열)
                for i in range(0, len(page_videos), 3):
                    cols = st.columns(3)
                    
                    for j, col in enumerate(cols):
                        if i + j < len(page_videos):
                            video = page_videos[i + j]
                            
                            with col:
                                # 컨테이너로 감싸서 안정성 향상
                                with st.container():
                                    # 썸네일 표시
                                    try:
                                        st.image(video['thumbnail_url'], use_container_width=True)
                                    except:
                                        st.error("썸네일 로드 실패")
                                    
                                    # 영상 정보 - 더 상세하게
                                    st.markdown(f"**{video['title'][:45]}{'...' if len(video['title']) > 45 else ''}**")
                                    st.markdown(f"📺 {video['channel_name']}")
                                    
                                    # 상세 정보를 컬럼으로 정리
                                    info_col1, info_col2 = st.columns(2)
                                    with info_col1:
                                        st.markdown(f"👀 {video['view_count']}")
                                        st.markdown(f"⏱️ {video['duration']}")
                                    with info_col2:
                                        st.markdown(f"📅 {video['published_time']}")
                                        # 추가 통계 정보
                                        if video['view_count_num'] >= 1000000:
                                            st.markdown("🔥 **인기 영상**")
                                        elif video['published_days_ago'] <= 7:
                                            st.markdown("🆕 **최신 영상**")
                                    
                                    # 영상 링크
                                    st.markdown(f"🔗 [영상 보기]({video['video_url']})")
                                    
                                    # 이미 선택된 영상인지 확인
                                    if video['video_id'] in st.session_state.selected_videos:
                                        st.success("✅ 이미 분석 목록에 추가됨")
                                    else:
                                        # 영상 추가 버튼 - 콜백 대신 세션 상태 직접 조작
                                        button_key = f"add_video_{video['video_id']}_{i}_{j}_{current_page if 'current_page' in locals() else 1}"
                                        if st.button(f"📝 분석 추가", key=button_key, use_container_width=True):
                                            # 세션 상태에 영상 추가
                                            st.session_state.selected_videos[video['video_id']] = video
                                            st.success(f"✅ '{video['title'][:30]}...' 분석 목록에 추가됨!")
                                            # 페이지 새로고침을 위한 rerun 호출
                                            st.rerun()
                                    
                                    st.markdown("---")
        
        # 선택된 영상들 표시 및 요약
        if st.session_state.selected_videos:
            st.markdown("---")
            st.subheader('📋 선택된 영상 분석')
            
            # 선택된 영상 개수 표시
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("선택된 영상", f"{len(st.session_state.selected_videos)}개")
            
            with col2:
                analyzed_count = len([k for k in st.session_state.video_summaries.keys() if k.startswith('summary_')])
                st.metric("분석 완료", f"{analyzed_count}개")
            
            with col3:
                pending_count = len(st.session_state.selected_videos) - analyzed_count
                st.metric("분석 대기", f"{pending_count}개")
            
            # 선택된 영상 목록
            video_ids = list(st.session_state.selected_videos.keys())
            
            for idx, video_id in enumerate(video_ids):
                video = st.session_state.selected_videos[video_id]
                
                # 각 영상을 expandable 섹션으로 만들어 관리 용이성 향상
                with st.expander(f"📺 {idx+1}. {video['title'][:60]}{'...' if len(video['title']) > 60 else ''}", expanded=True):
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        # 영상 정보
                        try:
                            st.image(video['thumbnail_url'], width=200)
                        except:
                            st.error("썸네일 로드 실패")
                        
                        st.markdown(f"**채널**: {video['channel_name']}")
                        st.markdown(f"**조회수**: {video['view_count']}")
                        st.markdown(f"**길이**: {video['duration']}")
                        st.markdown(f"**업로드**: {video['published_time']}")
                        
                        # 영상 품질 지표
                        if video.get('view_count_num', 0) >= 1000000:
                            st.success("🔥 인기 영상")
                        elif video.get('view_count_num', 0) >= 100000:
                            st.info("📈 조회수 양호")
                        
                        if video.get('published_days_ago', 999) <= 7:
                            st.success("🆕 최신 영상")
                        elif video.get('published_days_ago', 999) <= 30:
                            st.info("📅 최근 영상")
                        
                        st.markdown(f"🔗 [원본 영상]({video['video_url']})")
                    
                    with col2:
                        # 요약 결과 표시
                        summary_key = f"summary_{video_id}"
                        
                        if summary_key in st.session_state.video_summaries:
                            # 이미 요약된 결과 표시
                            summary_data = st.session_state.video_summaries[summary_key]
                            
                            if summary_data['type'] == 'gpt_summary':
                                st.subheader("🤖 AI 요약 분석")
                                if summary_data.get('error'):
                                    st.error(f"요약 실패: {summary_data['error']}")
                                else:
                                    st.markdown(summary_data['content'])
                                    
                                    # 요약의 품질 평가
                                    if len(summary_data['content']) > 500:
                                        st.success("📊 상세한 분석 완료")
                                    else:
                                        st.info("📝 기본 요약 완료")
                                        
                            elif summary_data['type'] == 'error':
                                st.error(f"❌ 분석 실패: {summary_data.get('error', '알 수 없는 오류')}")
                                if summary_data.get('content'):
                                    st.text_area("부분 내용", summary_data['content'], height=150, 
                                            key=f"error_content_{video_id}", disabled=True)
                            else:
                                st.subheader("📝 영상 내용 (일부)")
                                st.text_area("자막 내용", summary_data['content'], height=200, 
                                        key=f"transcript_display_{video_id}", disabled=True)
                                
                                if not summary_data.get('had_api_key'):
                                    st.info("💡 OpenAI API 키를 입력하시면 AI 기반 요약 분석을 받으실 수 있습니다.")
                        else:
                            # 아직 요약되지 않음
                            st.info("🔄 요약을 시작하려면 옆의 '🚀 요약 시작' 버튼을 클릭하세요.")
                            
                            # 영상 예상 분석 시간 표시
                            duration_seconds = video.get('duration_seconds', 0)
                            if duration_seconds > 1800:  # 30분 이상
                                st.warning("⏰ 긴 영상입니다. 분석에 시간이 걸릴 수 있습니다.")
                            elif duration_seconds > 0:
                                st.info(f"⏱️ 예상 분석 시간: 약 {duration_seconds//60 + 1}분")
                    
                    with col3:
                        # 액션 버튼들을 form으로 감싸서 안정성 향상
                        if f"summary_{video_id}" not in st.session_state.video_summaries:
                            # 요약 시작 버튼
                            with st.form(f"analyze_form_{video_id}"):
                                analyze_button = st.form_submit_button(f"🚀 요약 시작", use_container_width=True, type="primary")
                                
                                if analyze_button:
                                    with st.spinner('영상을 분석하는 중...'):
                                        try:
                                            # 자막 추출
                                            transcript = YouTubeAnalyzer.get_video_transcript(video_id)
                                            
                                            if openai_api_key and transcript and len(transcript.strip()) > 50:
                                                # GPT 요약
                                                summary_result = YouTubeAnalyzer.summarize_video_with_gpt(
                                                    transcript, video['title'], openai_api_key
                                                )
                                                
                                                if summary_result.get('error'):
                                                    st.session_state.video_summaries[f"summary_{video_id}"] = {
                                                        'type': 'error',
                                                        'content': transcript[:1000] + "..." if len(transcript) > 1000 else transcript,
                                                        'error': summary_result['error'],
                                                        'had_api_key': True
                                                    }
                                                else:
                                                    st.session_state.video_summaries[f"summary_{video_id}"] = {
                                                        'type': 'gpt_summary',
                                                        'content': summary_result['summary'],
                                                        'had_api_key': True
                                                    }
                                            else:
                                                # API 키가 없거나 자막 추출 실패
                                                content = transcript[:1000] + "..." if transcript and len(transcript) > 1000 else transcript
                                                if not content or len(content.strip()) < 10:
                                                    content = "자막을 가져올 수 없습니다. 이 영상은 자막이 제공되지 않거나 접근할 수 없는 상태입니다."
                                                
                                                st.session_state.video_summaries[f"summary_{video_id}"] = {
                                                    'type': 'basic',
                                                    'content': content,
                                                    'had_api_key': bool(openai_api_key)
                                                }
                                            
                                            st.success("✅ 요약 완료!")
                                            
                                        except Exception as e:
                                            st.session_state.video_summaries[f"summary_{video_id}"] = {
                                                'type': 'error',
                                                'content': '',
                                                'error': f"분석 중 오류 발생: {str(e)}",
                                                'had_api_key': bool(openai_api_key)
                                            }
                                            st.error(f"❌ 분석 실패: {str(e)}")
                                        
                                        # 상태 변경 후 페이지 새로고침
                                        st.rerun()
                        else:
                            # 이미 요약됨 - 새로고침 버튼
                            with st.form(f"refresh_form_{video_id}"):
                                refresh_button = st.form_submit_button(f"🔄 다시 요약", use_container_width=True)
                                
                                if refresh_button:
                                    # 기존 요약 삭제 후 다시 요약
                                    if f"summary_{video_id}" in st.session_state.video_summaries:
                                        del st.session_state.video_summaries[f"summary_{video_id}"]
                                    st.info("🔄 요약이 초기화되었습니다. '🚀 요약 시작' 버튼을 다시 클릭하세요.")
                                    st.rerun()
                        
                        st.markdown("---")
                        
                        # 삭제 버튼
                        with st.form(f"remove_form_{video_id}"):
                            remove_button = st.form_submit_button(f"🗑️ 목록에서 제거", use_container_width=True, type="secondary")
                            
                            if remove_button:
                                # 영상과 요약 모두 삭제
                                if video_id in st.session_state.selected_videos:
                                    del st.session_state.selected_videos[video_id]
                                if f"summary_{video_id}" in st.session_state.video_summaries:
                                    del st.session_state.video_summaries[f"summary_{video_id}"]
                                st.success("✅ 목록에서 제거되었습니다!")
                                st.rerun()
            
            # 일괄 작업 버튼들
            st.markdown("---")
            st.subheader("🔧 일괄 작업")
            
            batch_col1, batch_col2, batch_col3 = st.columns(3)
            
            with batch_col1:
                with st.form("analyze_all_form"):
                    analyze_all_button = st.form_submit_button("🚀 전체 영상 일괄 분석", use_container_width=True, type="primary")
                    
                    if analyze_all_button:
                        if not openai_api_key:
                            st.error("⚠️ 일괄 분석을 위해서는 OpenAI API 키가 필요합니다.")
                        else:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            unanalyzed_videos = [vid for vid in st.session_state.selected_videos.keys() 
                                            if f"summary_{vid}" not in st.session_state.video_summaries]
                            
                            for i, video_id in enumerate(unanalyzed_videos):
                                video = st.session_state.selected_videos[video_id]
                                status_text.text(f"분석 중: {video['title'][:30]}... ({i+1}/{len(unanalyzed_videos)})")
                                
                                try:
                                    transcript = YouTubeAnalyzer.get_video_transcript(video_id)
                                    if transcript and len(transcript.strip()) > 50:
                                        summary_result = YouTubeAnalyzer.summarize_video_with_gpt(
                                            transcript, video['title'], openai_api_key
                                        )
                                        
                                        if summary_result.get('error'):
                                            st.session_state.video_summaries[f"summary_{video_id}"] = {
                                                'type': 'error',
                                                'content': transcript[:1000] + "..." if len(transcript) > 1000 else transcript,
                                                'error': summary_result['error'],
                                                'had_api_key': True
                                            }
                                        else:
                                            st.session_state.video_summaries[f"summary_{video_id}"] = {
                                                'type': 'gpt_summary',
                                                'content': summary_result['summary'],
                                                'had_api_key': True
                                            }
                                    else:
                                        st.session_state.video_summaries[f"summary_{video_id}"] = {
                                            'type': 'basic',
                                            'content': "자막을 가져올 수 없습니다.",
                                            'had_api_key': True
                                        }
                                except Exception as e:
                                    st.session_state.video_summaries[f"summary_{video_id}"] = {
                                        'type': 'error',
                                        'content': '',
                                        'error': f"분석 중 오류: {str(e)}",
                                        'had_api_key': True
                                    }
                                
                                progress_bar.progress((i + 1) / len(unanalyzed_videos))
                            
                            progress_bar.empty()
                            status_text.empty()
                            st.success(f"✅ {len(unanalyzed_videos)}개 영상 일괄 분석 완료!")
                            st.rerun()
            
            with batch_col2:
                # 내보내기 준비 버튼 (form 안에)
                with st.form("export_form"):
                    export_button = st.form_submit_button("📤 분석 결과 준비", use_container_width=True)
                    
                    if export_button:
                        # 세션 상태에 내보내기 데이터 저장
                        from datetime import datetime
                        export_text = f"# 유튜브 영상 분석 결과\n\n검색어: {st.session_state.last_search_query}\n분석일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        
                        for idx, (video_id, video) in enumerate(st.session_state.selected_videos.items()):
                            export_text += f"## {idx+1}. {video['title']}\n\n"
                            export_text += f"- **채널**: {video['channel_name']}\n"
                            export_text += f"- **조회수**: {video['view_count']}\n"
                            export_text += f"- **길이**: {video['duration']}\n"
                            export_text += f"- **업로드**: {video['published_time']}\n"
                            export_text += f"- **링크**: {video['video_url']}\n\n"
                            
                            summary_key = f"summary_{video_id}"
                            if summary_key in st.session_state.video_summaries:
                                summary_data = st.session_state.video_summaries[summary_key]
                                export_text += f"**분석 결과:**\n{summary_data.get('content', '분석 실패')}\n\n"
                            else:
                                export_text += "**분석 결과:** 분석되지 않음\n\n"
                            
                            export_text += "---\n\n"
                        
                        # 세션 상태에 저장
                        st.session_state.export_ready = True
                        st.session_state.export_data = export_text
                        st.success("✅ 내보내기 파일이 준비되었습니다!")
                
                # 다운로드 버튼 (form 밖에)
                if st.session_state.get('export_ready', False) and st.session_state.get('export_data'):
                    from datetime import datetime
                    st.download_button(
                        label="📥 마크다운 파일 다운로드",
                        data=st.session_state.export_data,
                        file_name=f"youtube_analysis_{st.session_state.last_search_query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
            
            with batch_col3:
                with st.form("clear_all_form"):
                    clear_all_button = st.form_submit_button("🗑️ 전체 목록 초기화", type="secondary", use_container_width=True)
                    
                    if clear_all_button:
                        st.session_state.selected_videos = {}
                        st.session_state.video_summaries = {}
                        st.session_state.search_results = []
                        st.session_state.filtered_results = []
                        st.session_state.last_search_query = ""
                        st.success("✅ 전체 목록이 초기화되었습니다!")
                        st.rerun()

        else:
            st.info("📝 위에서 영상을 검색하고 '📝 분석 추가' 버튼을 눌러 분석할 영상을 선택하세요.")
        
        # 통계 정보 표시
        if st.session_state.search_results or st.session_state.selected_videos:
            st.markdown("---")
            st.subheader("📊 세션 통계")
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.metric("검색된 영상", len(st.session_state.search_results))
            
            with stat_col2:
                st.metric("필터링된 영상", len(st.session_state.filtered_results))
            
            with stat_col3:
                st.metric("선택된 영상", len(st.session_state.selected_videos))
            
            with stat_col4:
                analyzed_count = len([k for k in st.session_state.video_summaries.keys() if k.startswith('summary_')])
                st.metric("분석 완료", analyzed_count)
        
        # 현재 세션 상태 디버깅 정보 (개발용)
        if st.checkbox("🔧 디버깅 정보 표시", help="개발자용 세션 상태 정보"):
            st.write("**현재 세션 상태:**")
            st.write(f"- 검색 결과 수: {len(st.session_state.search_results)}")
            st.write(f"- 필터링된 결과 수: {len(st.session_state.filtered_results)}")
            st.write(f"- 선택된 영상 수: {len(st.session_state.selected_videos)}")
            st.write(f"- 요약된 영상 수: {len(st.session_state.video_summaries)}")
            st.write(f"- 마지막 검색어: {st.session_state.last_search_query}")
            
            if st.button("🔄 세션 상태 강제 초기화 (디버깅용)"):
                for key in ['selected_videos', 'video_summaries', 'search_results', 'filtered_results', 'last_search_query']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("세션 상태가 강제로 초기화되었습니다.")
                st.rerun()
        
        # 사용법 안내
        with st.expander('💡 유튜브 분석 사용법 (향상된 기능)'):
            st.markdown("""
            **🆕 새로운 기능**
            - **대용량 검색**: 최대 100개 영상 검색 가능
            - **고급 필터링**: 조회수, 업로드일, 재생시간으로 필터링
            - **다양한 정렬**: 관련도, 조회수, 최신순, 재생시간순 정렬
            - **일괄 분석**: 선택된 모든 영상을 한 번에 분석
            - **결과 내보내기**: 분석 결과를 마크다운 파일로 다운로드
            - **페이지네이션**: 많은 검색 결과를 페이지로 나누어 표시
            - **통계 대시보드**: 세션별 분석 현황 표시
            
            **🔍 효과적인 검색 방법**
            - **종목 중심**: "AAPL stock analysis", "Tesla earnings review"
            - **시점 중심**: "2024 Q4 earnings", "latest market update"
            - **분석 유형**: "technical analysis", "fundamental analysis"
            - **전망 중심**: "price prediction", "investment outlook"
            
            **🎛️ 필터링 활용법**
            - **인기 영상 찾기**: 조회수 100,000회 이상 필터
            - **최신 정보**: 7일 이내 업로드 필터
            - **심층 분석**: 20분 이상 긴 영상 필터
            - **빠른 정보**: 4분 이하 짧은 영상 필터
            
            **🚀 분석 효율성 팁**
            1. **검색 → 필터링 → 선택 → 일괄 분석** 순서로 진행
            2. 관심 있는 영상만 선별해서 API 비용 절약
            3. 분석 결과는 마크다운으로 내보내서 보관
            4. 여러 키워드로 검색해서 다양한 관점 수집
            
            **⚠️ 주의사항**
            - 유튜브 검색 제한으로 모든 영상을 가져오지 못할 수 있음
            - 일괄 분석은 API 사용량이 많으니 신중하게 사용
            - 영상 자막이 없는 경우 분석이 제한적일 수 있음
            - 긴 영상일수록 분석 시간이 오래 걸림
            
            **💡 고급 활용법**
            - **경쟁 분석**: 같은 종목에 대한 여러 채널 의견 비교
            - **시점별 분석**: 실적 발표 전후 영상들 비교
            - **채널별 특성**: 특정 채널의 분석 패턴 파악
            - **키워드 트렌드**: 인기 검색어 변화 추적
            """)

    with tab9:
        st.header('📈 구글 검색 트렌드 분석')
        
        # 세션 상태 초기화 - 한 번만 실행
        if 'trends_data_cache' not in st.session_state:
            st.session_state.trends_data_cache = None
        if 'trends_keywords_cache' not in st.session_state:
            st.session_state.trends_keywords_cache = ""
        if 'trends_analysis_done' not in st.session_state:
            st.session_state.trends_analysis_done = False
        if 'gpt_trends_result' not in st.session_state:
            st.session_state.gpt_trends_result = None
        
        # 안내 메시지
        st.info("""
        📊 **구글 트렌드 분석이란?**
        
        구글에서 특정 키워드가 얼마나 많이 검색되었는지를 시간에 따라 분석하는 도구입니다.
        투자자들의 관심도와 시장 심리를 파악하는 데 유용합니다.
        """)
        
        # 검색어 설정 섹션
        st.subheader('🔍 검색어 설정')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 자동 키워드 생성 옵션
            if symbols:
                use_stock_keywords = st.checkbox(
                    f"입력된 종목 기반 키워드 사용 ({', '.join(symbols)})",
                    value=True,
                    help="메인에서 입력한 종목을 기반으로 자동으로 검색 키워드를 생성합니다."
                )
            else:
                use_stock_keywords = False
                st.info("메인에서 종목을 먼저 입력하시면 자동 키워드 생성이 가능합니다.")
        
        with col2:
            # 분석 기간 선택
            timeframe_options = {
                '지난 1시간': 'now 1-H',
                '지난 4시간': 'now 4-H',
                '지난 1일': 'now 1-d',
                '지난 7일': 'now 7-d',
                '지난 1개월': 'today 1-m',
                '지난 3개월': 'today 3-m',
                '지난 12개월': 'today 12-m',
                '지난 5년': 'today 5-y',
                '2004년~현재': 'all'
            }
            
            selected_timeframe_label = st.selectbox(
                '분석 기간 선택',
                list(timeframe_options.keys()),
                index=6
            )
            selected_timeframe = timeframe_options[selected_timeframe_label]
        
        # 키워드 자동 생성
        default_keywords = ""
        if use_stock_keywords and symbols:
            auto_keywords = []
            for symbol in symbols:
                try:
                    financial_data = DataManager.get_financial_data(symbol)
                    if financial_data and 'info' in financial_data:
                        company_name = financial_data['info'].get('shortName', symbol)
                    else:
                        company_name = symbol
                except:
                    company_name = symbol
                
                try:
                    stock_keywords = GoogleTrendsAnalyzer.get_stock_related_keywords(symbol, company_name)
                    auto_keywords.extend(stock_keywords)
                except:
                    auto_keywords.append(symbol)
            
            auto_keywords = list(dict.fromkeys(auto_keywords))[:5]
            default_keywords = ', '.join(auto_keywords)
            
            if auto_keywords:
                st.success(f"자동 생성된 키워드: {default_keywords}")
        
        # 키워드 입력
        manual_keywords = st.text_input(
            '검색 키워드 입력 (쉼표로 구분, 최대 5개)',
            value=default_keywords,
            placeholder='예: AAPL, Apple stock, iPhone, Tesla, TSLA stock',
            help='구글에서 검색할 키워드를 입력하세요. 영어 키워드를 권장합니다.'
        )
        
        if manual_keywords:
            keywords = [k.strip() for k in manual_keywords.split(',') if k.strip()][:5]
            
            # 지역 설정
            geo_options = {
                '전 세계': '',
                '미국': 'US',
                '한국': 'KR',
                '일본': 'JP',
                '중국': 'CN',
                '독일': 'DE',
                '영국': 'GB'
            }
            
            selected_geo_label = st.selectbox('지역 선택', list(geo_options.keys()))
            selected_geo = geo_options[selected_geo_label]
            
            # 키워드가 변경되었는지 확인
            current_keywords_string = f"{','.join(keywords)}_{selected_timeframe}_{selected_geo}"
            keywords_changed = (current_keywords_string != st.session_state.trends_keywords_cache)
            
            # 분석 시작 버튼
            if st.button('📊 트렌드 분석 시작', type='primary', use_container_width=True) or keywords_changed:
                
                if not keywords:
                    st.error("키워드를 입력해주세요.")
                else:
                    # 새로운 분석 시작
                    st.session_state.trends_analysis_done = False
                    st.session_state.gpt_trends_result = None
                    st.session_state.trends_keywords_cache = current_keywords_string
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        status_text.text("🔍 구글 트렌드 데이터를 수집하는 중...")
                        progress_bar.progress(25)
                        
                        trends_data = GoogleTrendsAnalyzer.get_trends_data(
                            keywords, 
                            timeframe=selected_timeframe,
                            geo=selected_geo
                        )
                        
                        # 세션 상태에 저장
                        st.session_state.trends_data_cache = trends_data
                        
                        progress_bar.progress(100)
                        status_text.text("✅ 분석 완료!")
                        
                        st.session_state.trends_analysis_done = True
                        
                    except Exception as e:
                        st.error(f"❌ 트렌드 분석 중 오류: {str(e)}")
                    finally:
                        progress_bar.empty()
                        status_text.empty()
            
            # 캐시된 트렌드 데이터가 있고 분석이 완료된 경우 결과 표시
            if st.session_state.trends_analysis_done and st.session_state.trends_data_cache:
                trends_data = st.session_state.trends_data_cache
                
                if trends_data.get('error'):
                    st.error(f"❌ {trends_data['error']}")
                    st.markdown("**💡 문제 해결 방법:**")
                    st.info("• 다른 키워드 시도")
                    st.info("• 분석 기간 변경")
                    st.info("• 지역 설정 변경")
                    st.info("• 영어 키워드 사용")
                else:
                    has_time_data = (trends_data.get('interest_over_time') is not None and 
                                not trends_data.get('interest_over_time').empty)
                    has_region_data = (trends_data.get('interest_by_region') is not None and 
                                    not trends_data.get('interest_by_region').empty)
                    has_related_data = trends_data.get('related_queries') is not None
                    
                    if not any([has_time_data, has_region_data, has_related_data]):
                        st.warning("⚠️ 수집된 데이터가 없습니다.")
                    else:
                        st.success(f"🎉 '{', '.join(keywords)}' 트렌드 분석 결과")
                        
                        # 시간별 트렌드 차트
                        if has_time_data:
                            st.subheader('📈 시간별 검색 트렌드')
                            trends_chart = GoogleTrendsAnalyzer.create_trends_chart(trends_data)
                            if trends_chart and hasattr(trends_chart, 'data') and trends_chart.data:
                                st.plotly_chart(trends_chart, use_container_width=True)
                                
                                # 트렌드 요약
                                interest_df = trends_data.get('interest_over_time')
                                if interest_df is not None and not interest_df.empty:
                                    st.subheader('📊 트렌드 요약')
                                    cols = st.columns(min(len(keywords), 4))
                                    for i, keyword in enumerate(keywords):
                                        if keyword in interest_df.columns and i < 4:
                                            with cols[i]:
                                                try:
                                                    keyword_data = interest_df[keyword].dropna()
                                                    if len(keyword_data) > 0:
                                                        recent_avg = keyword_data.tail(min(30, len(keyword_data))).mean()
                                                        max_val = keyword_data.max()
                                                        
                                                        if len(keyword_data) >= 14:
                                                            recent_trend = (keyword_data.tail(7).mean() - 
                                                                        keyword_data.tail(14).head(7).mean())
                                                        else:
                                                            recent_trend = 0
                                                        
                                                        st.metric(
                                                            keyword,
                                                            f"{recent_avg:.1f}",
                                                            delta=f"{recent_trend:+.1f}" if recent_trend != 0 else "0.0",
                                                            help=f"최고값: {max_val}, 최근 평균: {recent_avg:.1f}"
                                                        )
                                                    else:
                                                        st.metric(keyword, "데이터 없음")
                                                except Exception as e:
                                                    st.warning(f"{keyword} 요약 생성 실패")
                        
                        # 지역별 관심도 차트
                        if has_region_data:
                            st.subheader('🌍 국가별 검색 관심도')
                            try:
                                regional_chart = GoogleTrendsAnalyzer.create_regional_chart(trends_data)
                                if regional_chart and hasattr(regional_chart, 'data') and regional_chart.data:
                                    st.plotly_chart(regional_chart, use_container_width=True)
                            except Exception as e:
                                st.warning('지역별 차트 생성 실패')
                        
                        # 관련 검색어 표시
                        if has_related_data:
                            related_queries = trends_data.get('related_queries', {})
                            if related_queries:
                                st.subheader('🔍 관련 검색어')
                                cols = st.columns(min(len(keywords), 2))
                                for i, keyword in enumerate(keywords[:2]):
                                    if keyword in related_queries:
                                        with cols[i]:
                                            st.write(f"**{keyword} 관련 상승 검색어:**")
                                            
                                            try:
                                                rising_data = related_queries[keyword].get('rising')
                                                if rising_data is not None and not rising_data.empty:
                                                    rising_df = rising_data.head(5)
                                                    for _, row in rising_df.iterrows():
                                                        value_str = str(row['value']) if pd.notna(row['value']) else 'N/A'
                                                        st.write(f"• {row['query']} (+{value_str})")
                                                else:
                                                    st.write("상승 검색어 데이터 없음")
                                            except:
                                                st.write("상승 검색어 표시 오류")
                                            
                                            st.write(f"**{keyword} 관련 인기 검색어:**")
                                            try:
                                                top_data = related_queries[keyword].get('top')
                                                if top_data is not None and not top_data.empty:
                                                    top_df = top_data.head(5)
                                                    for _, row in top_df.iterrows():
                                                        value_str = str(row['value']) if pd.notna(row['value']) else 'N/A'
                                                        st.write(f"• {row['query']} ({value_str})")
                                                else:
                                                    st.write("인기 검색어 데이터 없음")
                                            except:
                                                st.write("인기 검색어 표시 오류")
                        
                        # GPT 분석 섹션
                        st.markdown('---')
                        st.subheader('🤖 AI 기반 트렌드 분석')
                        
                        # GPT 분석 결과가 이미 있는 경우 표시
                        if st.session_state.gpt_trends_result:
                            gpt_result = st.session_state.gpt_trends_result
                            
                            if gpt_result.get('error'):
                                st.error(f"🤖 분석 오류: {gpt_result['error']}")
                            else:
                                st.success("🎉 GPT 분석 완료!")
                                
                                # 분석 점수 표시
                                score = gpt_result.get('score', 3.0)
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        "트렌드 종합 점수",
                                        f"{score:.1f}/5.0",
                                        delta=f"{(score/5.0)*100:.0f}%"
                                    )
                                
                                with col2:
                                    if score >= 4.0:
                                        st.success("🔥 높은 관심도")
                                    elif score >= 3.0:
                                        st.info("📊 보통 관심도")
                                    else:
                                        st.warning("📉 낮은 관심도")
                                
                                with col3:
                                    if score >= 4.0:
                                        st.success("📈 높은 대중 관심")
                                    elif score >= 2.5:
                                        st.info("⚖️ 중간 관심도")
                                    else:
                                        st.error("😴 낮은 관심도")
                                
                                # GPT 분석 내용 표시
                                st.markdown("### 📋 트렌드 분석 리포트")
                                st.markdown(gpt_result['analysis'])
                                
                                # 추가 액션 제안
                                st.markdown("---")
                                st.subheader("🔄 다음 단계 제안")
                                
                                if score >= 4.0:
                                    st.success("""
                                    **높은 관심도 활용 전략:**
                                    • 관련 뉴스 분석 탭에서 최신 동향 확인
                                    • 재무 건전성 탭에서 기업 실력 검증
                                    • 주가 예측 탭에서 진입 타이밍 분석
                                    """)
                                elif score >= 2.5:
                                    st.info("""
                                    **중간 관심도 대응 전략:**
                                    • 경제지표 탭에서 거시환경 점검
                                    • 유튜브 분석 탭에서 전문가 의견 수집
                                    • 섹터 트리맵에서 업종 전체 동향 확인
                                    """)
                                else:
                                    st.warning("""
                                    **낮은 관심도 주의사항:**
                                    • 워렌 버핏 조언 탭에서 가치 투자 관점 확인
                                    • 다른 키워드로 트렌드 재분석
                                    • 장기 투자 관점에서 접근 고려
                                    """)
                        
                        else:
                            # GPT 분석이 아직 안 된 경우 - Form으로 분석 실행
                            with st.form("gpt_analysis_form", clear_on_submit=False):
                                st.info("🤖 AI 기반 트렌드 분석을 실행하시겠습니까?")
                                
                                openai_api_key = st.text_input(
                                    "OpenAI API 키",
                                    type="password",
                                    help="GPT를 이용한 전문가 수준의 트렌드 분석"
                                )
                                
                                analyze_gpt = st.form_submit_button(
                                    "🚀 AI 분석 시작",
                                    type="primary",
                                    use_container_width=True
                                )
                                
                                if analyze_gpt:
                                    if not openai_api_key:
                                        st.error("⚠️ OpenAI API 키를 입력해주세요.")
                                    elif not openai_api_key.startswith('sk-'):
                                        st.error("❌ 올바른 API 키 형식이 아닙니다.")
                                    else:
                                        try:
                                            with st.spinner('🤖 GPT 분석 중...'):
                                                gpt_analysis = GoogleTrendsAnalyzer.analyze_trends_with_gpt(
                                                    trends_data, openai_api_key
                                                )
                                            
                                            # 결과를 세션 상태에 저장
                                            st.session_state.gpt_trends_result = gpt_analysis
                                            
                                            st.success("✅ 분석이 완료되었습니다! 페이지를 새로고침하지 마세요.")
                                            st.rerun()  # 결과를 바로 표시하기 위해 rerun
                                            
                                        except Exception as e:
                                            error_result = {"error": f"분석 실행 중 오류: {str(e)}"}
                                            st.session_state.gpt_trends_result = error_result
                                            st.error(f"분석 실패: {str(e)}")
                            
                            # API 키 없는 경우 안내
                            with st.expander('🔑 GPT 분석 없이도 확인 가능한 정보'):
                                st.info("""
                                **현재 확인 가능한 분석:**
                                • 📈 시간별 트렌드 패턴
                                • 📊 최근 관심도 변화
                                • 🌍 지역별 관심 분포
                                • 🔍 관련 검색어 트렌드
                                
                                **GPT 분석 추가 혜택:**
                                • 투자 관점 해석
                                • 종합 점수 평가
                                • 다음 단계 가이드
                                • 시장 심리 분석
                                """)

        # 사용법 안내
        with st.expander('💡 구글 트렌드 분석 사용법'):
            st.markdown("""
            **🔍 검색 및 분석**
            1. 키워드 입력 (영어 권장)
            2. 기간 및 지역 선택
            3. "📊 트렌드 분석 시작" 클릭
            4. 결과 확인 후 GPT 분석 선택
            
            **💡 효과적인 키워드**
            - "AAPL stock" (종목코드 + stock)
            - "Apple investment" (회사명 + investment)
            - "Tesla earnings" (이벤트 관련)
            
            **📊 해석 방법**
            - 높은 수치: 해당 기간 최고 관심도
            - 상승 트렌드: 관심도 증가
            - 지역별 차이: 글로벌 vs 로컬 관심
            - 관련 검색어: 추가 분석 포인트
            
            **⚠️ 주의사항**
            - 검색량 ≠ 투자 수익
            - 다른 지표와 함께 분석
            - 급증/급감 시 원인 파악 필요
            """)
    
    # 사이드바 - 추가 정보
    st.sidebar.markdown('---')
    st.sidebar.subheader('📖 사용법 안내')
    st.sidebar.markdown("""
    1. **종목 코드 입력**: 분석하고 싶은 종목 코드를 입력하세요
    2. **예측 모델 선택**: Prophet 또는 ARIMA 모델을 선택하세요
    3. **각 탭 확인**: 
       - 홈: 서비스 개요
       - 섹터 트리맵: 실시간 섹터별 시가총액 시각화
       - 주가 예측: AI 모델 기반 미래 주가 예측
       - 뉴스 분석: 뉴스 감성 분석
       - 재무 건전성: 현재 재무상태 + 3개년 실적
       - 경제지표: 거시경제 환경 분석
       - 버핏 조언: 종합 투자 판단
    4. **종합 판단**: 모든 분석 결과를 종합하여 투자 결정을 내리세요
    """)
    
    st.sidebar.markdown('---')
    st.sidebar.subheader('🆕 새로 추가된 기능')
    st.sidebar.markdown("""
    **📊 섹터별 트리맵**
    • S&P 500 기반 실시간 데이터
    • 시가총액 상위 20개 종목
    • 일일 변동률에 따른 색상 표시
    • 인터랙티브 호버 정보
    • 섹터별 종합 비교 차트
    
    **🎨 시각화 특징**
    • 트리맵 크기: 시가총액 비례
    • 색상: 일일 변동률 기준
    • 호버: 상세 정보 표시
    • 반응형: 클릭 및 확대/축소 가능
    """)
    
    st.sidebar.markdown('---')
    st.sidebar.subheader('📊 기존 기능')
    st.sidebar.markdown("""
    **📈 3개년 재무 실적**
    • 매출, 영업이익, 순이익 추이
    • 연도별 성장 패턴 분석
    • 수익성 변화 추적
    
    **🌍 대외 경제지표**
    • 주요 주식 지수 (S&P500, 나스닥, 다우)
    • 미국 국채 수익률 (2년, 10년)
    • 원자재 (금, 원유)
    • 달러 인덱스, VIX 공포지수
    """)
    
    st.sidebar.markdown('---')
    st.sidebar.subheader('📊 모델 비교')
    st.sidebar.markdown("""
    **Prophet**
    • Facebook이 개발한 시계열 예측 모델
    • 계절성과 트렌드를 잘 감지
    • 빠른 학습 및 예측
    • 적은 데이터로도 작동
    
    **ARIMA**
    • 전통적이고 검증된 통계 모델
    • 자기회귀통합이동평균 모델
    • 선형 관계에 적합
    • 빠른 처리 속도
    """)
    
    st.sidebar.markdown('---')
    st.sidebar.subheader('⚠️ 면책 고지')
    st.sidebar.markdown("""
    - 본 분석은 참고용이며 투자 보장하지 않습니다
    - 모든 투자는 본인 책임입니다
    - 충분한 조사 후 투자 결정하세요
    - 분산 투자를 권장합니다
    """)
    
    # 푸터
    st.markdown('---')
    st.markdown(
        '<div style="text-align: center; color: gray; font-size: 0.8rem;">'
        '© 2025 서학개미의 투자 탐구생활. 워렌 버핏의 투자 철학을 기반으로 한 AI 분석 플랫폼'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()