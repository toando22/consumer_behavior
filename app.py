import numpy as np
import pandas as pd
from scipy import stats
import warnings
import logging
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import io
import base64
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import tempfile
import os

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tắt cảnh báo
warnings.filterwarnings("ignore")

# Tùy chỉnh giao diện Streamlit
st.set_page_config(page_title="Phân Tích Hành Vi Người Tiêu Dùng", layout="wide", page_icon="📊")
st.markdown("""
    <style>
    .stApp {
        background-color: #f9fafb;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: nowrap;
        background-color: #e5e7eb;
        border-radius: 4px;
        color: #1f2937;
        font-weight: 500;
        padding: 0 20px;
        display: flex;
        align-items: center;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #d1d5db;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
    .stFileUploader label {
        font-size: 16px;
        color: #1f2937;
    }
    .stProgress > div > div > div {
        background-color: #3b82f6;
    }
    .stMarkdown h1 {
        color: #1f2937;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 8px;
    }
    .stMarkdown h2 {
        color: #1f2937;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# Hàm xử lý dữ liệu
@st.cache_data(show_spinner="Đang xử lý dữ liệu...")
def preprocess_data(df):
    try:
        required_columns = ['ID', 'Dt_Customer', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
                           'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                           'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'AcceptedCmp1', 'AcceptedCmp2',
                           'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Complain', 'Response']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Thiếu các cột bắt buộc: {', '.join(missing_cols)}")
        
        # Tạo bản sao để tránh thay đổi dataframe gốc
        df = df.copy()
        
        # Xóa các cột không cần thiết
        df.drop(['ID', 'Z_CostContact', 'Z_Revenue'], axis=1, inplace=True, errors='ignore')
        
        # Chuyển đổi cột Dt_Customer thành datetime
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format="%d-%m-%Y", errors='coerce')
        if df['Dt_Customer'].isna().any():
            invalid_dates = df[df['Dt_Customer'].isna()].index.tolist()
            raise ValueError(f"Cột Dt_Customer chứa giá trị không hợp lệ tại các hàng: {invalid_dates}")
        
        latest_date = df['Dt_Customer'].max()
        df['Days_is_client'] = (latest_date - df['Dt_Customer']).dt.days

        # Chuẩn hóa các cột phân loại
        df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'], 'Partner')
        df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'], 'Single')
        df['Education'] = df['Education'].replace(['PhD', 'Master'], 'Postgraduate')
        df['Education'] = df['Education'].replace(['2n Cycle', 'Graduation'], 'Graduate')
        df['Education'] = df['Education'].replace(['Basic'], 'Undergraduate')

        # Tạo các cột tổng hợp
        df['Kids'] = df['Kidhome'] + df['Teenhome']
        df['Expenses'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + \
                         df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
        df['TotalAcceptedCmp'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + \
                                 df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5']
        df['TotalNumPurchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + \
                                  df['NumStorePurchases'] + df['NumDealsPurchases']

        # Chọn các cột cần thiết
        selected_columns = ['Education', 'Marital_Status', 'Income', 'Kids', 'Days_is_client', 
                           'Recency', 'Expenses', 'TotalNumPurchases', 'TotalAcceptedCmp', 
                           'Complain', 'Response', 'Dt_Customer']
        df = df[selected_columns]

        # Xóa hàng trùng lặp và giá trị thiếu
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)

        # Xóa giá trị ngoại lai
        numerical_columns = ['Income', 'Kids', 'Days_is_client', 'Recency', 
                            'Expenses', 'TotalNumPurchases', 'TotalAcceptedCmp']
        z_scores = pd.DataFrame(stats.zscore(df[numerical_columns]), columns=numerical_columns)
        outliers = z_scores[(np.abs(z_scores) > 3).any(axis=1)]
        df = df.drop(outliers.index)

        # Phân loại cột
        binary_columns = [col for col in df.columns if df[col].nunique() == 2 and col != 'Dt_Customer']
        categorical_columns = [col for col in df.columns if 2 < df[col].nunique() < 10 and col != 'Dt_Customer']
        
        return df, numerical_columns, categorical_columns, binary_columns, None
    except Exception as e:
        logger.error(f"Lỗi xử lý dữ liệu: {str(e)}", exc_info=True)
        return None, None, None, None, str(e)

# Hàm phân cụm khách hàng
@st.cache_data(show_spinner="Đang phân cụm dữ liệu...")
def cluster_analysis(df, numerical_columns, n_clusters):
    try:
        if df is None or len(df) == 0:
            raise ValueError("Dữ liệu đầu vào trống hoặc không hợp lệ")
            
        categorical_columns = df.select_dtypes(include=['object']).columns
        X_encoded = pd.get_dummies(df.drop(columns=['Dt_Customer']), columns=categorical_columns, drop_first=True, dtype=int)
        
        # Kiểm tra dữ liệu sau khi mã hóa
        if X_encoded.shape[0] == 0:
            raise ValueError("Không có dữ liệu sau khi mã hóa")
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        # Tính toán Elbow và Silhouette
        ssd = []
        silhouette_scores = []
        range_n_clusters = range(2, min(11, X_scaled.shape[0]))  # Đảm bảo không vượt quá số mẫu
        
        for n in range_n_clusters:
            kmeans_temp = KMeans(n_clusters=n, max_iter=50, random_state=101, n_init=10)
            kmeans_temp.fit(X_scaled)
            ssd.append(kmeans_temp.inertia_)
            
            # Silhouette score yêu cầu ít nhất 2 cụm và ít nhất 2 mẫu mỗi cụm
            if n > 1 and all(np.bincount(kmeans_temp.labels_) > 1):
                score = silhouette_score(X_scaled, kmeans_temp.labels_)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)

        # Phân cụm với số cụm được chọn
        kmeans = KMeans(n_clusters=n_clusters, max_iter=50, random_state=101, n_init=10)
        y_kmeans = kmeans.fit_predict(X_scaled)

        df_clusters = df.copy()
        df_clusters['Cluster'] = y_kmeans
        return df_clusters, X_scaled, scaler, X_encoded, ssd, silhouette_scores, None
    except Exception as e:
        logger.error(f"Lỗi phân cụm: {str(e)}", exc_info=True)
        return None, None, None, None, None, None, str(e)

# Hàm dự đoán khách hàng quay lại
@st.cache_data(show_spinner="Đang dự đoán khách hàng quay lại...")
def churn_prediction(df, X_scaled):
    try:
        if df is None or X_scaled is None:
            raise ValueError("Dữ liệu đầu vào không hợp lệ")
            
        if 'Response' not in df.columns:
            raise ValueError("Thiếu cột 'Response' trong dữ liệu")
            
        X = X_scaled
        y = df['Response']
        
        # Kiểm tra nếu chỉ có 1 lớp trong y
        if len(np.unique(y)) < 2:
            raise ValueError("Dữ liệu chỉ chứa một lớp, không thể phân loại")
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=101)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        churn_report = classification_report(y_test, y_pred, output_dict=True)
        churn_probability = rf_model.predict_proba(X)[:, 1]
        df['Churn_Probability'] = churn_probability
        return df, churn_report, None
    except Exception as e:
        logger.error(f"Lỗi dự đoán khách hàng: {str(e)}", exc_info=True)
        return None, None, str(e)

# Hàm dự báo xu hướng
@st.cache_data(show_spinner="Đang dự báo xu hướng...")
def trend_forecasting(df):
    try:
        if df is None or len(df) == 0:
            raise ValueError("Dữ liệu đầu vào trống hoặc không hợp lệ")
            
        if 'Dt_Customer' not in df.columns or 'Expenses' not in df.columns:
            raise ValueError("Thiếu cột 'Dt_Customer' hoặc 'Expenses'")
            
        prophet_df = df[['Dt_Customer', 'Expenses']].copy()
        prophet_df['Dt_Customer'] = pd.to_datetime(prophet_df['Dt_Customer'], format="%d-%m-%Y", errors='coerce')
        
        # Kiểm tra dữ liệu ngày
        if prophet_df['Dt_Customer'].isna().any():
            invalid_dates = prophet_df[prophet_df['Dt_Customer'].isna()].index.tolist()
            raise ValueError(f"Cột Dt_Customer chứa giá trị không hợp lệ tại các hàng: {invalid_dates}")
            
        prophet_df = prophet_df.groupby(prophet_df['Dt_Customer'].dt.to_period('M').dt.to_timestamp())['Expenses'].sum().reset_index()
        prophet_df.columns = ['ds', 'y']
        
        # Kiểm tra số lượng điểm dữ liệu
        if len(prophet_df) < 2:
            raise ValueError("Không đủ dữ liệu lịch sử để dự báo (cần ít nhất 2 điểm dữ liệu)")
            
        # Tạo và huấn luyện mô hình Prophet
        prophet_model = Prophet(seasonality_mode='multiplicative')
        prophet_model.fit(prophet_df)
        
        # Tạo dữ liệu tương lai và dự báo
        future = prophet_model.make_future_dataframe(periods=12, freq='M')
        forecast = prophet_model.predict(future)
        
        return prophet_df, forecast, None
    except Exception as e:
        logger.error(f"Lỗi dự báo xu hướng: {str(e)}", exc_info=True)
        return None, None, str(e)

# Hàm tạo báo cáo PDF
def generate_pdf_report(df, df_clusters, churn_report, prophet_df, forecast):
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Tiêu đề
        title_style = styles['Title']
        title_style.alignment = 1  # Center alignment
        story.append(Paragraph("Báo Cáo Phân Tích Khách Hàng", title_style))
        story.append(Spacer(1, 24))

        # Tổng quan
        heading2_style = styles['Heading2']
        heading2_style.textColor = colors.HexColor('#3b82f6')
        story.append(Paragraph("1. Tổng Quan", heading2_style))
        
        # Tạo bảng tổng quan
        overview_data = [
            ["Chỉ số", "Giá trị"],
            ["Tổng số khách hàng", f"{len(df):,}"],
            ["Chi tiêu trung bình", f"{df['Expenses'].mean():,.2f}"],
            ["Xác suất quay lại TB", f"{df_clusters['Churn_Probability'].mean():.2%}"],
            ["Số cụm khách hàng", f"{df_clusters['Cluster'].nunique()}"]
        ]
        
        overview_table = Table(overview_data, colWidths=[200, 100])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f3f4f6')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d1d5db')),
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 24))

        # Phân cụm
        story.append(Paragraph("2. Phân Cụm Khách Hàng", heading2_style))
        cluster_summary = df_clusters.groupby('Cluster').agg({
            'Income': 'mean', 
            'Expenses': 'mean', 
            'TotalNumPurchases': 'mean', 
            'Churn_Probability': 'mean',
            'Kids': 'mean'
        }).round(2).reset_index()
        
        cluster_data = [['Cụm', 'Thu Nhập TB', 'Chi Tiêu TB', 'Số Mua Hàng TB', 'Xác Suất Quay Lại TB', 'Số Con TB']]
        for _, row in cluster_summary.iterrows():
            cluster_data.append([
                str(row['Cluster']),
                f"{row['Income']:,.2f}",
                f"{row['Expenses']:,.2f}",
                f"{row['TotalNumPurchases']:.2f}",
                f"{row['Churn_Probability']:.2%}",
                f"{row['Kids']:.2f}"
            ])
        
        cluster_table = Table(cluster_data, colWidths=[60, 80, 80, 80, 80, 60])
        cluster_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f3f4f6')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d1d5db')),
        ]))
        story.append(cluster_table)
        story.append(Spacer(1, 24))

        # Dự đoán quay lại
        story.append(Paragraph("3. Dự Đoán Khách Hàng Quay Lại", heading2_style))
        
        if churn_report:
            churn_data = [
                ["Chỉ số", "Giá trị"],
                ["Độ chính xác", f"{churn_report['accuracy']:.2%}"],
                ["Precision (Lớp 0)", f"{churn_report['0']['precision']:.2%}"],
                ["Recall (Lớp 0)", f"{churn_report['0']['recall']:.2%}"],
                ["Precision (Lớp 1)", f"{churn_report['1']['precision']:.2%}"],
                ["Recall (Lớp 1)", f"{churn_report['1']['recall']:.2%}"]
            ]
            
            churn_table = Table(churn_data, colWidths=[150, 100])
            churn_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f3f4f6')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d1d5db')),
            ]))
            story.append(churn_table)
            story.append(Spacer(1, 24))

        # Dự báo chi tiêu
        story.append(Paragraph("4. Dự Báo Chi Tiêu", heading2_style))
        if prophet_df is not None and forecast is not None:
            forecast_data = [
                ["Thời gian", "Chi tiêu dự báo", "Giới hạn dưới", "Giới hạn trên"],
                [
                    forecast['ds'].dt.strftime('%Y-%m-%d').iloc[-1],
                    f"{forecast['yhat'].iloc[-1]:,.2f}",
                    f"{forecast['yhat_lower'].iloc[-1]:,.2f}",
                    f"{forecast['yhat_upper'].iloc[-1]:,.2f}"
                ]
            ]
            
            forecast_table = Table(forecast_data, colWidths=[100, 80, 80, 80])
            forecast_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f3f4f6')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d1d5db')),
            ]))
            story.append(forecast_table)
        
        story.append(Spacer(1, 24))
        story.append(Paragraph("Báo cáo được tạo tự động bởi Hệ thống Phân tích Khách hàng", styles['Italic']))

        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        logger.error(f"Lỗi tạo báo cáo PDF: {str(e)}", exc_info=True)
        return None

# Hàm tạo dashboard
def create_dashboard(df, df_clusters, churn_report, prophet_df, forecast, ssd, silhouette_scores):
    try:
        st.subheader("Tổng Quan Phân Tích")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tổng Số Khách Hàng", f"{len(df):,}")
        with col2:
            st.metric("Chi Tiêu Trung Bình", f"{df['Expenses'].mean():,.2f}")
        with col3:
            st.metric("Xác Suất Quay Lại TB", f"{df_clusters['Churn_Probability'].mean():.2%}")
        with col4:
            st.metric("Số Cụm", df_clusters['Cluster'].nunique())

        # Phân bố cụm và trình độ học vấn
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(df_clusters, names='Cluster', title='Phân Bố Cụm Khách Hàng',
                            color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            fig_bar = px.histogram(df, x='Education', title='Phân Bố Trình Độ Học Vấn',
                                  color='Education', 
                                  color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Xu hướng chi tiêu
        st.subheader("Xu Hướng Chi Tiêu")
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=prophet_df['ds'], y=prophet_df['y'], 
            mode='lines', name='Thực Tế',
            line=dict(color='#3b82f6', width=2)
        ))
        fig_trend.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'], 
            mode='lines', name='Dự Báo',
            line=dict(color='#10b981', width=2, dash='dot')
        ))
        fig_trend.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_upper'], 
            fill=None, mode='lines',
            line=dict(width=0), showlegend=False
        ))
        fig_trend.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_lower'], 
            fill='tonexty', mode='lines',
            fillcolor='rgba(59, 130, 246, 0.2)',
            line=dict(width=0), name='Khoảng Tin Cậy'
        ))
        fig_trend.update_layout(
            title='Xu Hướng Chi Tiêu Theo Thời Gian',
            xaxis_title='Ngày',
            yaxis_title='Chi Tiêu',
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Phân tích từng cụm
        st.subheader("Phân Tích Từng Cụm")
        cluster_summary = df_clusters.groupby('Cluster').agg({
            'Income': ['mean', 'median', 'std'], 
            'Expenses': ['mean', 'median', 'std'], 
            'TotalNumPurchases': ['mean', 'median'], 
            'Churn_Probability': 'mean',
            'Kids': 'mean'
        }).round(2)
        st.dataframe(cluster_summary.style.background_gradient(cmap='Blues'))

        # Elbow và Silhouette Plot
        st.subheader("Xác Định Số Cụm Tối Ưu")
        col1, col2 = st.columns(2)
        with col1:
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=list(range(2, len(ssd)+2)), y=ssd, 
                mode='lines+markers', 
                line=dict(color='#3b82f6', width=2),
                marker=dict(size=8, color='#3b82f6')
            ))
            fig_elbow.update_layout(
                title='Elbow Method',
                xaxis_title='Số Cụm',
                yaxis_title='Tổng Bình Phương Khoảng Cách',
                template='plotly_white'
            )
            st.plotly_chart(fig_elbow, use_container_width=True)
        with col2:
            fig_silhouette = go.Figure()
            fig_silhouette.add_trace(go.Scatter(
                x=list(range(2, len(silhouette_scores)+2)), y=silhouette_scores, 
                mode='lines+markers',
                line=dict(color='#10b981', width=2),
                marker=dict(size=8, color='#10b981')
            ))
            fig_silhouette.update_layout(
                title='Silhouette Score',
                xaxis_title='Số Cụm',
                yaxis_title='Điểm Silhouette',
                template='plotly_white'
            )
            st.plotly_chart(fig_silhouette, use_container_width=True)

        # Nút tải xuống
        st.subheader("Tải Xuống Kết Quả")
        col1, col2 = st.columns(2)
        with col1:
            # Tải xuống báo cáo PDF
            pdf_buffer = generate_pdf_report(df, df_clusters, churn_report, prophet_df, forecast)
            if pdf_buffer:
                st.download_button(
                    label="📄 Tải xuống Báo Cáo PDF",
                    data=pdf_buffer,
                    file_name="customer_analysis_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        with col2:
            # Tải xuống dữ liệu cụm
            csv = df_clusters.to_csv(index=False)
            st.download_button(
                label="📊 Tải xuống dữ liệu cụm (CSV)",
                data=csv,
                file_name="customer_clusters.csv",
                mime="text/csv",
                use_container_width=True
            )
    except Exception as e:
        st.error(f"Lỗi khi tạo dashboard: {str(e)}")

# Giao diện chính
def main():
    st.title("📊 Phân Tích Khách Hàng")
    st.markdown("""
        Ứng dụng này giúp phân tích dữ liệu khách hàng, bao gồm thống kê, phân cụm, dự đoán khách hàng quay lại, 
        dự báo xu hướng chi tiêu, và ma trận tương quan. Vui lòng tải file CSV theo định dạng yêu cầu.
        
        **📌 Yêu cầu file CSV:** Phải chứa các cột bắt buộc như ID, Dt_Customer, Income, Kidhome, Teenhome, 
        Recency, các cột Mnt (chi tiêu), Num (số lần mua), AcceptedCmp (chiến dịch), Complain, Response.
    """)

    # Khởi tạo session_state
    if 'df' not in st.session_state:
        st.session_state.df = None
        st.session_state.numerical_columns = None
        st.session_state.categorical_columns = None
        st.session_state.binary_columns = None
        st.session_state.df_clusters = None
        st.session_state.X_scaled = None
        st.session_state.scaler = None
        st.session_state.X_encoded = None
        st.session_state.ssd = None
        st.session_state.silhouette_scores = None
        st.session_state.churn_report = None
        st.session_state.prophet_df = None
        st.session_state.forecast = None

    # Tải file CSV
    with st.expander("📤 Tải lên dữ liệu", expanded=True):
        uploaded_file = st.file_uploader("Chọn file CSV", type="csv", help="File CSV phải có định dạng phù hợp")
        if uploaded_file:
            with st.spinner("Đang xử lý dữ liệu..."):
                try:
                    df = pd.read_csv(uploaded_file, sep='\t')
                    df, numerical_columns, categorical_columns, binary_columns, error = preprocess_data(df)
                    if error:
                        st.error(f"❌ Lỗi: {error}")
                    else:
                        st.session_state.df = df
                        st.session_state.numerical_columns = numerical_columns
                        st.session_state.categorical_columns = categorical_columns
                        st.session_state.binary_columns = binary_columns
                        st.success("✅ Dữ liệu đã được tải và xử lý thành công!")
                except Exception as e:
                    st.error(f"❌ Lỗi khi đọc file: {str(e)}")

    # Thanh điều hướng
    tabs = st.tabs(["📊 Tổng Quan", "📈 Thống Kê", "🔍 Phân Cụm", "🎯 Dự Đoán", "📅 Dự Báo", "🔄 Tương Quan", "📖 Hướng Dẫn"])

    # Tab Tổng Quan
    with tabs[0]:
        st.header("Dashboard Tổng Quan")
        if st.session_state.df is not None:
            with st.spinner("Đang phân tích dữ liệu..."):
                # Tính toán phân cụm với số cụm mặc định (3 nếu chưa có silhouette_scores)
                if st.session_state.silhouette_scores is None:
                    default_clusters = 3
                else:
                    default_clusters = np.argmax(st.session_state.silhouette_scores) + 2
                
                df_clusters, X_scaled, scaler, X_encoded, ssd, silhouette_scores, error = cluster_analysis(
                    st.session_state.df, st.session_state.numerical_columns, n_clusters=default_clusters)
                
                if error:
                    st.error(f"❌ Lỗi phân cụm: {error}")
                else:
                    df_clusters, churn_report, error = churn_prediction(df_clusters, X_scaled)
                    if error:
                        st.error(f"❌ Lỗi dự đoán: {error}")
                    else:
                        prophet_df, forecast, error = trend_forecasting(st.session_state.df)
                        if error:
                            st.error(f"❌ Lỗi dự báo: {error}")
                        else:
                            st.session_state.df_clusters = df_clusters
                            st.session_state.X_scaled = X_scaled
                            st.session_state.scaler = scaler
                            st.session_state.X_encoded = X_encoded
                            st.session_state.ssd = ssd
                            st.session_state.silhouette_scores = silhouette_scores
                            st.session_state.churn_report = churn_report
                            st.session_state.prophet_df = prophet_df
                            st.session_state.forecast = forecast
                            create_dashboard(st.session_state.df, df_clusters, churn_report, prophet_df, forecast, ssd, silhouette_scores)
        else:
            st.info("ℹ️ Vui lòng tải file CSV để bắt đầu phân tích.")

    # Tab Thống Kê
    with tabs[1]:
        st.header("Thống Kê Dữ Liệu")
        if st.session_state.df is not None:
            st.write("### Thống Kê Mô Tả")
            st.dataframe(st.session_state.df.describe().style.background_gradient(cmap='Blues'))
            
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(st.session_state.df, x='Income', title='Phân Bố Thu Nhập',
                                       nbins=20, color_discrete_sequence=['#3b82f6'])
                st.plotly_chart(fig_hist, use_container_width=True)
            with col2:
                fig_box = px.box(st.session_state.df, y='Expenses', title='Boxplot Chi Tiêu',
                                 color_discrete_sequence=['#10b981'])
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Hiển thị phân bố cho các cột số
            for column in st.session_state.numerical_columns:
                fig_hist = px.histogram(st.session_state.df, x=column, title=f'Phân Bố của {column}', 
                                       nbins=20, color_discrete_sequence=['#3b82f6'])
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Hiển thị phân bố cho các cột phân loại
            for column in st.session_state.categorical_columns + st.session_state.binary_columns:
                fig_count = px.histogram(st.session_state.df, x=column, title=f'Phân Bố của {column}',
                                         color=column, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_count.update_layout(showlegend=False)
                st.plotly_chart(fig_count, use_container_width=True)
            
            # Nút tải xuống thống kê
            csv = st.session_state.df.describe().to_csv()
            st.download_button(
                label="📥 Tải xuống thống kê (CSV)",
                data=csv,
                file_name="statistics.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ Vui lòng tải file CSV để xem thống kê.")

    # Tab Phân Cụm
    with tabs[2]:
        st.header("Phân Cụm Khách Hàng")
        if st.session_state.df is not None:
            if st.session_state.silhouette_scores is None:
                default_n_clusters = 3
            else:
                default_n_clusters = np.argmax(st.session_state.silhouette_scores) + 2
            
            n_clusters = st.slider("Chọn số lượng cụm", min_value=2, max_value=10, value=default_n_clusters)
            
            with st.spinner("Đang phân cụm..."):
                df_clusters, X_scaled, scaler, X_encoded, ssd, silhouette_scores, error = cluster_analysis(
                    st.session_state.df, st.session_state.numerical_columns, n_clusters)
                
                if error:
                    st.error(f"❌ Lỗi phân cụm: {error}")
                else:
                    st.write(f"### Số cụm được chọn: {n_clusters}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig1 = px.scatter(df_clusters, x='Income', y='Expenses', color='Cluster',
                                         title='Phân Cụm (Thu Nhập vs Chi Tiêu)',
                                         color_discrete_sequence=px.colors.qualitative.Pastel)
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        fig2 = px.scatter(df_clusters, x='Income', y='TotalNumPurchases', color='Cluster',
                                         title='Phân Cụm (Thu Nhập vs Tổng Số Mua Hàng)',
                                         color_discrete_sequence=px.colors.qualitative.Pastel)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Hiển thị boxplot cho từng cụm
                    for col in st.session_state.numerical_columns:
                        fig = px.box(df_clusters, x='Cluster', y=col, title=f'Boxplot của {col} theo Cụm',
                                     color='Cluster', color_discrete_sequence=px.colors.qualitative.Pastel)
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Hiển thị phân bố phân loại cho từng cụm
                    for col in st.session_state.categorical_columns:
                        data = df_clusters.groupby(['Cluster', col]).size().unstack().fillna(0)
                        fig = go.Figure(data=[
                            go.Bar(name=val, x=data.index, y=data[val]) for val in data.columns
                        ])
                        fig.update_layout(
                            barmode='stack', 
                            title=f'Phân Bố {col} Theo Cụm',
                            xaxis_title='Cụm', 
                            yaxis_title='Số Lượng',
                            legend_title=col
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Nút tải xuống dữ liệu cụm
                    csv = df_clusters.to_csv(index=False)
                    st.download_button(
                        label="📥 Tải xuống dữ liệu cụm (CSV)",
                        data=csv,
                        file_name="customer_clusters.csv",
                        mime="text/csv"
                    )
        else:
            st.info("ℹ️ Vui lòng tải file CSV để phân cụm.")

    # Tab Dự Đoán
    with tabs[3]:
        st.header("Dự Đoán Khách Hàng Quay Lại")
        if st.session_state.df_clusters is not None and 'Churn_Probability' in st.session_state.df_clusters.columns:
            st.write("### Báo Cáo Dự Đoán")
            st.json(st.session_state.churn_report)
            
            st.write("### Xác Suất Khách Hàng Quay Lại")
            st.dataframe(st.session_state.df_clusters[['Income', 'Expenses', 'Churn_Probability']].sort_values('Churn_Probability', ascending=False))
            
            fig = px.histogram(st.session_state.df_clusters, x='Churn_Probability', 
                              title='Phân Bố Xác Suất Quay Lại',
                              nbins=20, color_discrete_sequence=['#3b82f6'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Nút tải xuống dự đoán
            csv = st.session_state.df_clusters[['Income', 'Expenses', 'Churn_Probability']].to_csv(index=False)
            st.download_button(
                label="📥 Tải xuống dữ liệu dự đoán (CSV)",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ Vui lòng tải file CSV và chạy phân tích tổng quan trước.")

    # Tab Dự Báo
    with tabs[4]:
        st.header("Dự Báo Xu Hướng Chi Tiêu")
        if st.session_state.prophet_df is not None and st.session_state.forecast is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.prophet_df['ds'], y=st.session_state.prophet_df['y'], 
                mode='lines', name='Thực Tế',
                line=dict(color='#3b82f6', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=st.session_state.forecast['ds'], y=st.session_state.forecast['yhat'], 
                mode='lines', name='Dự Báo',
                line=dict(color='#10b981', width=2, dash='dot')
            ))
            fig.add_trace(go.Scatter(
                x=st.session_state.forecast['ds'], y=st.session_state.forecast['yhat_upper'], 
                fill=None, mode='lines', line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=st.session_state.forecast['ds'], y=st.session_state.forecast['yhat_lower'], 
                fill='tonexty', mode='lines',
                fillcolor='rgba(59, 130, 246, 0.2)',
                line=dict(width=0), name='Khoảng Tin Cậy'
            ))
            fig.update_layout(
                title='Dự Báo Chi Tiêu Theo Thời Gian',
                xaxis_title='Ngày',
                yaxis_title='Chi Tiêu',
                hovermode='x unified',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Hiển thị dữ liệu dự báo
            st.write("### Dữ Liệu Dự Báo")
            st.dataframe(st.session_state.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))
            
            # Nút tải xuống dự báo
            csv = st.session_state.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
            st.download_button(
                label="📥 Tải xuống dữ liệu dự báo (CSV)",
                data=csv,
                file_name="forecast.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ Vui lòng tải file CSV và chạy phân tích tổng quan trước.")

    # Tab Tương Quan
    with tabs[5]:
        st.header("Ma Trận Tương Quan")
        if st.session_state.df is not None:
            corr_df = st.session_state.df.corr(numeric_only=True)
            fig = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.index,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                hoverongaps=False
            ))
            fig.update_layout(
                title='Ma Trận Tương Quan Giữa Các Biến',
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                xaxis_zeroline=False,
                yaxis_zeroline=False,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Nút tải xuống ma trận tương quan
            csv = corr_df.to_csv()
            st.download_button(
                label="📥 Tải xuống ma trận tương quan (CSV)",
                data=csv,
                file_name="correlation_matrix.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ Vui lòng tải file CSV để xem ma trận tương quan.")

    # Tab Hướng Dẫn
    with tabs[6]:
        st.header("Hướng Dẫn Sử Dụng")
        st.markdown("""
            ### 📌 Hướng dẫn sử dụng ứng dụng

            1. **Tải file CSV**:
               - Tải file CSV một lần ở phần đầu giao diện. File cần có định dạng giống file mẫu.
               - Các cột bắt buộc: `ID`, `Dt_Customer` (dd-mm-yyyy), `Income`, `Kidhome`, `Teenhome`, `Recency`, 
                 `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`,
                 `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, 
                 `AcceptedCmp1`, `AcceptedCmp2`, `AcceptedCmp3`, `AcceptedCmp4`, `AcceptedCmp5`, `Complain`, `Response`.

            2. **Chọn tab phân tích**:
               - **Tổng Quan**: Dashboard tóm tắt với các số liệu quan trọng
               - **Thống Kê**: Thống kê mô tả và phân bố dữ liệu
               - **Phân Cụm**: Phân nhóm khách hàng với K-Means
               - **Dự Đoán**: Dự đoán khả năng khách hàng quay lại
               - **Dự Báo**: Dự báo xu hướng chi tiêu trong tương lai
               - **Tương Quan**: Mối quan hệ giữa các biến số
               - **Hướng Dẫn**: Hướng dẫn sử dụng chi tiết

            3. **Tải xuống kết quả**:
               - Tải xuống dữ liệu CSV hoặc báo cáo PDF từ các tab tương ứng

            ### 🛠️ Yêu cầu hệ thống
            - Môi trường Anaconda với Python 3.9+
            - Các thư viện cần thiết: `streamlit`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `plotly`, `prophet`, `reportlab`

            ### ⚠️ Khắc phục lỗi thường gặp
            - **Lỗi thiếu cột**: Kiểm tra file CSV có đúng định dạng không
            - **Lỗi định dạng ngày**: Đảm bảo cột `Dt_Customer` có định dạng `dd-mm-yyyy`
            - **Lỗi Prophet**: Cài đặt lại Prophet với `pip install prophet` hoặc khởi động lại kernel

            ### 📧 Liên hệ hỗ trợ
            Nếu cần hỗ trợ, vui lòng liên hệ qua email: doduytoan2201@gmail.com
        """)

if __name__ == "__main__":
    main()