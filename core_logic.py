import numpy as np
import pandas as pd
from scipy import stats
import warnings
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from prophet import Prophet
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tắt cảnh báo
warnings.filterwarnings("ignore")

# Hàm xử lý dữ liệu
def preprocess_data(df):
    try:
        required_columns = ['ID', 'Dt_Customer', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
                           'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                           'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'AcceptedCmp1', 'AcceptedCmp2',
                           'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Complain', 'Response']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Thiếu các cột bắt buộc: {', '.join(missing_cols)}")
        
        df = df.copy()
        df.drop(['ID', 'Z_CostContact', 'Z_Revenue'], axis=1, inplace=True, errors='ignore')
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format="%d-%m-%Y", errors='coerce')
        if df['Dt_Customer'].isna().any():
            invalid_dates = df[df['Dt_Customer'].isna()].index.tolist()
            raise ValueError(f"Cột Dt_Customer chứa giá trị không hợp lệ tại các hàng: {invalid_dates}")
        
        latest_date = df['Dt_Customer'].max()
        df['Days_is_client'] = (latest_date - df['Dt_Customer']).dt.days
        df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'], 'Partner')
        df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'], 'Single')
        df['Education'] = df['Education'].replace(['PhD', 'Master'], 'Postgraduate')
        df['Education'] = df['Education'].replace(['2n Cycle', 'Graduation'], 'Graduate')
        df['Education'] = df['Education'].replace(['Basic'], 'Undergraduate')
        df['Kids'] = df['Kidhome'] + df['Teenhome']
        df['Expenses'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + \
                         df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
        df['TotalAcceptedCmp'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + \
                                 df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5']
        df['TotalNumPurchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + \
                                  df['NumStorePurchases'] + df['NumDealsPurchases']
        selected_columns = ['Education', 'Marital_Status', 'Income', 'Kids', 'Days_is_client', 
                           'Recency', 'Expenses', 'TotalNumPurchases', 'TotalAcceptedCmp', 
                           'Complain', 'Response', 'Dt_Customer']
        df = df[selected_columns]
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        numerical_columns = ['Income', 'Kids', 'Days_is_client', 'Recency', 
                            'Expenses', 'TotalNumPurchases', 'TotalAcceptedCmp']
        z_scores = pd.DataFrame(stats.zscore(df[numerical_columns]), columns=numerical_columns)
        outliers = z_scores[(np.abs(z_scores) > 3).any(axis=1)]
        df = df.drop(outliers.index)
        binary_columns = [col for col in df.columns if df[col].nunique() == 2 and col != 'Dt_Customer']
        categorical_columns = [col for col in df.columns if 2 < df[col].nunique() < 10 and col != 'Dt_Customer']
        
        return df, numerical_columns, categorical_columns, binary_columns, None
    except Exception as e:
        logger.error(f"Lỗi xử lý dữ liệu: {str(e)}", exc_info=True)
        return None, None, None, None, str(e)

# Hàm phân cụm khách hàng
def cluster_analysis(df, numerical_columns, n_clusters):
    try:
        if df is None or len(df) == 0:
            raise ValueError("Dữ liệu đầu vào trống hoặc không hợp lệ")
        categorical_columns = df.select_dtypes(include=['object']).columns
        X_encoded = pd.get_dummies(df.drop(columns=['Dt_Customer']), columns=categorical_columns, drop_first=True, dtype=int)
        if X_encoded.shape[0] == 0:
            raise ValueError("Không có dữ liệu sau khi mã hóa")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)
        ssd = []
        silhouette_scores = []
        range_n_clusters = range(2, min(11, X_scaled.shape[0]))
        for n in range_n_clusters:
            kmeans_temp = KMeans(n_clusters=n, max_iter=50, random_state=101, n_init=10)
            kmeans_temp.fit(X_scaled)
            ssd.append(kmeans_temp.inertia_)
            if n > 1 and all(np.bincount(kmeans_temp.labels_) > 1):
                score = silhouette_score(X_scaled, kmeans_temp.labels_)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)
        kmeans = KMeans(n_clusters=n_clusters, max_iter=50, random_state=101, n_init=10)
        y_kmeans = kmeans.fit_predict(X_scaled)
        df_clusters = df.copy()
        df_clusters['Cluster'] = y_kmeans
        return df_clusters, X_scaled, scaler, X_encoded, ssd, silhouette_scores, None
    except Exception as e:
        logger.error(f"Lỗi phân cụm: {str(e)}", exc_info=True)
        return None, None, None, None, None, None, str(e)

# Hàm dự đoán khách hàng quay lại
def churn_prediction(df, X_scaled):
    try:
        if df is None or X_scaled is None:
            raise ValueError("Dữ liệu đầu vào không hợp lệ")
        if 'Response' not in df.columns:
            raise ValueError("Thiếu cột 'Response' trong dữ liệu")
        X = X_scaled
        y = df['Response']
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
def trend_forecasting(df):
    try:
        if df is None or len(df) == 0:
            raise ValueError("Dữ liệu đầu vào trống hoặc không hợp lệ")
        if 'Dt_Customer' not in df.columns or 'Expenses' not in df.columns:
            raise ValueError("Thiếu cột 'Dt_Customer' hoặc 'Expenses'")
        prophet_df = df[['Dt_Customer', 'Expenses']].copy()
        prophet_df['Dt_Customer'] = pd.to_datetime(prophet_df['Dt_Customer'], format="%d-%m-%Y", errors='coerce')
        if prophet_df['Dt_Customer'].isna().any():
            invalid_dates = prophet_df[prophet_df['Dt_Customer'].isna()].index.tolist()
            raise ValueError(f"Cột Dt_Customer chứa giá trị không hợp lệ tại các hàng: {invalid_dates}")
        prophet_df = prophet_df.groupby(prophet_df['Dt_Customer'].dt.to_period('M').dt.to_timestamp())['Expenses'].sum().reset_index()
        prophet_df.columns = ['ds', 'y']
        if len(prophet_df) < 2:
            raise ValueError("Không đủ dữ liệu lịch sử để dự báo (cần ít nhất 2 điểm dữ liệu)")
        prophet_model = Prophet(seasonality_mode='multiplicative')
        prophet_model.fit(prophet_df)
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
        title_style = styles['Title']
        title_style.alignment = 1
        story.append(Paragraph("Báo Cáo Phân Tích Khách Hàng", title_style))
        story.append(Spacer(1, 24))
        heading2_style = styles['Heading2']
        heading2_style.textColor = colors.HexColor('#3b82f6')
        story.append(Paragraph("1. Tổng Quan", heading2_style))
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