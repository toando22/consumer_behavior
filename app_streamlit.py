import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from core_logic import preprocess_data, cluster_analysis, churn_prediction, trend_forecasting, generate_pdf_report

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
        st.subheader("Phân Tích Từng Cụm")
        cluster_summary = df_clusters.groupby('Cluster').agg({
            'Income': ['mean', 'median', 'std'], 
            'Expenses': ['mean', 'median', 'std'], 
            'TotalNumPurchases': ['mean', 'median'], 
            'Churn_Probability': 'mean',
            'Kids': 'mean'
        }).round(2)
        st.dataframe(cluster_summary.style.background_gradient(cmap='Blues'))
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
        st.subheader("Tải Xuống Kết Quả")
        col1, col2 = st.columns(2)
        with col1:
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
    tabs = st.tabs(["📊 Tổng Quan", "📈 Thống Kê", "🔍 Phân Cụm", "🎯 Dự Đoán", "📅 Dự Báo", "🔄 Tương Quan", "📖 Hướng Dẫn"])
    with tabs[0]:
        st.header("Dashboard Tổng Quan")
        if st.session_state.df is not None:
            with st.spinner("Đang phân tích dữ liệu..."):
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
            for column in st.session_state.numerical_columns:
                fig_hist = px.histogram(st.session_state.df, x=column, title=f'Phân Bố của {column}', 
                                       nbins=20, color_discrete_sequence=['#3b82f6'])
                st.plotly_chart(fig_hist, use_container_width=True)
            for column in st.session_state.categorical_columns + st.session_state.binary_columns:
                fig_count = px.histogram(st.session_state.df, x=column, title=f'Phân Bố của {column}',
                                         color=column, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_count.update_layout(showlegend=False)
                st.plotly_chart(fig_count, use_container_width=True)
            csv = st.session_state.df.describe().to_csv()
            st.download_button(
                label="📥 Tải xuống thống kê (CSV)",
                data=csv,
                file_name="statistics.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ Vui lòng tải file CSV để xem thống kê.")
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
                    for col in st.session_state.numerical_columns:
                        fig = px.box(df_clusters, x='Cluster', y=col, title=f'Boxplot của {col} theo Cụm',
                                     color='Cluster', color_discrete_sequence=px.colors.qualitative.Pastel)
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
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
                    csv = df_clusters.to_csv(index=False)
                    st.download_button(
                        label="📥 Tải xuống dữ liệu cụm (CSV)",
                        data=csv,
                        file_name="customer_clusters.csv",
                        mime="text/csv"
                    )
        else:
            st.info("ℹ️ Vui lòng tải file CSV để phân cụm.")
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
            csv = st.session_state.df_clusters[['Income', 'Expenses', 'Churn_Probability']].to_csv(index=False)
            st.download_button(
                label="📥 Tải xuống dữ liệu dự đoán (CSV)",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ Vui lòng tải file CSV và chạy phân tích tổng quan trước.")
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
            st.write("### Dữ Liệu Dự Báo")
            st.dataframe(st.session_state.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))
            csv = st.session_state.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
            st.download_button(
                label="📥 Tải xuống dữ liệu dự báo (CSV)",
                data=csv,
                file_name="forecast.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ Vui lòng tải file CSV và chạy phân tích tổng quan trước.")
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
            csv = corr_df.to_csv()
            st.download_button(
                label="📥 Tải xuống ma trận tương quan (CSV)",
                data=csv,
                file_name="correlation_matrix.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ Vui lòng tải file CSV để xem ma trận tương quan.")
    with tabs[6]:
        st.header("Hướng Dẫn Sử Dụng")
        st.markdown("""
            ### 📌 Hướng dẫn sử dụng ứng dụng
            1. **Tải file CSV**:
               - Tải file CSV một lần ở phần đầu giao diện. File cần có định dạng giống file mẫu.
               - Các cột bắt buộc: `ID`, `Dt_Customer` (dd-mm-yyyy), `Income`, `Kidhome`, `Teenhome`, 
                 `Recency`, `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`,
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