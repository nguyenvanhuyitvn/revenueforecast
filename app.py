import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Set page configuration
st.set_page_config(page_title="Dự đoán doanh thu", layout="wide", initial_sidebar_state="expanded")
st.title("Dự đoán doanh thu thông minh")

# Các hàm xủ lý dữ liệu
def load_data(uploaded_file):
    """Đọc và tiền xử lý dữ liệu từ file excel."""
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            #Kiểm tra các cột cần thiết
            required_columns = ['Ngày', 'Giá', 'Số lượng', 'Doanh thu', 'Danh mục', 'Sản phẩm']
            for col in required_columns:
                if col not in df.columns:
                    st.error(f"Cột '{col}' không tồn tại trong file.")
                    return None
            df['Ngày'] = pd.to_datetime(df['Ngày'], errors='coerce')
            return df
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi đọc file: {e}")
            return None
    return None

def feature_engineering(df):
    """Thêm các đặc trưng mới cho dữ liệu."""
    df_processed = df.copy()
    df_processed['Quý'] = df_processed['Ngày'].dt.quarter
    df_processed['Tháng'] = df_processed['Ngày'].dt.month
    df_processed['Ngày trong tuần'] = df_processed['Ngày'].dt.dayofweek
    df_processed['Là cuối tuần'] = df_processed['Ngày trong tuần'].isin([5, 6]).astype(int)
    return df_processed
def train_model(df, model_choice):
    """Chuẩn bị dữ liệu, huấn luyện mô hình và trả về mô hình đã huấn luyện."""
    numeric_features = ['Giá', 'Số lượng', 'Quý', 'Tháng', 'Ngày trong tuần', 'Là cuối tuần']
    #Tự xác định các cột phân loại
    categorical_features = df.select_dtypes(include=['object','category']).columns.tolist()
    categorical_features.remove('Ngày') if 'Ngày' in categorical_features else None
    X = df[numeric_features + categorical_features]
    y = df['Doanh thu']
    #Tạo pepeline tiền xử lý
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    #Chọn mô hình
    if model_choice == 'Linear Regression':
        regressor = LinearRegression() 
    else:
        regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', regressor)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, numeric_features, categorical_features

#Giao diện chính
st.sidebar.header("Tải dữ liệu")
uploaded_file = st.sidebar.file_uploader("Chọn file Excel", type=["xlsx"])
if uploaded_file is None:
    st.info("Vui lòng tải lên file Excel để bắt đầu.")
else:
    df_raw = load_data(uploaded_file)
    if df_raw is not None:
        st.success("Dữ liệu đã được tải lên thành công!")
        df = feature_engineering(df_raw)
        # ---------------------------
        # Sidebar cho các tùy chọn
        # ---------------------------
        with st.sidebar:
            st.header("Cài đặt mô hình")
            model_choice = st.selectbox(
                        "Chọn mô hình dự đoán", 
                        ("Linear Regression", "Random Forest")
            )
            st.info(f"Đang xử lý dữ liệu và huấn luyện mô hình {model_choice}...")
        # ---------------------------
        # Huấn luyện mô hình
        # ---------------------------
        model, X_test, y_test, numeric_features, categorical_features = train_model(feature_engineering(df_raw), model_choice)
        y_prd = model.predict(X_test)
        # ---------------------------
        #Table layout
        # ---------------------------
        tab1, tab2, tab3 = st.tabs(["Tổng quan & Đánh giá mô hình", "Phân tích dữ liệu", "Dự đoán doanh thu"])
        with tab1:
            st.header("Tổng quan Kinh doanh")
            col1, col2, col3 = st.columns(3)
            col1.metric("Tổng Doanh thu", f"{df_raw['Doanh thu'].sum():,.0f} VND")
            col2.metric("Tổng Số lượng bán", f"{df_raw['Số lượng'].sum():,.0f}")
            col3.metric("Đơn hàng trung bình", f"{df_raw['Doanh thu'].mean():,.0f} VND")
            st.header(f"Đánh giá mô hình: {model_choice}")
            col1, col2, col3 = st.columns(3)
            r2 = r2_score(y_test, y_prd)
            mae = mean_absolute_error(y_test, y_prd)
            rmse = np.sqrt(mean_squared_error(y_test, y_prd))
            col1.metric("R-squared (R²)", f"{r2:.2f}")
            col2.metric("Mean Absolute Error (MAE)", f"{mae:,.0f} VND")
            col3.metric("Root Mean Squared Error (RMSE)", f"{rmse:,.0f} VND")
            st.markdown(f"""
            - **R-squared ($R^2$)**: Mô hình của bạn giải thích được **{r2:.1%}** sự biến thiên của doanh thu. Càng gần 1 càng tốt.
            - **MAE**: Trung bình, dự đoán của mô hình sai lệch khoảng **{mae:,.0f} VND** so với thực tế.
            - **RMSE**: Cho thấy độ lớn của sai số, nhấn mạnh các lỗi dự đoán lớn.
            """)
            #Biểu đồ so sánh
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=y_test, y=y_prd, alpha =0.6, ax=ax)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Đường tham chiếu (Thực tế = Dự đoán)")
            ax.set_xlabel("Doanh thu thực tế")
            ax.set_ylabel("Doanh thu dự đoán")
            ax.set_title(f"So sánh Doanh thu Thực tế và Dự đoán ({model_choice})")
            ax.legend()
            st.pyplot(fig)

        with tab2:
            st.header("Phân tích dữ liệu khám phá (EDA)")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Doanh thu theo tháng")
                monthly_revenue = df.groupby("Tháng")["Doanh thu"].sum()
                fig, ax = plt.subplots(figsize=(10, 5))
                monthly_revenue.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title("Tổng Doanh thu theo Tháng")
                ax.set_xlabel("Tháng")
                ax.set_ylabel("Tổng Doanh thu (VND)")
                st.pyplot(fig)
            with col2:
                st.subheader("Top 10 sản phẩm có Doanh thu cao nhất")
                top_products = df.groupby("Sản phẩm")["Doanh thu"].sum().nlargest(10)
                fig, ax = plt.subplots(figsize=(10, 5))
                top_products.sort_values().plot(kind='barh', ax=ax, color='salmon')
                ax.set_title("Top 10 Sản phẩm theo Doanh thu")
                ax.set_xlabel("Tổng Doanh thu (VND)")
                ax.set_ylabel("Sản phẩm")
                st.pyplot(fig)
        with tab3:
            st.header("Dựa báo doanh thu")
            with st.form("prediction_form"):
                st.subheader("Nhập thông tin sản phẩm để dự đoán doanh thu")
                col1, col2 = st.columns(2)
                with col1:
                    gia = st.number_input("Giá sản phẩm (VND)", min_value=1000, value=500000, step=1000)
                    soluong = st.number_input("Số lượng bán", min_value=1, value=10, step=1)
                    ngay = st.date_input("Ngày bán", value=pd.to_datetime("2023-01-01"))
                with col2:
                    input_data_cat = {}
                    for feature in categorical_features:
                        unique_values = df[feature].unique()
                        input_data_cat[feature] = st.selectbox(f"Chọn {feature}", unique_values)
                submitted = st.form_submit_button("Dự đoán Doanh thu")
            if submitted:
                input_data = {
                    "Giá" : gia,
                    "Số lượng": soluong,
                    "Tháng" : ngay.month,
                    "Quý" : (ngay.month -1)//3 +1,
                    "Ngày trong tuần": ngay.weekday(),
                    "Là cuối tuần": int(ngay.weekday() in [5,6])
                }
                input_data.update(input_data_cat)
                input_df = pd.DataFrame([input_data])
                # Sắp xếp lại cột cho đúng thứ tự
                input_df = input_df[numeric_features + categorical_features]
                predicted_revenue = model.predict(input_df)[0]
                st.success(f"Dự đoán Doanh thu: {predicted_revenue:,.0f} VND")
                #------ BIỂU ĐỒ -------
                st.markdown("----------------------------------")
                st.subheader("Trực quan hóa kết quả dự đoán")
                col1, col2 = st.columns(2)
                with col1:
                    #1. Biểu đồ Cột so sánh
                    st.markdown("**So sánh Doanh thu Dự đoán và Trung bình**")
                    avg_revenue = df['Doanh thu'].mean()
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.barplot(
                        x=['Dự đoán', 'Trung bình'], 
                        y=[predicted_revenue, avg_revenue], 
                        palette='viridis', 
                        ax=ax)
                    ax.set_ylabel("Doanh thu (VND)")
                    ax.set_title("So sánh Doanh thu Dự đoán và Trung bình")
                    for p in ax.patches:
                        ax.annotate(f'{p.get_height():,.0f}', 
                                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
                    st.pyplot(fig)
                with col2:
                    #2. Biểu đồ đường theo số lượng
                    st.markdown("**Dự đoán Doanh thu theo Số lượng bán**")
                    # Tạo một dải số lượng để dự đoán, ví dụ từ 1 đến 50
                    quantity_range = np.arange(max(1, soluong-20), soluong+20, 1)
                    revenue_predictions = []
                    for qty in quantity_range:
                        temp_df = input_df.copy()
                        temp_df['Số lượng'] = qty
                        pred = model.predict(temp_df)[0]
                        revenue_predictions.append(pred)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(quantity_range, revenue_predictions, marker='o', linestyle='--', label='Doanh thu Dự đoán')
                    # Đánh dấu điểm dự đoán hiện tại
                    ax.plot(soluong, predicted_revenue, '*', markersize=15, color = 'red', label=f'Dự đoán hiện tại ({soluong:,.0f} VND)')
                    ax.set_xlabel("Số lượng bán")
                    ax.set_ylabel("Doanh thu (VND)")
                    ax.set_title("Dự đoán Doanh thu theo Số lượng bán")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)