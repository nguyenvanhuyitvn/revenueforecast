import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Data Analysis and ML App", layout="wide")
st.title("Data Analysis and Machine Learning Application")
#---- File Upload ----#
st.sidebar.header("Upload your Excel data")
uploaded_file = st.sidebar.file_uploader("Upload your input Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.header("Xem dữ liệu thô")
    st.dataframe(df.head())
    st.subheader("📊 Thông tin cột dữ liệu")   
    st.write(f"👉 Dữ liệu có {df.shape[0]} dòng, {df.shape[1]} cột")
    dtypes_df = df.dtypes.astype(str).reset_index()
    dtypes_df.columns = ["Cột", "Kiểu dữ liệu"]
    st.dataframe(dtypes_df)

    #Kiểm tra giá trị thiếu
    st.subheader("***Kiểm tra giá trị thiếu***")
    describe_null = df.isnull().sum().reset_index()
    describe_null.columns = ["Cột", "Số giá trị thiếu"]
    st.dataframe(describe_null)
    # Thống kê mô tả cơ bản
    st.subheader("📈 Mô tả dữ liệu (describe)")
    st.write(df.describe(include="all").transpose())
    st.subheader("📊 Thống kê mô tả")
    desc = df[["Số lượng", "Giá", "Giảm giá (%)", "Doanh thu"]].describe()
    st.dataframe(desc)
    mean_value = df["Doanh thu"].mean()
    median_value = df["Doanh thu"].median()
    st.write(f"👉 Mean Doanh thu: {mean_value} VND")
    st.write(f"👉 Median Doanh thu: {median_value} VND")

    #--- 3. Histogram Doanh thu ---
    st.header("Histogram Doanh thu- Phân phối Doanh thu")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Doanh thu"], bins=30, kde=True, ax=ax)
    ax.axvline(mean_value, color="red", linestyle="--", label=f"Mean = {mean_value:,.0f}")
    ax.axvline(median_value, color="green", linestyle="--", label=f"Median = {median_value:,.0f}")
    ax.set_title("Phân phối Doanh thu với Mean và Median")
    ax.legend()
    st.pyplot(fig)
    #--- 4. Boxplot Doanh thu ---
    st.header("Boxplot Doanh thu - Phát hiện ngoại lệ")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=df["Doanh thu"], ax=ax2)
    ax2.set_title("Boxplot Doanh thu (Phát hiện ngoại lệ)")
    st.pyplot(fig2)
    #--- 5. Scatter plot Số lượng vs Doanh thu ---
    st.header("Scatter plot Số lượng vs Doanh thu")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x="Số lượng", y="Doanh thu", data=df, ax=ax3)
    ax3.set_title("Số lượng vs Doanh thu")
    st.pyplot(fig3)
    #--- 6. Bar chart - Doanh thu theo khu vực ---
    st.header("Bar chart - Doanh thu theo khu vực")
    revenue_by_region = df.groupby("Khu vực")["Doanh thu"].sum().reset_index()
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Khu vực", y="Doanh thu", data=revenue_by_region, ax=ax4)
    ax4.set_title("Tổng Doanh thu theo Khu vực")
    st.pyplot(fig4)
    #--- 7. Line chart - Doanh thu theo thời gian ---
    st.header("Line chart - Doanh thu theo thời gian")
    df["Ngày"] = pd.to_datetime(df["Ngày"], errors='coerce')
    revenue_over_time = df.groupby("Ngày")["Doanh thu"].sum().reset_index()
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x="Ngày", y="Doanh thu", data=revenue_over_time, ax=ax5)
    ax5.set_title("Tổng Doanh thu theo Thời gian")
    st.pyplot(fig5)
    # --- 9. Dự đoán Doanh thu bằng sklearn ---
    st.header("9️⃣ Dự đoán Doanh thu (Machine Learning)")

    # Chọn các cột để dự báo
    X = df[["Số lượng", "Giá", "Giảm giá (%)", "Danh mục", "Nhà cung cấp", "Khu vực", "Khách hàng", "Phương thức thanh toán"]]
    y = df["Doanh thu"]

    # Xử lý biến categorical
    categorical_cols = ["Danh mục", "Nhà cung cấp", "Khu vực", "Khách hàng", "Phương thức thanh toán"]
    numeric_cols = ["Số lượng", "Giá", "Giảm giá (%)"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols)
        ]
    )

    # Pipeline mô hình
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    #    model = Pipeline(steps=[
    #     ("preprocessor", preprocessor),
    #     ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    # ])


    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện
    model.fit(X_train, y_train)

    # Đánh giá
    y_pred = model.predict(X_test)
    
    # --- Form nhập dữ liệu để dự đoán ---
    st.subheader("🔮 Thử nhập dữ liệu để dự đoán Doanh thu")

    col1, col2 = st.columns(2)
    with col1:
        so_luong_in = st.number_input("Số lượng", min_value=1, max_value=100, value=5)
        gia_in = st.number_input("Giá sản phẩm (VND)", min_value=1000, max_value=50000000, value=2000000)
        giam_gia_in = st.slider("Giảm giá (%)", 0, 30, 5)
    with col2:
        danh_muc_in = st.selectbox("Danh mục", df["Danh mục"].unique())
        nha_cc_in = st.selectbox("Nhà cung cấp", df["Nhà cung cấp"].unique())
        khu_vuc_in = st.selectbox("Khu vực", df["Khu vực"].unique())
        khach_hang_in = st.selectbox("Khách hàng", df["Khách hàng"].unique())
        pttt_in = st.selectbox("Phương thức thanh toán", df["Phương thức thanh toán"].unique())

    # Tạo dataframe input
    input_data = pd.DataFrame({
        "Số lượng": [so_luong_in],
        "Giá": [gia_in],
        "Giảm giá (%)": [giam_gia_in],
        "Danh mục": [danh_muc_in],
        "Nhà cung cấp": [nha_cc_in],
        "Khu vực": [khu_vuc_in],
        "Khách hàng": [khach_hang_in],
        "Phương thức thanh toán": [pttt_in]
    })

    # Dự đoán
    y_pred_input = model.predict(input_data)[0]
    st.success(f"📈 Doanh thu dự đoán: {y_pred_input:,.0f} VND")
    # 2. BIỂU ĐỒ ĐƯỜNG  "What-If" THEO SỐ LƯỢNG
    # input_df = pd.DataFrame([input_data])
    input_df = input_data.copy()
    input_df = input_df[numeric_cols + categorical_cols]
    st.markdown("##### Phân tích nếu thay đổi Số lượng")
    
    # Tạo một dải số lượng để dự đoán, ví dụ từ 1 đến 50
    quantity_range = np.arange(max(1, so_luong_in - 20), so_luong_in + 20, 1)
    what_if_predictions = []
    
    # Lặp qua từng giá trị số lượng để dự đoán
    for qty in quantity_range:
        temp_df = input_data.copy()
        temp_df['Số lượng'] = qty
        pred = model.predict(temp_df)[0]
        what_if_predictions.append(pred)
        
    fig, ax = plt.subplots()
    ax.plot(quantity_range, what_if_predictions, marker='o', linestyle='--', label='Xu hướng Doanh thu')
    # Đánh dấu điểm dự đoán hiện tại
    ax.plot(so_luong_in, y_pred_input, marker='*', markersize=15, color='red', label=f'Dự đoán hiện tại ({so_luong_in} sp)')
    ax.set_xlabel("Số lượng sản phẩm")
    ax.set_ylabel("Doanh thu Dự đoán (VND)")
    ax.legend()
    ax.grid(True)
    ax.ticklabel_format(style='plain', axis='y')
    st.pyplot(fig)
else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích dữ liệu.")