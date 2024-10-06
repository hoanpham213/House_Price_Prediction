import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Tải các mô hình đã huấn luyện
model_dir = os.path.join(os.path.dirname(__file__), 'models')
lr_model = joblib.load(os.path.join(model_dir, 'linear_regression_model.joblib'))
ridge_model = joblib.load(os.path.join(model_dir, 'ridge_regression_model.joblib'))
mlp_model = joblib.load(os.path.join(model_dir, 'mlp_regressor_model.joblib'))
stacking_model = joblib.load(os.path.join(model_dir, 'stacking_regressor_model.joblib'))

# Tạo dictionary cho các mô hình
models = {
    'Linear Regression': lr_model,
    'Ridge Regression': ridge_model,
    'Neural Network': mlp_model,
    'Stacking Regressor': stacking_model
}

# Bắt đầu Streamlit app
st.title("House Price Prediction App")

# Tạo form nhập liệu
st.header("Input data for prediction")

# Tạo các trường nhập liệu cho người dùng
longitude = st.number_input('Longitude', value=0.0)
latitude = st.number_input('Latitude', value=0.0)
housing_median_age = st.number_input('Housing Median Age', value=0.0)
total_rooms = st.number_input('Total Rooms', value=0.0)
total_bedrooms = st.number_input('Total Bedrooms', value=0.0)
population = st.number_input('Population', value=0.0)
households = st.number_input('Households', value=0.0)
median_income = st.number_input('Median Income', value=0.0)

# Biến phân loại ocean_proximity
ocean_proximity = st.selectbox('Ocean Proximity', ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'])

# Lựa chọn mô hình
model_name = st.selectbox(
    'Mô hình dự đoán',
    ['Linear Regression', 'Ridge Regression', 'Neural Network', 'Stacking Regressor']
)

# Define the numeric columns (these are the columns you want to convert to numeric)
numeric_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                   'population', 'households', 'median_income']

# Khi người dùng nhấn vào nút "Dự đoán"
if st.button('Dự đoán'):
    # Tạo DataFrame từ dữ liệu người dùng nhập vào
    input_data = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity
    }

    # Convert dictionary to DataFrame for prediction
    input_df = pd.DataFrame([input_data])

    # Make sure all the values are correctly converted to numeric
    input_df[numeric_columns] = input_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Dự đoán kết quả
    try:
        model = models[model_name]
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted house price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

    # Thêm thông tin về độ tin cậy của mô hình
    model_scores = {
        'Linear Regression': 69043.17,
        'Ridge Regression': 69043.17,
        'Neural Network': 56023.45,
        'Stacking Regressor': 55012.34
    }
    confidence = model_scores.get(model_name)
    st.info(f"Model confidence (RMSE): {confidence}")
