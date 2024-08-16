import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# Function to load data
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Function to create sample data
def create_sample_data():
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    categories = ['A', 'B', 'C', 'D']
    data = {
        'Date': dates.repeat(len(categories)),
        'Category': categories * len(dates),
        'Value': np.random.randint(1, 100, size=len(dates) * len(categories)),
        'Value2': np.random.randint(1, 100, size=len(dates) * len(categories))
    }
    return pd.DataFrame(data)

# Function for data transformation
def transform_data(data, columns, method):
    if method == "Min-Max Normalization":
        scaler = MinMaxScaler()
    elif method == "Standard Scaling":
        scaler = StandardScaler()
    elif method == "Log Transform":
        return data[columns].apply(np.log1p)
    else:
        return data[columns]
    
    return pd.DataFrame(scaler.fit_transform(data[columns]), columns=columns, index=data.index)

# Function for time series forecasting
def forecast_time_series(data, date_column, value_column, periods=30):
    data = data.sort_values(date_column)
    data = data.set_index(date_column)
    data = data[value_column].resample('D').mean()
    
    model = ARIMA(data, order=(1,1,1))
    results = model.fit()
    forecast = results.forecast(steps=periods)
    
    return forecast

# Main function
def main():
    st.set_page_config(layout="wide", page_title="Advanced Data Viz Dashboard")
    st.title("Advanced Interactive Data Visualization Dashboard")

    # Sidebar for data loading and global controls
    with st.sidebar:
        st.header("Data Loading")
        data_source = st.radio("Choose data source", ["Upload CSV", "Use Sample Data"])
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                data = load_data(uploaded_file)
            else:
                st.info("Please upload a CSV file to proceed.")
                return
        else:
            data = create_sample_data()

        st.header("Global Controls")
        chart_type = st.selectbox("Select Chart Type", [
            "Line", "Bar", "Scatter", "Heatmap", "Pie", "Box", "Violin", "3D Scatter"
        ])
        color_scheme = st.selectbox("Color Scheme", ["Viridis", "Plasma", "Inferno", "Magma"])

    # Main area
    if data is not None:
        st.write("### Data Preview")
        st.dataframe(data.head())

        # Data transformation
        st.write("### Data Transformation")
        transform_columns = st.multiselect("Select columns for transformation", data.select_dtypes(include=['float64', 'int64']).columns)
        transform_method = st.selectbox("Select transformation method", ["None", "Min-Max Normalization", "Standard Scaling", "Log Transform"])
        
        if transform_columns and transform_method != "None":
            data[transform_columns] = transform_data(data, transform_columns, transform_method)
            st.success(f"Applied {transform_method} to selected columns.")

        # Data filtering
        st.write("### Data Filtering")
        col1, col2, col3 = st.columns(3)
        with col1:
            date_columns = data.select_dtypes(include=['datetime64', 'object']).columns
            date_column = st.selectbox("Select Date Column", date_columns)
            if date_column:
                data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
                min_date, max_date = data[date_column].min(), data[date_column].max()
                date_range = st.date_input("Select Date Range", [min_date, max_date])
        with col2:
            categorical_column = st.selectbox("Select Categorical Column", data.select_dtypes(include=['object']).columns)
            if categorical_column:
                categories = st.multiselect("Select Categories", data[categorical_column].unique())
        with col3:
            numeric_column = st.selectbox("Select Numeric Column", data.select_dtypes(include=['float64', 'int64']).columns)
            if numeric_column:
                min_val, max_val = float(data[numeric_column].min()), float(data[numeric_column].max())
                value_range = st.slider("Select Value Range", min_val, max_val, (min_val, max_val))

        # Apply filters
        filtered_data = data.copy()
        if date_column and len(date_range) == 2:
            filtered_data = filtered_data[(filtered_data[date_column].dt.date >= date_range[0]) & (filtered_data[date_column].dt.date <= date_range[1])]
        if categorical_column and categories:
            filtered_data = filtered_data[filtered_data[categorical_column].isin(categories)]
        if numeric_column:
            filtered_data = filtered_data[(filtered_data[numeric_column] >= value_range[0]) & (filtered_data[numeric_column] <= value_range[1])]

        # Visualization
        st.write("### Data Visualization")
        fig = None
        if chart_type == "Line":
            fig = px.line(filtered_data, x=date_column, y=numeric_column, color=categorical_column, color_discrete_sequence=px.colors.sequential.Viridis)
        elif chart_type == "Bar":
            fig = px.bar(filtered_data, x=categorical_column, y=numeric_column, color=categorical_column, color_discrete_sequence=px.colors.sequential.Viridis)
        elif chart_type == "Scatter":
            fig = px.scatter(filtered_data, x=numeric_column, y=numeric_column, color=categorical_column, color_discrete_sequence=px.colors.sequential.Viridis)
        elif chart_type == "Heatmap":
            pivot = filtered_data.pivot_table(values=numeric_column, index=categorical_column, columns=date_column, aggfunc='mean')
            fig = px.imshow(pivot, color_continuous_scale=color_scheme)
        elif chart_type == "Pie":
            fig = px.pie(filtered_data, values=numeric_column, names=categorical_column, color_discrete_sequence=px.colors.sequential.Viridis)
        elif chart_type == "Box":
            fig = px.box(filtered_data, x=categorical_column, y=numeric_column, color=categorical_column, color_discrete_sequence=px.colors.sequential.Viridis)
        elif chart_type == "Violin":
            fig = px.violin(filtered_data, x=categorical_column, y=numeric_column, color=categorical_column, box=True, points="all", color_discrete_sequence=px.colors.sequential.Viridis)
        elif chart_type == "3D Scatter":
            numeric_columns = filtered_data.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_columns) >= 3:
                fig = px.scatter_3d(filtered_data, x=numeric_columns[0], y=numeric_columns[1], z=numeric_columns[2],
                                    color=categorical_column, color_discrete_sequence=px.colors.sequential.Viridis)

        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Time series forecasting
        if date_column and numeric_column:
            st.write("### Time Series Forecasting")
            forecast_periods = st.slider("Select number of periods to forecast", 1, 365, 30)
            if st.button("Generate Forecast"):
                with st.spinner("Generating forecast..."):
                    forecast = forecast_time_series(filtered_data, date_column, numeric_column, periods=forecast_periods)
                    fig_forecast = go.Figure()
                    fig_forecast.add_trace(go.Scatter(x=filtered_data[date_column], y=filtered_data[numeric_column], mode='lines', name='Historical'))
                    fig_forecast.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines', name='Forecast'))
                    fig_forecast.update_layout(title=f"{numeric_column} Forecast", xaxis_title="Date", yaxis_title=numeric_column)
                    st.plotly_chart(fig_forecast, use_container_width=True)

        # Data statistics
        st.write("### Data Statistics")
        st.write(filtered_data.describe())

        # Correlation matrix
        st.write("### Correlation Matrix")
        corr_matrix = filtered_data.select_dtypes(include=['float64', 'int64']).corr()
        fig_corr = px.imshow(corr_matrix, color_continuous_scale=color_scheme)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Export options
        st.write("### Export Options")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export Visualization"):
                buffer = BytesIO()
                fig.write_image(buffer, format="png")
                st.download_button(
                    label="Download Visualization",
                    data=buffer,
                    file_name="visualization.png",
                    mime="image/png"
                )
        with col2:
            if st.button("Export Filtered Data"):
                csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=filtered_data.to_csv(index=False),
                    file_name="filtered_data.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()