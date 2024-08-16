# Advanced Interactive Data Visualization Dashboard

## Overview

This project is an advanced, interactive data visualization dashboard built with Streamlit. It allows users to upload their own CSV data or use sample data, apply various transformations, create different types of visualizations, and even perform basic time series forecasting.

## Features

- **Data Loading**: Upload CSV files or use generated sample data.
- **Data Transformation**: Apply Min-Max Normalization, Standard Scaling, or Log Transform to numeric columns.
- **Interactive Filtering**: Filter data by date range, categories, and numeric ranges.
- **Multiple Chart Types**:
  - Line Chart
  - Bar Chart
  - Scatter Plot
  - Heatmap
  - Pie Chart
  - Box Plot
  - Violin Plot
  - 3D Scatter Plot
- **Time Series Forecasting**: Use ARIMA model to forecast future values.
- **Data Statistics**: View basic statistics of the data.
- **Correlation Matrix**: Visualize correlations between numeric columns.
- **Export Options**: Download visualizations as PNG and filtered data as CSV.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/data-viz-dashboard.git
   cd data-viz-dashboard
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install streamlit pandas numpy plotly scikit-learn statsmodels
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to `http://localhost:8501` (or the URL provided in the terminal).

3. Use the sidebar to:
   - Choose between uploading a CSV file or using sample data.
   - Select the chart type and color scheme.

4. In the main area:
   - Preview your data.
   - Apply data transformations.
   - Filter your data.
   - View and interact with the generated visualization.
   - Perform time series forecasting (if applicable).
   - View data statistics and correlation matrix.
   - Export visualizations and filtered data.

## Data Format

If you're uploading your own CSV file, ensure it has at least:
- One column that can be parsed as dates
- One categorical column
- One or more numeric columns

## Contributing

Contributions to improve the dashboard are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

If you have any questions or feedback, please open an issue in this repository.

---

Happy data visualizing!
