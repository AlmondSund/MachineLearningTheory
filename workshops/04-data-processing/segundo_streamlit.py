import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load the California Housing data
housing = fetch_california_housing(as_frame=True).frame

# Application title
st.title('Data Exploration: California Housing')

# Introductory description
st.write("""
### Welcome
This interactive application allows the California Housing dataset to be explored.
You may:
1. View the first records.
2. Inspect general information about the dataset.
3. Generate dynamic plots.
""")

# Section for data exploration
st.sidebar.header('Data Exploration')

# Dynamically display the first rows
if st.sidebar.checkbox('Show first rows'):
    n_rows = st.sidebar.slider('Number of rows to display:', 1, 50, 5)
    st.write(f'### First {n_rows} rows of the dataset')
    st.write(housing.head(n_rows))

# Display dataset information
import io

if st.sidebar.checkbox('Show dataset information'):
    st.write('### Dataset Information')

    # Capture the output of info() in a buffer
    buffer = io.StringIO()
    housing.info(buf=buffer)

    # Convert the buffer contents into text
    info_text = buffer.getvalue()
    st.text(info_text)

# Descriptive statistics
if st.sidebar.checkbox('Show descriptive statistics'):
    st.write('### Descriptive Statistics')
    st.write(housing.describe())

# Section for dynamic plots
st.sidebar.header('Dynamic Plots')

# Select the variables to plot
x_var = st.sidebar.selectbox('Select the X variable:', housing.columns)
y_var = st.sidebar.selectbox('Select the Y variable:', housing.columns)

# Plot type
chart_type = st.sidebar.radio(
    'Select the plot type:',
    ('Scatter', 'Histogram', 'Boxplot')
)

# Display the plot
st.write('### Plots')
if chart_type == 'Scatter':
    st.write(f'#### Scatter plot: {x_var} vs {y_var}')
    fig, ax = plt.subplots()
    sns.scatterplot(data=housing, x=x_var, y=y_var, ax=ax)
    st.pyplot(fig)
elif chart_type == 'Histogram':
    st.write(f'#### Histogram of {x_var}')
    fig, ax = plt.subplots()
    sns.histplot(housing[x_var], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
elif chart_type == 'Boxplot':
    st.write(f'#### Boxplot of {y_var} by {x_var}')
    fig, ax = plt.subplots()
    sns.boxplot(data=housing, x=x_var, y=y_var, ax=ax)
    st.pyplot(fig)

# Closing message
st.write('Use the sidebar to explore additional options.')
