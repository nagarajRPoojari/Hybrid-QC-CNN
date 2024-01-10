import streamlit as st
st.set_page_config(layout="wide")
st.title("this is a bmsit project")
st.image("https://imgs.search.brave.com/Y2-lY7NA7o2rti-Uned1jQyOxQEs3XRFn9CT2z47chg/rs:fit:500:0:0/g:ce/aHR0cHM6Ly93YWxs/cGFwZXJjYXZlLmNv/bS93cC93cDY5NTA1/NDUuanBn")

import pandas as pd
import plotly.express as px

# Read data from CSV file
csv_file_path = "HybridResnet18.csv"
df = pd.read_csv(csv_file_path)

# Create a line chart with Plotly Express
fig = px.line(df, x='Epochs', y=['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'],
              labels={'value': 'Metrics'},
              title='Training and Validation Metrics Over Epochs for resnet 18')

# Customize the layout
fig.update_layout(
    xaxis_title='Epochs',
    yaxis_title='Metrics',
    legend_title='Metrics'
)

# Display the Plotly chart in Streamlit
st.plotly_chart(fig)
csv_file_path2 = "HybridResnet50.csv"
df2= pd.read_csv(csv_file_path2)

# Create a line chart with Plotly Express
fig2 = px.line(df2, x='Epochs', y=['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'],
              labels={'value': 'Metrics'},
              title='Training and Validation Metrics Over Epochs for resnet 50')

# Customize the layout
fig2.update_layout(
    xaxis_title='Epochs',
    yaxis_title='Metrics',
    legend_title='Metrics'
)

# Display the Plotly chart in Streamlit
st.plotly_chart(fig2)
