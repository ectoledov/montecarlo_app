import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
from montecarlo_engine import run_simulation

st.set_page_config(page_title="Monte Carlo Schedule Simulator", layout="wide")
st.title("📊 Monte Carlo Schedule Risk Analysis")

st.markdown("""
Upload your schedule Excel file using the official template. This should include:
- Sheet `Activities_Template`: Activity ID, Durations, Distribution
- Sheet `Logic_Template`: From, To, Link Type, Lag
""")

uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type="xlsx")
iterations = st.slider("Number of Iterations", min_value=10, max_value=5000, step=10, value=1000)

if uploaded_file:
    with st.spinner("Running simulation..."):
        results = run_simulation(uploaded_file, iterations)

        st.success("Simulation complete!")

        # Show Summary Table
        st.subheader("Summary Statistics")
        st.dataframe(results['summary'])

        # Distribution Chart
        st.subheader("Project Duration Distribution")
        fig1 = results['distribution_chart']
        st.pyplot(fig1)

        # Tornado Chart
        st.subheader("Tornado Chart: Criticality Ranking")
        fig2 = results['tornado_chart']
        fig2.axes[0].legend(["Criticality"], loc='lower right')
        st.pyplot(fig2)

        # Download Excel Output
        st.download_button(
            label="📥 Download Full Excel Output",
            data=results['excel_bytes'],
            file_name='MonteCarlo_Output.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
