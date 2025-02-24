import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_data, fit_polynomial_models
from utils.visualization import create_channel_analysis_plots

def plot_polynomial_fits(df_seap1, df_seap2):
    """Creates polynomial fit visualizations for SEA channels."""
    fig = go.Figure()
    
    # SEA P1 fit
    x_p1 = df_seap1['Cost'].values
    y_p1 = df_seap1['Applications'].values
    poly_p1 = np.poly1d(np.polyfit(x_p1, y_p1, 2))
    x_new = np.linspace(x_p1.min(), x_p1.max(), 100)
    
    fig.add_trace(go.Scatter(
        x=x_p1, y=y_p1,
        mode='markers',
        name='SEA P1 Data',
        marker=dict(color='blue', size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_new, y=poly_p1(x_new),
        mode='lines',
        name='SEA P1 Fit',
        line=dict(color='blue', dash='dash')
    ))
    
    # SEA P2 fit
    x_p2 = df_seap2['Cost'].values
    y_p2 = df_seap2['Applications'].values
    poly_p2 = np.poly1d(np.polyfit(x_p2, y_p2, 2))
    x_new = np.linspace(x_p2.min(), x_p2.max(), 100)
    
    fig.add_trace(go.Scatter(
        x=x_p2, y=y_p2,
        mode='markers',
        name='SEA P2 Data',
        marker=dict(color='red', size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_new, y=poly_p2(x_new),
        mode='lines',
        name='SEA P2 Fit',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='Polynomial Fits for SEA Channels',
        xaxis_title='Daily Cost (€)',
        yaxis_title='Daily Applications',
        height=500
    )
    
    return fig

def plot_cost_efficiency(df_seap1, df_seap2, df_progd):
    """Creates cost efficiency analysis visualization."""
    fig = go.Figure()
    
    # Calculate cost per application for each channel
    sea_p1_cpa = df_seap1['Cost'] / df_seap1['Applications']
    sea_p2_cpa = df_seap2['Cost'] / df_seap2['Applications']
    prog_cpa = df_progd['Costs'] / df_progd['Applications']
    
    fig.add_trace(go.Box(
        y=sea_p1_cpa,
        name='SEA P1',
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8
    ))
    
    fig.add_trace(go.Box(
        y=sea_p2_cpa,
        name='SEA P2',
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8
    ))
    
    fig.add_trace(go.Box(
        y=prog_cpa,
        name='Programmatic',
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8
    ))
    
    fig.update_layout(
        title='Cost per Application Distribution by Channel',
        yaxis_title='Cost per Application (€)',
        height=500
    )
    
    return fig

def main():
    st.title("Channel Analysis Dashboard")
    
    # Load data
    file_path = Path("data/data_cs.xlsx")
    data = load_data(file_path)
    df_seap1 = data["SEA P1"]
    df_seap2 = data["SEA P2"]
    df_progd = data["Programmatic Display"]
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Channel Comparison", "Cost-Benefit Analysis", "Model Fitting"]
    )
    
    if analysis_type == "Channel Comparison":
        st.subheader("Channel Performance Comparison")
        
        # Plot SEA comparison
        plots = create_channel_analysis_plots(df_seap1, df_seap2, df_progd)
        st.plotly_chart(plots['sea_comparison'], use_container_width=True)
        
        # Display key statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg. SEA P1 Applications", 
                     f"{int(df_seap1['Applications'].mean()):,}")
        with col2:
            st.metric("Avg. SEA P2 Applications", 
                     f"{int(df_seap2['Applications'].mean()):,}")
        with col3:
            st.metric("Avg. Programmatic Applications", 
                     f"{int(df_progd['Applications'].mean()):,}")
    
    elif analysis_type == "Cost-Benefit Analysis":
        st.subheader("Cost Efficiency Analysis")
        
        # Plot cost efficiency
        cost_eff_fig = plot_cost_efficiency(df_seap1, df_seap2, df_progd)
        st.plotly_chart(cost_eff_fig, use_container_width=True)
        
        # Display cost metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            cpa_p1 = df_seap1['Cost'].sum() / df_seap1['Applications'].sum()
            st.metric("SEA P1 CPA", f"€{cpa_p1:.2f}")
        with col2:
            cpa_p2 = df_seap2['Cost'].sum() / df_seap2['Applications'].sum()
            st.metric("SEA P2 CPA", f"€{cpa_p2:.2f}")
        with col3:
            cpa_prog = df_progd['Costs'].sum() / df_progd['Applications'].sum()
            st.metric("Programmatic CPA", f"€{cpa_prog:.2f}")
    
    else:  # Model Fitting
        st.subheader("Polynomial Model Fitting")
        
        # Plot polynomial fits
        poly_fig = plot_polynomial_fits(df_seap1, df_seap2)
        st.plotly_chart(poly_fig, use_container_width=True)
        
        # Model statistics
        st.subheader("Model Statistics")
        
        # Calculate R-squared for both models
        def calc_r2(x, y, poly):
            y_pred = poly(x)
            return 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        
        x_p1, y_p1 = df_seap1['Cost'].values, df_seap1['Applications'].values
        x_p2, y_p2 = df_seap2['Cost'].values, df_seap2['Applications'].values
        
        poly_p1 = np.poly1d(np.polyfit(x_p1, y_p1, 2))
        poly_p2 = np.poly1d(np.polyfit(x_p2, y_p2, 2))
        
        r2_p1 = calc_r2(x_p1, y_p1, poly_p1)
        r2_p2 = calc_r2(x_p2, y_p2, poly_p2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("SEA P1 R²", f"{r2_p1:.3f}")
        with col2:
            st.metric("SEA P2 R²", f"{r2_p2:.3f}")

if __name__ == "__main__":
    main()