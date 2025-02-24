import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import sys
import os

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_data, fit_polynomial_models

def estimate_applications(budget_alloc, poly_p1, poly_p2, df_progd):
    """
    Estimates applications for all channels.
    """
    daily_budget_p1 = budget_alloc['SEA P1'] / 365.0
    daily_budget_p2 = budget_alloc['SEA P2'] / 365.0
    
    apps_p1 = poly_p1(daily_budget_p1) * 365.0
    apps_p2 = poly_p2(daily_budget_p2) * 365.0
    
    monthly_budget_prog = budget_alloc['Programmatic'] / 12.0
    apps_prog = np.interp(monthly_budget_prog, 
                         df_progd['Costs'].values, 
                         df_progd['Applications'].values) * 12.0
    
    apps_partner = (budget_alloc['Partner'] / 0.10) * 0.06  # CPC = 0.10€, conversion = 6%
    
    total_apps = apps_p1 + apps_p2 + apps_prog + apps_partner
    
    return {
        'Total Applications': total_apps,
        'Candidates per Listing': total_apps / 10000,  # Assuming 10,000 listings
        'Breakdown': {
            'SEA P1': apps_p1,
            'SEA P2': apps_p2,
            'Programmatic': apps_prog,
            'Partner': apps_partner
        }
    }

def create_sankey_diagram(budget_alloc, channels, total_budget, poly_p1, poly_p2, df_prog, total_listings, metrics):
    """
    Creates a Sankey diagram visualization.
    """
    labels = ["Total Budget"] + channels
    sources = []
    targets = []
    values = []
    
    for i, ch in enumerate(channels):
        sources.append(0)
        targets.append(i + 1)
        values.append(budget_alloc[ch])
    
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels,
            pad=20,
            thickness=20
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])
    
    fig.update_layout(
        title_text=f"Budget Allocation Sankey Diagram - CPL: {metrics['Candidates per Listing']:.2f}",
        font_size=12,
        height=400
    )
    
    return fig

def create_donut_chart(budget_alloc, channels):
    """
    Creates a donut chart for budget allocation.
    """
    values = [budget_alloc[ch] for ch in channels]
    percentages = [v/sum(values)*100 for v in values]
    
    fig = go.Figure(data=[go.Pie(
        labels=channels,
        values=percentages,
        hole=.6,
        textinfo='label+percent',
        marker=dict(colors=['#4daf4a','#377eb8','#ff7f00','#984ea3'])
    )])
    
    fig.update_layout(
        title_text="Budget Distribution",
        annotations=[dict(text='Budget', x=0.5, y=0.5, font_size=20, showarrow=False)],
        height=400
    )
    
    return fig

def update_budget_controls():
    """
    Creates and manages budget allocation input controls.
    """
    st.sidebar.header("Budget Allocation Controls")
    
    # Initialize session state if not exists
    if 'allocations' not in st.session_state:
        st.session_state.allocations = {
            'SEA P1': 25,
            'SEA P2': 25,
            'Programmatic': 25,
            'Partner': 25
        }
    
    # Input fields for each channel
    new_allocations = {}
    for channel in ['SEA P1', 'SEA P2', 'Programmatic', 'Partner']:
        new_allocations[channel] = st.sidebar.number_input(
            f"{channel} (%)",
            min_value=0,
            max_value=100,
            value=int(st.session_state.allocations[channel]),
            step=1,
            key=f"input_{channel}"
        )
    
    # Calculate total and show warning if not 100%
    total_allocation = sum(new_allocations.values())
    
    # Display total with color-coded feedback
    if total_allocation != 100:
        st.sidebar.markdown(
            f"""
            <div style="padding: 10px; border-radius: 5px; background-color: #ff4b4b; color: white;">
            ⚠️ Total allocation: {total_allocation}%<br>
            Total must equal 100%
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            f"""
            <div style="padding: 10px; border-radius: 5px; background-color: #00cc00; color: white;">
            ✓ Total allocation: {total_allocation}%
            </div>
            """,
            unsafe_allow_html=True
        )
    
    return new_allocations, total_allocation == 100

def main():
    st.title("Budget Allocation Dashboard")
    
    # Load and prepare data
    file_path = Path("data/data_cs.xlsx")
    data = load_data(file_path)
    df_seap1 = data["SEA P1"]
    df_seap2 = data["SEA P2"]
    df_progd = data["Programmatic Display"]
    
    # Fit polynomial models
    poly_p1, poly_p2 = fit_polynomial_models(df_seap1, df_seap2)
    
    # Setup
    total_budget = 10_000_000
    channels = ['SEA P1', 'SEA P2', 'Programmatic', 'Partner']
    
    # Get budget allocations and validation status
    allocations, is_valid = update_budget_controls()
    
    # Only update visualizations if allocations are valid
    if is_valid:
        # Calculate budget allocation
        budget_alloc = {
            ch: (allocations[ch]/100) * total_budget 
            for ch in channels
        }
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        # Calculate metrics
        metrics = estimate_applications(budget_alloc, poly_p1, poly_p2, df_progd)
        
        with col1:
            st.plotly_chart(
                create_sankey_diagram(
                    budget_alloc, channels, total_budget,
                    poly_p1, poly_p2, df_progd, 10000, metrics
                ),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                create_donut_chart(budget_alloc, channels),
                use_container_width=True
            )
        
        # Display metrics
        st.subheader("Key Metrics")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Total Applications", f"{int(metrics['Total Applications']):,}")
        with metric_col2:
            st.metric("Candidates per Listing", f"{metrics['Candidates per Listing']:.2f}")
        
        # Display breakdown
        st.subheader("Channel Breakdown")
        breakdown_df = pd.DataFrame({
            'Channel': channels,
            'Budget (€)': [f"{budget_alloc[ch]:,.2f}" for ch in channels],
            'Applications': [f"{int(metrics['Breakdown'][ch]):,}" for ch in channels],
            'Percentage': [f"{allocations[ch]:.1f}%" for ch in channels]
        })
        st.dataframe(breakdown_df, use_container_width=True)
    else:
        st.warning("Please adjust the allocations to total 100% to view the analysis.")

if __name__ == "__main__":
    main()