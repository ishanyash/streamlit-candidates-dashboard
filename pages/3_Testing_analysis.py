import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_data

def simulate_incrementality_test(base_conversions, incrementality_rate, noise_factor=0.1, n_days=30):
    """
    Simulates incrementality test data for test and control groups.
    
    Parameters:
    - base_conversions: Average daily organic conversions
    - incrementality_rate: True incremental effect of the channel
    - noise_factor: Random variation in daily conversions
    - n_days: Number of days to simulate
    """
    # Generate daily organic conversions with noise
    organic = np.random.normal(base_conversions, base_conversions * noise_factor, n_days)
    
    # Generate test group data (organic + incremental effect)
    test_group = organic * (1 + incrementality_rate) + np.random.normal(0, base_conversions * noise_factor, n_days)
    
    # Control group just gets organic conversions with noise
    control_group = organic + np.random.normal(0, base_conversions * noise_factor, n_days)
    
    dates = [datetime.now() + timedelta(days=x) for x in range(n_days)]
    
    return pd.DataFrame({
        'date': dates,
        'test_group': np.maximum(test_group, 0),  # Ensure no negative conversions
        'control_group': np.maximum(control_group, 0)
    })

def simulate_attribution_data(total_conversions, channel_weights, n_days=30):
    """
    Simulates multi-touch attribution data.
    
    Parameters:
    - total_conversions: Average daily total conversions
    - channel_weights: Dictionary of channel names and their true contribution weights
    - n_days: Number of days to simulate
    """
    channels = list(channel_weights.keys())
    data = []
    
    for day in range(n_days):
        daily_conv = np.random.poisson(total_conversions)
        
        # Distribute conversions across channels with some noise
        for _ in range(daily_conv):
            # Randomly select channels that contributed to this conversion
            n_touchpoints = np.random.randint(1, len(channels) + 1)
            touching_channels = np.random.choice(channels, n_touchpoints, replace=False)
            
            # Create a row for this conversion
            conversion_data = {ch: 1 if ch in touching_channels else 0 for ch in channels}
            conversion_data['date'] = datetime.now() + timedelta(days=day)
            conversion_data['conversion_value'] = np.random.lognormal(4, 0.5)  # Random conversion value
            data.append(conversion_data)
    
    return pd.DataFrame(data)

def simulate_ab_test(base_conv_rate, lift, sample_size):
    """
    Simulates A/B test results for a channel optimization.
    
    Parameters:
    - base_conv_rate: Base conversion rate
    - lift: Expected lift from the treatment
    - sample_size: Number of users in each group
    """
    # Control group conversions
    control_conv = np.random.binomial(sample_size, base_conv_rate)
    
    # Treatment group conversions (with lift)
    treatment_conv = np.random.binomial(sample_size, base_conv_rate * (1 + lift))
    
    return {
        'control': {
            'users': sample_size,
            'conversions': control_conv,
            'conv_rate': control_conv / sample_size
        },
        'treatment': {
            'users': sample_size,
            'conversions': treatment_conv,
            'conv_rate': treatment_conv / sample_size
        }
    }

def main():
    st.title("Marketing Channel Testing Analysis")
    
    # Define test types
    test_type = st.sidebar.selectbox(
        "Select Test Type",
        ["Incrementality Testing", "Attribution Analysis", "A/B Testing"]
    )
    
    st.sidebar.markdown("""
    ### Key Assumptions:
    1. Organic baseline is stable
    2. No seasonal effects
    3. No cross-channel interaction
    4. Linear channel effects
    5. Normal distribution of noise
    """)
    
    if test_type == "Incrementality Testing":
        st.subheader("Incrementality Test Simulation")
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            channel = st.selectbox("Select Channel", 
                                 ["SEA P1", "SEA P2", "Programmatic", "Partner"])
            base_conv = st.number_input("Base Daily Conversions", 
                                      value=100, min_value=1)
        with col2:
            incr_rate = st.slider("True Incrementality Rate", 
                                 min_value=0.0, max_value=1.0, value=0.3)
            noise = st.slider("Noise Factor", 
                            min_value=0.0, max_value=0.5, value=0.1)
        with col3:
            days = st.number_input("Test Duration (Days)", 
                                 value=30, min_value=5, max_value=90)
        
        # Simulate data
        test_data = simulate_incrementality_test(base_conv, incr_rate, noise, days)
        
        # Plot results
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_data['date'], y=test_data['test_group'],
                               name='Test Group', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=test_data['date'], y=test_data['control_group'],
                               name='Control Group', mode='lines+markers'))
        
        fig.update_layout(title=f"Incrementality Test Results - {channel}",
                         xaxis_title="Date",
                         yaxis_title="Daily Conversions")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display metrics
        avg_lift = (test_data['test_group'].mean() - test_data['control_group'].mean()) / test_data['control_group'].mean()
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Measured Lift", f"{avg_lift:.1%}")
        with metric_col2:
            st.metric("True Incrementality", f"{incr_rate:.1%}")
        with metric_col3:
            st.metric("Measurement Error", f"{(avg_lift - incr_rate):.1%}")
    
    elif test_type == "Attribution Analysis":
        st.subheader("Multi-Touch Attribution Simulation")
        
        # Parameters
        true_weights = {
            'SEA P1': 0.3,
            'SEA P2': 0.25,
            'Programmatic': 0.25,
            'Partner': 0.2
        }
        
        conv_per_day = st.slider("Average Daily Conversions", 
                                min_value=50, max_value=500, value=200)
        
        # Simulate attribution data
        attr_data = simulate_attribution_data(conv_per_day, true_weights)
        
        # Calculate different attribution models
        last_touch = attr_data[list(true_weights.keys())].sum()
        even_touch = attr_data[list(true_weights.keys())].mean()
        
        # Display attribution comparison
        attribution_comparison = pd.DataFrame({
            'Channel': true_weights.keys(),
            'True Weight': true_weights.values(),
            'Last Touch': last_touch / last_touch.sum(),
            'Even Touch': even_touch / even_touch.sum()
        })
        
        st.write("Attribution Model Comparison:")
        st.dataframe(attribution_comparison)
        
        # Visualization
        fig = go.Figure()
        for model in ['True Weight', 'Last Touch', 'Even Touch']:
            fig.add_trace(go.Bar(name=model, x=attribution_comparison['Channel'],
                               y=attribution_comparison[model]))
        
        fig.update_layout(barmode='group', title="Attribution Model Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # A/B Testing
            st.subheader("A/B Test Simulation")
            
            # Parameters
            col1, col2 = st.columns(2)
            with col1:
                base_rate = st.slider("Base Conversion Rate", 
                                    min_value=0.01, max_value=0.20, value=0.05)
                expected_lift = st.slider("Expected Lift", 
                                        min_value=0.0, max_value=1.0, value=0.15)
            with col2:
                sample_size = st.number_input("Sample Size per Group", 
                                            value=10000, min_value=1000)
            
            # Run simulation
            ab_results = simulate_ab_test(base_rate, expected_lift, sample_size)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Control Conversion Rate", 
                        f"{ab_results['control']['conv_rate']:.2%}")
                st.metric("Control Conversions", 
                        ab_results['control']['conversions'])
            with col2:
                st.metric("Treatment Conversion Rate", 
                        f"{ab_results['treatment']['conv_rate']:.2%}")
                st.metric("Treatment Conversions", 
                        ab_results['treatment']['conversions'])
            
            # Calculate and display lift
            measured_lift = (ab_results['treatment']['conv_rate'] - 
                            ab_results['control']['conv_rate']) / ab_results['control']['conv_rate']
            
            st.metric("Measured Lift", f"{measured_lift:.2%}")
            
            # Statistical significance using chi-square test
            from scipy import stats
            
            # Create contingency table
            contingency = [
                [ab_results['control']['conversions'], 
                ab_results['control']['users'] - ab_results['control']['conversions']],
                [ab_results['treatment']['conversions'], 
                ab_results['treatment']['users'] - ab_results['treatment']['conversions']]
            ]
            
            chi2, p_val = stats.chi2_contingency(contingency)[:2]
            
            # Display statistical significance results
            st.subheader("Statistical Significance")
            
            significance_cols = st.columns(3)
            with significance_cols[0]:
                st.metric("Chi-Square Statistic", f"{chi2:.2f}")
            with significance_cols[1]:
                st.metric("P-value", f"{p_val:.4f}")
            with significance_cols[2]:
                st.metric("Significant at α=0.05", 
                        "Yes" if p_val < 0.05 else "No")
            
            # Visualization of results
            fig = go.Figure()
            
            # Add bars for conversion rates
            fig.add_trace(go.Bar(
                x=['Control', 'Treatment'],
                y=[ab_results['control']['conv_rate'], 
                ab_results['treatment']['conv_rate']],
                text=[f"{ab_results['control']['conv_rate']:.2%}", 
                    f"{ab_results['treatment']['conv_rate']:.2%}"],
                textposition='auto',
                name='Conversion Rate'
            ))
            
            fig.update_layout(
                title='A/B Test Results Comparison',
                yaxis_title='Conversion Rate',
                yaxis_tickformat=',.1%',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional insights
            st.subheader("Test Insights")
            
            # Calculate total sample size correctly
            total_sample_size = ab_results['control']['users'] + ab_results['treatment']['users']
            
            # Calculate effect size (Cramér's V)
            effect_size = np.sqrt(chi2 / (total_sample_size))
            
            insights_cols = st.columns(2)
            with insights_cols[0]:
                st.metric("Effect Size (Cramér's V)", f"{effect_size:.3f}")
                
                # Interpret effect size
                effect_interpretation = (
                    "Small effect size" if effect_size < 0.1
                    else "Medium effect size" if effect_size < 0.3
                    else "Large effect size"
                )
                st.write(effect_interpretation)
            
            with insights_cols[1]:
                relative_risk = (ab_results['treatment']['conv_rate'] / 
                            ab_results['control']['conv_rate'])
                st.metric("Relative Risk", f"{relative_risk:.2f}")
                st.write(f"Treatment group is {relative_risk:.2f}x more likely to convert")
                
if __name__ == "__main__":
    main()