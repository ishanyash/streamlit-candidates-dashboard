import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import sys
import os

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_data

def simulate_incrementality_test(base_conversions, incrementality_rate, noise_factor=0.1, n_days=30):
    """
    Simulates incrementality test data for test and control groups.
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
        'test_group': np.maximum(test_group, 0),
        'control_group': np.maximum(control_group, 0)
    })

def simulate_attribution_data(total_conversions, channel_weights, n_days=30):
    """
    Simulates multi-touch attribution data.
    """
    channels = list(channel_weights.keys())
    data = []
    
    for day in range(n_days):
        daily_conv = np.random.poisson(total_conversions)
        
        for _ in range(daily_conv):
            n_touchpoints = np.random.randint(1, len(channels) + 1)
            touching_channels = np.random.choice(channels, n_touchpoints, replace=False)
            
            conversion_data = {ch: 1 if ch in touching_channels else 0 for ch in channels}
            conversion_data['date'] = datetime.now() + timedelta(days=day)
            conversion_data['conversion_value'] = np.random.lognormal(4, 0.5)
            data.append(conversion_data)
    
    return pd.DataFrame(data)

def simulate_ab_test(base_conv_rate, lift, sample_size):
    """
    Simulates A/B test results.
    """
    control_conv = np.random.binomial(sample_size, base_conv_rate)
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

def simulate_mmm_data(n_periods=52):
    """
    Simulates marketing mix modeling data.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=n_periods, freq='W')
    
    # Generate channel spends
    data = pd.DataFrame({
        'Date': dates,
        'SEA_P1_Spend': np.random.normal(10000, 2000, n_periods),
        'SEA_P2_Spend': np.random.normal(8000, 1500, n_periods),
        'Programmatic_Spend': np.random.normal(12000, 2500, n_periods),
        'Partner_Spend': np.random.normal(9000, 1800, n_periods)
    })
    
    # Add external factors
    data['Seasonality'] = np.sin(2 * np.pi * np.arange(n_periods) / 52) + 1
    data['Competition_Index'] = np.random.normal(100, 10, n_periods)
    data['Market_Demand'] = np.random.normal(1000, 100, n_periods)
    
    # Generate channel effects with diminishing returns
    def channel_effect(spend, power=0.6, scale=2.0):
        return scale * np.power(spend, power)
    
    # Calculate total applications
    data['Baseline'] = 500 * data['Seasonality'] * (data['Market_Demand'] / 1000)
    
    data['Applications'] = (
        data['Baseline'] +
        channel_effect(data['SEA_P1_Spend']) * (1 + 0.1 * np.random.randn(n_periods)) +
        channel_effect(data['SEA_P2_Spend']) * (1 + 0.1 * np.random.randn(n_periods)) +
        channel_effect(data['Programmatic_Spend']) * (1 + 0.15 * np.random.randn(n_periods)) +
        channel_effect(data['Partner_Spend']) * (1 + 0.12 * np.random.randn(n_periods))
    )
    
    return data

def fit_mmm_model(data):
    """
    Fits a marketing mix model to the simulated data.
    """
    features = [
        'SEA_P1_Spend', 'SEA_P2_Spend', 'Programmatic_Spend', 'Partner_Spend',
        'Seasonality', 'Competition_Index', 'Market_Demand'
    ]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])
    y = data['Applications']
    
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    contributions = pd.DataFrame({
        'Channel': features,
        'Coefficient': model.coef_,
        'Std_Coefficient': model.coef_ * np.std(data[features], axis=0)
    })
    
    return {
        'model': model,
        'scaler': scaler,
        'r2': r2,
        'contributions': contributions,
        'predictions': y_pred
    }

def plot_mmm_results(data, mmm_results):
    """
    Creates visualizations for MMM results.
    """
    # Contribution plot
    fig_contrib = go.Figure()
    contrib_data = mmm_results['contributions']
    
    fig_contrib.add_trace(go.Bar(
        x=contrib_data['Channel'],
        y=contrib_data['Std_Coefficient'],
        name='Channel Contribution'
    ))
    
    fig_contrib.update_layout(
        title='Marketing Channel Contributions',
        xaxis_title='Channel',
        yaxis_title='Standardized Coefficient',
        height=400
    )
    
    # Actual vs predicted plot
    fig_fit = go.Figure()
    
    fig_fit.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Applications'],
        name='Actual',
        mode='lines'
    ))
    
    fig_fit.add_trace(go.Scatter(
        x=data['Date'],
        y=mmm_results['predictions'],
        name='Predicted',
        mode='lines'
    ))
    
    fig_fit.update_layout(
        title='Model Fit: Actual vs Predicted Applications',
        xaxis_title='Date',
        yaxis_title='Applications',
        height=400
    )
    
    return fig_contrib, fig_fit

def main():
    st.title("Testing Analysis Dashboard")
    
    test_type = st.sidebar.selectbox(
        "Select Test Type",
        ["Incrementality Testing", "Attribution Analysis", "A/B Testing", "Marketing Mix Modeling"]
    )
    
    if test_type == "Incrementality Testing":
        st.subheader("Incrementality Test Simulation")
        
        st.info("""
        **Key Assumptions in Incrementality Testing:**
        
        1. **Test Design**
           - Clean separation between test and control groups
           - No contamination between groups
           - Sufficient test duration (minimum 30 days)
        
        2. **Data Quality**
           - Stable baseline conversion rate
           - Normal distribution of noise in conversions
           - No major external events during test period
        
        3. **Channel Independence**
           - No significant cross-channel effects
           - Test group only sees additional marketing in tested channel
           - Control group behavior remains consistent
        
        4. **Market Conditions**
           - Stable competitive environment
           - No seasonal effects (unless accounted for)
           - No major market disruptions during test
        """)
        
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
        
        test_data = simulate_incrementality_test(base_conv, incr_rate, noise, days)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_data['date'], y=test_data['test_group'],
                               name='Test Group', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=test_data['date'], y=test_data['control_group'],
                               name='Control Group', mode='lines+markers'))
        
        fig.update_layout(title=f"Incrementality Test Results - {channel}",
                         xaxis_title="Date",
                         yaxis_title="Daily Conversions")
        
        st.plotly_chart(fig, use_container_width=True)
        
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
        
        st.info("""
        **Key Assumptions in Attribution Analysis:**
        
        1. **User Journey**
           - All touchpoints are correctly tracked and recorded
           - Users can be uniquely identified across channels
           - Channel order matters in conversion path
        
        2. **Channel Impact**
           - Each channel has independent contribution
           - No offline channel influence
           - Fixed conversion window length
        
        3. **Data Collection**
           - Complete user journey data
           - Accurate timestamp for each interaction
           - Reliable channel identification
        
        4. **Model Assumptions**
           - Consistent attribution rules across channels
           - No external factors influence conversion probability
           - Static channel effectiveness over time
        """)
        
        true_weights = {
            'SEA P1': 0.3,
            'SEA P2': 0.25,
            'Programmatic': 0.25,
            'Partner': 0.2
        }
        
        conv_per_day = st.slider("Average Daily Conversions", 
                                min_value=50, max_value=500, value=200)
        
        attr_data = simulate_attribution_data(conv_per_day, true_weights)
        
        last_touch = attr_data[list(true_weights.keys())].sum()
        even_touch = attr_data[list(true_weights.keys())].mean()
        
        attribution_comparison = pd.DataFrame({
            'Channel': true_weights.keys(),
            'True Weight': true_weights.values(),
            'Last Touch': last_touch / last_touch.sum(),
            'Even Touch': even_touch / even_touch.sum()
        })
        
        st.write("Attribution Model Comparison:")
        formatted_df = attribution_comparison.style.format({
            'True Weight': '{:.2%}',
            'Last Touch': '{:.2%}',
            'Even Touch': '{:.2%}'
        })
        st.dataframe(formatted_df)
        
        fig = go.Figure()
        for model in ['True Weight', 'Last Touch', 'Even Touch']:
            fig.add_trace(go.Bar(name=model, x=attribution_comparison['Channel'],
                               y=attribution_comparison[model]))
        
        fig.update_layout(
            barmode='group', 
            title="Attribution Model Comparison",
            yaxis_tickformat='.1%'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif test_type == "A/B Testing":
        st.subheader("A/B Test Simulation")
        
        st.info("""
        **Key Assumptions in A/B Testing:**
        
        1. **Sample Design**
           - Random assignment to test groups
           - Large enough sample size
           - No selection bias
        
        2. **Statistical Assumptions**
           - Independent observations
           - Fixed sample size
           - No multiple testing issues
        
        3. **Test Conditions**
           - No interaction between groups
           - Stable test conditions throughout
           - Single variant testing
        
        4. **Measurement**
           - Accurate conversion tracking
           - Consistent measurement across groups
           - No technical issues affecting data collection
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            base_rate = st.slider("Base Conversion Rate", 
                                min_value=0.01, max_value=0.20, value=0.05)
            expected_lift = st.slider("Expected Lift", 
                                    min_value=0.0, max_value=1.0, value=0.15)
        with col2:
            sample_size = st.number_input("Sample Size per Group", 
                                        value=10000, min_value=1000)
        
        ab_results = simulate_ab_test(base_rate, expected_lift, sample_size)
        
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
        
        measured_lift = (ab_results['treatment']['conv_rate'] - 
                        ab_results['control']['conv_rate']) / ab_results['control']['conv_rate']
        
        st.metric("Measured Lift", f"{measured_lift:.2%}")
        
        contingency = [
            [ab_results['control']['conversions'], 
             ab_results['control']['users'] - ab_results['control']['conversions']],
            [ab_results['treatment']['conversions'], 
             ab_results['treatment']['users'] - ab_results['treatment']['conversions']]
        ]
        
        chi2, p_val = stats.chi2_contingency(contingency)[:2]
        
        total_sample_size = ab_results['control']['users'] + ab_results['treatment']['users']
        effect_size = np.sqrt(chi2 / total_sample_size)
        
        st.subheader("Statistical Significance")
        
        significance_cols = st.columns(3)
        with significance_cols[0]:
            st.metric("Chi-Square Statistic", f"{chi2:.2f}")
        with significance_cols[1]:
            st.metric("P-value", f"{p_val:.4f}")
        with significance_cols[2]:
            st.metric("Significant at α=0.05", 
                     "Yes" if p_val < 0.05 else "No")
        
        insights_cols = st.columns(2)
        with insights_cols[0]:
            st.metric("Effect Size (Cramér's V)", f"{effect_size:.3f}")
            
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
        
        fig = go.Figure()
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
    
    else:  # Marketing Mix Modeling
        st.subheader("Marketing Mix Modeling Simulation")
        
        st.info("""
        **Key Assumptions in Marketing Mix Modeling:**
        
        1. **Model Structure**
           - Linear relationships after transformation
           - Adequate handling of diminishing returns
           - Proper lag structure incorporation
        
        2. **Channel Effects**
           - No perfect multicollinearity
           - Stable channel effectiveness
           - Captured all significant channels
        
        3. **External Factors**
           - Correctly specified seasonality
           - Accounted for market conditions
           - Captured competitive effects
        
        4. **Data Quality**
           - Sufficient historical data
           - Consistent measurement over time
           - No major data gaps
        
        5. **Business Environment**
           - Stable market conditions
           - No major business model changes
           - Consistent tracking methodology
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            n_periods = st.slider("Number of Weeks", min_value=26, max_value=104, value=52)
        
        data = simulate_mmm_data(n_periods)
        mmm_results = fit_mmm_model(data)
        
        st.metric("Model R-squared", f"{mmm_results['r2']:.3f}")
        
        fig_contrib, fig_fit = plot_mmm_results(data, mmm_results)
        
        st.plotly_chart(fig_contrib, use_container_width=True)
        st.plotly_chart(fig_fit, use_container_width=True)
        
        st.subheader("Channel Contribution Analysis")
        contribution_df = mmm_results['contributions'].sort_values('Std_Coefficient', ascending=False)
        st.dataframe(contribution_df.style.format({
            'Coefficient': '{:.2f}',
            'Std_Coefficient': '{:.3f}'
        }))
        
        st.subheader("Model Insights")
        
        marketing_channels = ['SEA_P1_Spend', 'SEA_P2_Spend', 'Programmatic_Spend', 'Partner_Spend']
        channel_names = ['SEA P1', 'SEA P2', 'Programmatic', 'Partner']
        
        channel_metrics = pd.DataFrame({
            'Channel': channel_names,
            'Average_Spend': [data[ch].mean() for ch in marketing_channels],
            'Contribution': contribution_df[contribution_df['Channel'].isin(marketing_channels)]['Std_Coefficient'].values
        })
        
        channel_metrics['ROI_Index'] = channel_metrics['Contribution'] / channel_metrics['Average_Spend']
        channel_metrics = channel_metrics.sort_values('ROI_Index', ascending=False)
        
        st.write("Channel ROI Analysis:")
        st.dataframe(channel_metrics.style.format({
            'Average_Spend': '{:,.2f}',
            'Contribution': '{:.3f}',
            'ROI_Index': '{:.3f}'
        }))
        
        st.markdown("""
        ### Recommendations based on Model Results:
        1. Focus budget allocation on channels with higher ROI Index
        2. Consider external factors when evaluating performance
        3. Monitor for changes in channel effectiveness over time
        4. Validate results with incrementality testing
        """)

if __name__ == "__main__":
    main()