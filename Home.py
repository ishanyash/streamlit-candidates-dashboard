import streamlit as st
from pathlib import Path

# Configure the page with custom theme
st.set_page_config(
    page_title="Home Page",  # This changes the browser tab title
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("StepStone Marketing Analytics Platform")
    
    st.markdown("""
    ## Introduction
    Welcome to the StepStone Marketing Analytics Platform. This tool provides comprehensive analysis 
    and optimization capabilities for marketing budget allocation across multiple channels. The platform 
    consists of three main sections, each designed to help you make data-driven decisions about your 
    marketing investments.
    
    ## Navigation Guide
    
    ### 1. Budget Allocation Dashboard
    This page helps you optimize your marketing budget allocation across four key channels:
    - **SEA P1**: Search Engine Advertising Portfolio 1
    - **SEA P2**: Search Engine Advertising Portfolio 2
    - **Programmatic**: Programmatic Display Advertising
    - **Partner**: Partner Channel Marketing
    
    **Key Features:**
    - Interactive budget allocation controls
    - Real-time visualization of budget distribution
    - Immediate calculation of expected outcomes
    - Detailed metrics including applications and candidates per listing
    
    **How to Use:**
    1. Enter percentage allocations for each channel
    2. Ensure total allocation equals 100%
    3. View the resulting metrics and visualizations
    
    ### 2. Channel Analysis Dashboard
    This section provides detailed analysis of channel performance and relationships:
    
    **Key Features:**
    - Channel performance comparisons
    - Cost-benefit analysis
    - Performance metrics over time
    - Statistical modeling and insights
    
    **How to Use:**
    1. Select analysis type from the sidebar
    2. Compare channel performances
    3. Review statistical metrics and trends
    
    ### 3. Testing Analysis Dashboard
    This page allows you to simulate and analyze different testing approaches:
    
    **Key Features:**
    - Incrementality testing simulation
    - Attribution analysis
    - A/B testing frameworks
    
    **How to Use:**
    1. Choose a testing methodology
    2. Set test parameters
    3. Review results and statistical significance
    
    ## Best Practices
    1. **Budget Allocation:**
       - Start with the Budget Allocation page to understand current distribution
       - Experiment with different allocations to optimize outcomes
    
    2. **Analysis:**
       - Use the Channel Analysis page to understand performance patterns
       - Review historical trends before making major changes
    
    3. **Testing:**
       - Utilize the Testing Analysis page to validate assumptions
       - Consider incrementality when evaluating channel effectiveness
    
    ## Data Considerations
    - All analyses use historical data from your marketing channels
    - Calculations account for various factors including:
        - Daily and monthly patterns
        - Channel-specific conversion rates
        - Cost per click and application rates
        - Channel interdependencies
    
    ## Getting Started
    1. Begin with the Budget Allocation page to understand your current distribution
    2. Use Channel Analysis to identify opportunities for optimization
    3. Validate your hypotheses using the Testing Analysis tools
    """)
    
    # Display system status
    st.subheader("System Status")
    try:
        file_path = Path("data/data_cs.xlsx")
        if file_path.exists():
            st.success("Data source connected successfully")
        else:
            st.error("Unable to locate data source. Please check data/data_cs.xlsx exists")
    except Exception as e:
        st.error(f"System error: {str(e)}")

if __name__ == "__main__":
    main()