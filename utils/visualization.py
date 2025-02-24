import plotly.graph_objects as go
import numpy as np

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

def normalize_percentages(allocations):
    """
    Normalizes percentages to ensure they sum to 100%.
    """
    total = sum(allocations.values())
    if total > 0:
        return {k: (v/total) * 100 for k, v in allocations.items()}
    return allocations

def create_channel_analysis_plots(df_seap1, df_seap2, df_progd):
    """
    Creates analysis plots for each channel.
    """
    plots = {}
    
    # SEA P1 vs SEA P2 comparison
    fig = go.Figure()
    
    metrics = ['Clicks', 'Applications']
    for metric in metrics:
        fig.add_trace(go.Scatter(
            x=df_seap1['Cost'],
            y=df_seap1[metric],
            mode='lines+markers',
            name=f'SEA P1 {metric}'
        ))
        fig.add_trace(go.Scatter(
            x=df_seap2['Cost'],
            y=df_seap2[metric],
            mode='lines+markers',
            name=f'SEA P2 {metric}'
        ))
    
    fig.update_layout(
        title='SEA P1 vs SEA P2 Performance',
        xaxis_title='Cost (€)',
        yaxis_title='Count',
        height=500
    )
    
    plots['sea_comparison'] = fig
    
    # Add more analysis plots as needed
    
    return plots