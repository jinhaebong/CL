import plotly.graph_objects as go

# Data
models = ['QWEN2-7B-RTUNE','QWEN2-7B-SFT', 'QWEN2-7B-COR-RAIT']
correct_rates = [0.36, 0.63, 0.64]
wrong_rates = [0.12, 0.22, 0.25]
refuse_rates = [0.17, 0.08, 0.06]
overrefuse_rates = [0.33, 0.052, 0.040]

# Create figure
fig = go.Figure()

# Add CORRECT bars
fig.add_trace(go.Bar(
    x=models,
    y=correct_rates,
    name='CORRECT',
    marker_color='lightblue',
    width=0.5
))

# Add OVERREFUSE bars
fig.add_trace(go.Bar(
    x=models,
    y=overrefuse_rates,
    name='OVERREFUSE',
    marker_color='#FFA500',  # More saturated orange
    width=0.5
))

# Add REFUSE bars
fig.add_trace(go.Bar(
    x=models,
    y=refuse_rates,
    name='REFUSE',
    marker_color='#FFFACD',  # Light yellow
    width=0.5
))

# Add WRONG bars
fig.add_trace(go.Bar(
    x=models,
    y=wrong_rates,
    name='WRONG',
    marker_color='#FFE4E1',  # Light pink
    width=0.5
))

# Update layout
fig.update_layout(
    barmode='stack',
    title=dict(
        text='Qwen2-7B',
        x=0.5,
        y=0.95,
        font=dict(size=14)
    ),
    xaxis_title='',
    yaxis_title='Ratio',
    width=800,
    height=500,
    yaxis=dict(
        range=[0, 1],
        tickformat='.1f',
        dtick=0.2,  # Set tick interval to 0.2
        gridcolor='#E5E5E5',  # Lighter gray for grid lines
    ),
    plot_bgcolor='white',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1,
        traceorder='reversed'  # Reverse legend order to match image
    ),
    margin=dict(r=150),  # Add right margin for note
)

# Update grid lines
fig.update_yaxes(
    showgrid=True, 
    gridwidth=1, 
    gridcolor='#E5E5E5',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='#E5E5E5'
)

# Remove x-axis line
fig.update_xaxes(showline=False)

fig.show()