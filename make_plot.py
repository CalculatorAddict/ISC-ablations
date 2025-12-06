import data
import numpy as np
import pandas as pd
import sys

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

def _rotate_row_titles(fig, row_labels):
    """Rotate row titles vertically on the left edge regardless of subplot count."""
    for ann in fig.layout.annotations:
        if ann.text in row_labels:
            ann.update(x=-0.07, textangle=-90, xanchor='center')

def make_model_plot(models, include_human=True):
    """Generates plot for model on the experiment."""

    num_models = len(models) # counts number of models used

    model_plot_data = []

    for name, path in models:
        model_plot_data += [(name, *generate_model_plot_data(path))]

    if include_human:
        num_models += 1
        model_plot_data += [('Human', *generate_human_plot_data())]

    # figure for response time
    fig = make_subplots(rows=2,cols=num_models,row_titles=['Categorically Blocked Condition','Interleaved Condition'],
                        column_titles=[name + ' Performance' for name,_,_,_,_ in model_plot_data])

    for i in range(len(model_plot_data)):
        fig.add_trace(model_plot_data[i][1].data[0],row=2,col=i+1)
        fig.add_trace(model_plot_data[i][1].data[1],row=2,col=i+1)
        fig.add_trace(model_plot_data[i][2].data[0],row=1,col=i+1)
        fig.add_trace(model_plot_data[i][2].data[1],row=1,col=i+1)
    
    for idx,trace in enumerate(fig.data):
        if idx > 1:
            trace.showlegend = False
    fig.update_layout(title='Log Reaction Time (log ms, Relative to Mean) by Condition',yaxis_side='left',width=1200,height=800,
                    titlefont=dict(size=20))
    fig.update_layout(plot_bgcolor='white',title_x=0.5,
                        legend=dict(yanchor='top',y=.99,xanchor='left',x=.01,font=dict(size=16)))
    fig.update_xaxes(showline=True,linewidth=1.5,linecolor='black',tickfont=dict(size=16),
                mirror=True,ticks='outside',showgrid=False,titlefont=dict(size=20)
                )
    fig.update_yaxes(showline=True,linewidth=1.5,linecolor='black',
                mirror=True,ticks='outside',showgrid=False,titlefont=dict(size=20),
                zeroline=True,zerolinecolor='black',zerolinewidth=1)
    _rotate_row_titles(fig, {"Categorically Blocked Condition", "Interleaved Condition"})
    fig.show()

    fig = make_subplots(rows=2,cols=num_models,row_titles=['Categorically Blocked Condition','Interleaved Condition'],
                        column_titles=[name + ' Performance' for name,_,_,_,_ in model_plot_data])
    for i in range(len(model_plot_data)):
        fig.add_trace(model_plot_data[i][3].data[0],row=2,col=i+1)
        fig.add_trace(model_plot_data[i][3].data[1],row=2,col=i+1)
        fig.add_trace(model_plot_data[i][4].data[0],row=1,col=i+1)
        fig.add_trace(model_plot_data[i][4].data[1],row=1,col=i+1)
    for idx,trace in enumerate(fig.data):
        if idx > 1:
            trace.showlegend = False
    fig.update_layout(title='Error Rate (%, Relative to Mean) by Condition',yaxis_side='left',width=1200,height=800,
                    titlefont=dict(size=20))
    fig.update_layout(plot_bgcolor='white',title_x=0.5,
                        legend=dict(yanchor='top',y=.99,xanchor='left',x=.01,font=dict(size=16)))
    fig.update_xaxes(showline=True,linewidth=1.5,linecolor='black',tickfont=dict(size=16),
                mirror=True,ticks='outside',showgrid=False,titlefont=dict(size=20)
                )
    fig.update_yaxes(showline=True,linewidth=1.5,linecolor='black',
                mirror=True,ticks='outside',showgrid=False,titlefont=dict(size=20),
                zeroline=True,zerolinecolor='black',zerolinewidth=1)
    _rotate_row_titles(fig, {"Categorically Blocked Condition", "Interleaved Condition"})
    fig.show()


def generate_model_plot_data(model_path):
    """Generate experiment plot data for a given model."""
    model_data = pd.read_csv(model_path)

    model_data = model_data[model_data.block_type!='random']
    model_data_interleaved = model_data[model_data.block_type=='interleaved']
    model_data_blocked = model_data[model_data.block_type=='blocked']

    model_data_interleaved = model_data_interleaved.groupby(['model','cat_condition','size_condition'],as_index=False).mean(numeric_only=True)
    model_data_blocked = model_data_blocked.groupby(['model','cat_condition','size_condition'],as_index=False).mean(numeric_only=True)
    plot_data_interleaved_model = data.add_within_subject_error_bars(model_data_interleaved,subject='model',dv='rt',remove_mean=True)
    plot_data_interleaved_model = plot_data_interleaved_model.groupby(['cat_condition','size_condition'],as_index=False).mean(numeric_only=True)
    plot_data_interleaved_model['cat_condition'] = plot_data_interleaved_model.cat_condition.replace({'c_ma':'Category Match','c_ms':'Category Mismatch'})
    plot_data_interleaved_model['size_condition'] = plot_data_interleaved_model.size_condition.replace({'s_ma':'Size Match','s_ms':'Size Mismatch'})
    plot_data_blocked_model = data.add_within_subject_error_bars(model_data_blocked,subject='model',dv='rt',remove_mean=True)
    plot_data_blocked_model = plot_data_blocked_model.groupby(['cat_condition','size_condition'],as_index=False).mean(numeric_only=True)
    plot_data_blocked_model['cat_condition'] = plot_data_blocked_model.cat_condition.replace({'c_ma':'Category Match','c_ms':'Category Mismatch'})
    plot_data_blocked_model['size_condition'] = plot_data_blocked_model.size_condition.replace({'s_ma':'Size Match','s_ms':'Size Mismatch'})
    plot_data_interleaved_acc_model = data.add_within_subject_error_bars(model_data_interleaved,subject='model',dv='error',remove_mean=True)
    plot_data_interleaved_acc_model = plot_data_interleaved_acc_model.groupby(['cat_condition','size_condition'],as_index=False).mean(numeric_only=True)
    plot_data_interleaved_acc_model['cat_condition'] = plot_data_interleaved_acc_model.cat_condition.replace({'c_ma':'Category Match','c_ms':'Category Mismatch'})
    plot_data_interleaved_acc_model['size_condition'] = plot_data_interleaved_acc_model.size_condition.replace({'s_ma':'Size Match','s_ms':'Size Mismatch'})
    plot_data_blocked_acc_model = data.add_within_subject_error_bars(model_data_blocked,subject='model',dv='error',remove_mean=True)
    plot_data_blocked_acc_model = plot_data_blocked_acc_model.groupby(['cat_condition','size_condition'],as_index=False).mean(numeric_only=True)
    plot_data_blocked_acc_model['cat_condition'] = plot_data_blocked_acc_model.cat_condition.replace({'c_ma':'Category Match','c_ms':'Category Mismatch'})
    plot_data_blocked_acc_model['size_condition'] = plot_data_blocked_acc_model.size_condition.replace({'s_ma':'Size Match','s_ms':'Size Mismatch'})

    model_rt_interleaved = px.bar(plot_data_interleaved_model,x='size_condition',color='cat_condition',
        y='rt_normalized',barmode='group',error_y='rt_error')
    model_rt_blocked = px.bar(plot_data_blocked_model,x='size_condition',color='cat_condition',
        y='rt_normalized',barmode='group',error_y='rt_error')
    model_acc_interleaved = px.bar(plot_data_interleaved_acc_model,x='size_condition',color='cat_condition',
        y='error_normalized',barmode='group',error_y='error_error')
    model_acc_blocked = px.bar(plot_data_blocked_acc_model,x='size_condition',color='cat_condition',
        y='error_normalized',barmode='group',error_y='error_error')
    
    return model_rt_interleaved, model_rt_blocked, model_acc_interleaved, model_acc_blocked

def generate_human_plot_data():
    behavioral_data = data.load_behavioral_data()
    behavioral_data['rt'] = np.log(behavioral_data.rt.values)

    acc_data = behavioral_data[behavioral_data.participant_type!='random']
    acc_data = acc_data.groupby(['block_type_agg','size_condition','cat_condition','participant'],as_index=False).mean(numeric_only=True)
    acc_data_interleaved = acc_data[acc_data.block_type_agg=='interleaved']
    acc_data_blocked = acc_data[acc_data.block_type_agg=='blocked']

    rt_data = behavioral_data[(behavioral_data.participant_type!='random')&(behavioral_data.correct==1)]
    rt_data = rt_data.groupby(['block_type_agg','size_condition','cat_condition','participant'],as_index=False).mean(numeric_only=True)
    rt_data_interleaved = rt_data[rt_data.block_type_agg=='interleaved']
    rt_data_blocked = rt_data[rt_data.block_type_agg=='blocked']

    plot_data_interleaved = data.add_within_subject_error_bars(rt_data_interleaved)
    plot_data_interleaved = plot_data_interleaved.groupby(['cat_condition','size_condition'],as_index=False).mean(numeric_only=True)
    plot_data_interleaved['cat_condition'] = plot_data_interleaved.cat_condition.replace({'c_ma':'Category Match','c_ms':'Category Mismatch'})
    plot_data_interleaved['size_condition'] = plot_data_interleaved.size_condition.replace({'s_ma':'Size Match','s_ms':'Size Mismatch'})
    plot_data_blocked = data.add_within_subject_error_bars(rt_data_blocked)
    plot_data_blocked = plot_data_blocked.groupby(['cat_condition','size_condition'],as_index=False).mean(numeric_only=True)
    plot_data_blocked['cat_condition'] = plot_data_blocked.cat_condition.replace({'c_ma':'Category Match','c_ms':'Category Mismatch'})
    plot_data_blocked['size_condition'] = plot_data_blocked.size_condition.replace({'s_ma':'Size Match','s_ms':'Size Mismatch'})

    acc_data_interleaved['error'] = 1-acc_data_interleaved.correct
    plot_data_interleaved_acc = data.add_within_subject_error_bars(acc_data_interleaved,dv='error',remove_mean=True)
    plot_data_interleaved_acc = plot_data_interleaved_acc.groupby(['cat_condition','size_condition'],as_index=False).mean(numeric_only=True)
    plot_data_interleaved_acc['cat_condition'] = plot_data_interleaved_acc.cat_condition.replace({'c_ma':'Category Match','c_ms':'Category Mismatch'})
    plot_data_interleaved_acc['size_condition'] = plot_data_interleaved_acc.size_condition.replace({'s_ma':'Size Match','s_ms':'Size Mismatch'})

    acc_data_blocked['error'] = 1-acc_data_blocked.correct
    plot_data_blocked_acc = data.add_within_subject_error_bars(acc_data_blocked,dv='error',remove_mean=True)
    plot_data_blocked_acc = plot_data_blocked_acc.groupby(['cat_condition','size_condition'],as_index=False).mean(numeric_only=True)
    plot_data_blocked_acc['cat_condition'] = plot_data_blocked_acc.cat_condition.replace({'c_ma':'Category Match','c_ms':'Category Mismatch'})
    plot_data_blocked_acc['size_condition'] = plot_data_blocked_acc.size_condition.replace({'s_ma':'Size Match','s_ms':'Size Mismatch'})

    human_rt_interleaved = px.bar(plot_data_interleaved,x='size_condition',color='cat_condition',
        y='rt_normalized',barmode='group',error_y='rt_error')
    human_rt_blocked = px.bar(plot_data_blocked,x='size_condition',color='cat_condition',
        y='rt_normalized',barmode='group',error_y='rt_error')
    human_acc_interleaved = px.bar(plot_data_interleaved_acc,x='size_condition',color='cat_condition',
        y='error_normalized',barmode='group',error_y='error_error')
    human_acc_blocked = px.bar(plot_data_blocked_acc,x='size_condition',color='cat_condition',
        y='error_normalized',barmode='group',error_y='error_error')
    
    return human_rt_interleaved, human_rt_blocked, human_acc_interleaved, human_acc_blocked


if __name__=='__main__':
    make_model_plot(
        [(
            'Model', 'data/experiment3_simulation_data_0200.csv'
        )],
        include_human=True
    )
