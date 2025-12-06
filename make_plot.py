import data
import numpy as np
import pandas as pd
import plotly.express as px
import sys

model_data = pd.read_csv('data/experiment3_simulation_data_0200.csv') #models that were trained to 0.200 avg error

model_data = model_data[model_data.block_type!='random']
model_data_interleaved = model_data[model_data.block_type=='interleaved']
model_data_blocked = model_data[model_data.block_type=='blocked']

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

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


model_rt_interleaved = px.bar(plot_data_interleaved_model,x='size_condition',color='cat_condition',
       y='rt_normalized',barmode='group',error_y='rt_error')
human_rt_interleaved = px.bar(plot_data_interleaved,x='size_condition',color='cat_condition',
       y='rt_normalized',barmode='group',error_y='rt_error')
model_rt_blocked = px.bar(plot_data_blocked_model,x='size_condition',color='cat_condition',
       y='rt_normalized',barmode='group',error_y='rt_error')
human_rt_blocked = px.bar(plot_data_blocked,x='size_condition',color='cat_condition',
       y='rt_normalized',barmode='group',error_y='rt_error')
human_acc_interleaved = px.bar(plot_data_interleaved_acc,x='size_condition',color='cat_condition',
       y='error_normalized',barmode='group',error_y='error_error')
human_acc_blocked = px.bar(plot_data_blocked_acc,x='size_condition',color='cat_condition',
       y='error_normalized',barmode='group',error_y='error_error')
model_acc_interleaved = px.bar(plot_data_interleaved_acc_model,x='size_condition',color='cat_condition',
       y='error_normalized',barmode='group',error_y='error_error')
model_acc_blocked = px.bar(plot_data_blocked_acc_model,x='size_condition',color='cat_condition',
       y='error_normalized',barmode='group',error_y='error_error')

fig = make_subplots(rows=2,cols=2,row_titles=['Categorically Blocked Condition','Interleaved Condition'],
                    column_titles=['Model Performance','Human Performance'])
fig.add_trace(model_rt_interleaved.data[0],row=2,col=1)
fig.add_trace(model_rt_interleaved.data[1],row=2,col=1)
fig.add_trace(human_rt_interleaved.data[0],row=2,col=2)
fig.add_trace(human_rt_interleaved.data[1],row=2,col=2)
fig.add_trace(model_rt_blocked.data[0],row=1,col=1)
fig.add_trace(model_rt_blocked.data[1],row=1,col=1)
fig.add_trace(human_rt_blocked.data[0],row=1,col=2)
fig.add_trace(human_rt_blocked.data[1],row=1,col=2)
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
#fig.add_annotation(dict(x=0,y=.25,xref='x1',yref='y1',text='test',showarrow=False))
fig['layout']['annotations'][2].update(text="Categorically Blocked Condition",x=-.07,textangle=-90)
fig['layout']['annotations'][3].update(text="Interleaved Condition",x=-.07,textangle=-90)
fig.show()

fig = make_subplots(rows=2,cols=2,row_titles=['Categorically Blocked Condition','Interleaved Condition'],
                    column_titles=['Model Performance','Human Performance'])
fig.add_trace(human_acc_interleaved.data[0],row=2,col=2)
fig.add_trace(human_acc_interleaved.data[1],row=2,col=2)
fig.add_trace(model_acc_interleaved.data[0],row=2,col=1)
fig.add_trace(model_acc_interleaved.data[1],row=2,col=1)
fig.add_trace(human_acc_blocked.data[0],row=1,col=2)
fig.add_trace(human_acc_blocked.data[1],row=1,col=2)
fig.add_trace(model_acc_blocked.data[0],row=1,col=1)
fig.add_trace(model_acc_blocked.data[1],row=1,col=1)
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
#fig.add_annotation(dict(x=0,y=.25,xref='x1',yref='y1',text='test',showarrow=False))
fig['layout']['annotations'][2].update(text="Categorically Blocked Condition",x=-.07,textangle=-90)
fig['layout']['annotations'][3].update(text="Interleaved Condition",x=-.07,textangle=-90)
fig.show()