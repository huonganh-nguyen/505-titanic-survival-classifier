##### Import your libraries #####
import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import pickle
from tabs import tab_1, tab_2, tab_3, tab_4
from utils import display_eval_metrics, Viridis


##### Define your variables #####
tabtitle = 'Titanic!'
Viridis=[
"#440154", "#440558", "#450a5c", "#450e60", "#451465", "#461969",
"#461d6d", "#462372", "#472775", "#472c7a", "#46307c", "#45337d",
"#433880", "#423c81", "#404184", "#3f4686", "#3d4a88", "#3c4f8a",
"#3b518b", "#39558b", "#37598c", "#365c8c", "#34608c", "#33638d",
"#31678d", "#2f6b8d", "#2d6e8e", "#2c718e", "#2b748e", "#29788e",
"#287c8e", "#277f8e", "#25848d", "#24878d", "#238b8d", "#218f8d",
"#21918d", "#22958b", "#23988a", "#239b89", "#249f87", "#25a186",
"#25a584", "#26a883", "#27ab82", "#29ae80", "#2eb17d", "#35b479",
"#3cb875", "#42bb72", "#49be6e", "#4ec16b", "#55c467", "#5cc863",
"#61c960", "#6bcc5a", "#72ce55", "#7cd04f", "#85d349", "#8dd544",
"#97d73e", "#9ed93a", "#a8db34", "#b0dd31", "#b8de30", "#c3df2e",
"#cbe02d", "#d6e22b", "#e1e329", "#eae428", "#f5e626", "#fde725"]
# source: https://bhaskarvk.github.io/colormap/reference/colormap.html
sourceurl = 'https://www.kaggle.com/c/titanic'
githublink = 'https://github.com/plotly-dash-apps/505-titanic-survival-classifier'
choices=['Comparison of Models']


##### Import dataframe #####
df=pd.read_csv('resources/compare_models.csv', index_col=0)


##### Set up the bar chart #####
mydata1 = go.Bar(
    x=df.loc['F1 score'].index,
    y=df.loc['F1 score'],
    name=df.index[0],
    marker=dict(color=Viridis[50])
)
mydata2 = go.Bar(
    x=df.loc['Accuracy'].index,
    y=df.loc['Accuracy'],
    name=df.index[1],
    marker=dict(color=Viridis[30])
)
mydata3 = go.Bar(
    x=df.loc['AUC score'].index,
    y=df.loc['AUC score'],
    name=df.index[2],
    marker=dict(color=Viridis[10])
)
mylayout = go.Layout(
    title='Random Forest model yields the best evaluation statistics',
    xaxis = dict(title = 'Predictive models'), # x-axis label
    yaxis = dict(title = 'Score'), # y-axis label
    
)
fig = go.Figure(data=[mydata1, mydata2, mydata3], layout=mylayout)
fig


##### Instantiate the app #####
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle


##### Layout of the app #####
app.layout = html.Div([
    html.H2('Surviving the Titanic - Model Evaluation Statistics'),
    html.Br(),
    dcc.Graph(id='display-value', 
        figure=fig),
    html.A('Code on Github', href=githublink),
    html.Br(),
    html.A("Data Source", href=sourceurl),
])


####### Deploy the app #######
if __name__ == '__main__':
    app.run_server(debug=True)
