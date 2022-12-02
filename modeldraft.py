import dash
import pathlib
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, dash_table
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from plotly import tools
import yfinance as yf
from collections import deque
import random
import plotly
import datetime

import plotly.graph_objs as go


X = deque(maxlen = 20)
X.append(1)

Y = deque(maxlen = 20)
Y.append(1)
# X = deque(maxlen=20)
# # X.append(1)
# Y = deque(maxlen=20)
# Y.append(1)


df = pd.DataFrame(columns=['time', 'cats'])


# def df_shift(df,lag=0, start=1, skip=1, rejected_columns = []):
#     # df = df.copy()
#     if not lag:
#         return df
#     cols ={}
#     for i in range(start,lag+1,skip):
#         for x in list(df.columns):
#             if x not in rejected_columns:
#                 if not x in cols:
#                     cols[x] = ['{}_{}'.format(x, i)]
#                 else:
#                     cols[x].append('{}_{}'.format(x, i))
#     for k,v in cols.items():
#         columns = v
#         dfn = pd.DataFrame(data=None, columns=columns, index=df.index)    
#         i = (skip - 1)
#         for c in columns:
#             dfn[c] = df[k].shift(periods=i)
#             i+=skip
#         df = pd.concat([df, dfn], axis=1, join_axes=[df.index])
#     return df

# app = dash.Dash(__name__)
app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
app.title = "Live Model Training"

server = app.server
demo_mode = True


app.layout = dbc.Container([html.Div([
    dcc.Graph(
        id='graphid',
        # figure={
        #     'data': [
        #         go.Scatter(x=df['time'], y=df['cats'], mode = 'lines+markers')
        #     ],
        #     'layout': {
        #         'title': 'Stock Price for X over time'
        #     }
        # }
    ),
    dcc.Interval(
        id='1-second-interval',
        interval=1000, # 2000 milliseconds = 2 seconds
        n_intervals=0
    ),
    
]), html.Div([
    html.Div(children=[
        dcc.Slider(min=0.5, max=5, step=0.5, value=1, id='interval-refresh'), 
    ], style={'width': '20%'}),
    html.Div(id='latest-timestamp', style={"padding": "20px"}),
    dcc.Interval(
            id='interval-component',
            interval=1 * 1000,
            n_intervals=0
    ),
])
, html.Div(style={"height": "100%"},
    children=[html.Div([html.H2(
                    "Live Model Training Viewer",
                    id="title",
                    className="eight columns",
                    style={"margin-left": "3%"},
                ),
                html.Button(
                    id="learn-more-button",
                    className="two columns",
                    children=["Learn More"],
                ),])]),
                html.Div(html.Div(id="demo-explanation", children=['hello there'])),
        html.Div(
            className="container",
            style={"padding": "35px 25px"},), dcc.Dropdown(value='BAC', options=[
       {'label': 'Apple', 'value': 'AAPL'},
       {'label': 'SPY', 'value': 'SPY'},
       {'label': 'Bank of America', 'value': 'BAC'}, 
   ], id='demo-dropdown'), 
    html.Div(id='dd-output-container'), html.Div(id='Stock-Output'),  html.Div(id='Stock-ConOutput'), 
    ]
)


@app.callback(Output('graphid', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph_scatter(n):
    X.append(X[-1]+1)
    Y.append(Y[-1]+Y[-1] * random.uniform(-0.1,0.1))

    data = plotly.graph_objs.Scatter(
            x=list(X),
            y=list(Y),
            name='Scatter',
            mode= 'lines+markers'
    )

    figure = {'data': [data],
            'layout' : go.Layout(xaxis=dict(
                    range=[min(X),max(X)]),yaxis = 
                    dict(range = [min(Y),max(Y)]),
                    )}

    return figure
    # df.loc[n] = [n,random.randint(1, 9)]
    # figure={
    #         'data': [
    #             go.Scatter(x=df['time'], y=df['cats'], mode = 'lines+markers')
    #         ],
    #         'layout': {
    #             'title': 'Stock Price for X over time'
    #         }
    #     }
    # return figure




@app.callback(
    [Output(component_id='interval-component', component_property='interval')],
    [Input('interval-refresh', 'value')])
def update_refresh_rate(value):
    return [value * 1000] 

@app.callback(
    [Output(component_id='latest-timestamp', component_property='children')],
    [Input('interval-component', 'n_intervals')]
)
def update_timestamp(interval):
    return [html.Span(f"Last updated: {datetime.datetime.now()}")] 





# @app.callback(Output('live-graph', 'figure'),
#               Input[('graph-update', 'interval')])
# def update_graph_scatter():
#     X.append(X[-1]+1)
#     Y.append(Y[-1]+Y[-1]*random.uniform(-0.1,0.1))

#     data = plotly.graph_objs.Scatter(
#             x=list(X),
#             y=list(Y),
#             name='Scatter',
#             mode= 'lines+markers'
#             )

#     return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
#                                                 yaxis=dict(range=[min(Y),max(Y)]),)}


# @app.callback(
#     Output('Stock-ConOutput', 'children'),
#     Input('demo-dropdown', 'value')
# )

# def update_con(value):
#     df = yf.download(tickers=value, period='1d')
#     print(df)
   
#     df['Date'] = pd.to_datetime(df['Date'])

#     df = df[['Date','Close']],

#     # df['Date'] = pd.to_datetime(df['Date'])

#     # df.to_dict('records'),

#     df_crosscorrelated = df_shift(df, lag = 10, start = 1, skip = 2,rejected_columns=['Date'])
#     df_crosscorrelated['ma7'] = df_crosscorrelated['Close'].rolling(7).mean()
#     df_crosscorrelated['ma14'] = df_crosscorrelated['Close'].rolling(14).mean()
#     df_crosscorrelated['ma25'] = df_crosscorrelated['Close'].rolling(25).mean()
#     print(df_crosscorrelated.head(10))
#     return (dash_table.DataTable(df_crosscorrelated.to_dict('records'),[{"name": i, "id": i} for i in df_crosscorrelated.columns], id='tbl'),)







@app.callback(
    Output('Stock-Output', 'children'),
    Input('demo-dropdown', 'value')
)
def update_output(value):
    df = yf.download(tickers=value, period='1d')

    return (dash_table.DataTable(df.to_dict('records'),[{"name": i, "id": i} for i in df.columns], id='tbl'),)



@app.callback(
    Output('dd-output-container', 'children'),
    Input('demo-dropdown', 'value')
)
def update_output(value):
    return f'You have selected {value}'



# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
