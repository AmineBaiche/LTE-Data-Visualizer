
###Amine Baiche PROJET CELL
from dash.dependencies import Input, Output
import dash
import dash_core_components as dcc
import dash_html_components as html
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

import plotly.express as px
import numpy as np
from pandas import DataFrame as df

import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


#le séparateur ici est ; car comme mentioné dans le rapport on a du modifier les valeurs nule qui coresspondent à
# ce symbole - dans le data original par NaN en utilisant excel
static=pd.read_csv("./LTE_Dataset/static.csv",sep=';')

pedestrian=pd.read_csv("./LTE_Dataset/pedestrian.csv",sep=';')

car=pd.read_csv("./LTE_Dataset/car.csv",sep=';')

train=pd.read_csv("./LTE_Dataset/train.csv",sep=';')

bus=pd.read_csv("./LTE_Dataset/bus.csv",sep=';')
#voir le nombre de missing data dans chaque colonne

#print(bus.isnull().sum())

#Imputation avec algo Iterative pour remplacer les valeurs NaN
bus=bus.drop(["NRxRSRP","NRxRSRQ",'State','Timestamp','NetworkMode','Operatorname'],axis=1)
imp=IterativeImputer()
bus[:]=imp.fit_transform(bus)

pedestrian=pedestrian.drop(["NRxRSRP","NRxRSRQ",'State','Timestamp','NetworkMode','Operatorname'],axis=1)
imp=IterativeImputer()
pedestrian[:]=imp.fit_transform(pedestrian)

car=car.drop(["NRxRSRP","NRxRSRQ",'State','Timestamp','NetworkMode','Operatorname'],axis=1)
imp=IterativeImputer()
car[:]=imp.fit_transform(car)

train=train.drop(["NRxRSRP","NRxRSRQ",'State','Timestamp','NetworkMode','Operatorname'],axis=1)
imp=IterativeImputer()
train[:]=imp.fit_transform(train)

static=static.drop(["NRxRSRP","NRxRSRQ",'State','Timestamp','NetworkMode','Operatorname'],axis=1)
imp=IterativeImputer()
static[:]=imp.fit_transform(static)
#


#drop duplicate values

car=car.drop_duplicates(["Longitude"])
train=train.drop_duplicates(["Longitude"])
bus=bus.drop_duplicates(["Longitude"])
pedestrian =pedestrian.drop_duplicates(["Longitude"])


#Cell Size n'a rien a voir avec la taille réél de la cellule j'aurai du nommer cette variable autrement
#10 = size dans scatter mapbox ,
#33629 : coresspond au nombre de ligne dans chaque data on
#75722
#10779
cell_size=[10]*10779
bus["cellsize"]=df(cell_size)

cell_size=[10]*38934
train["cellsize"]=df(cell_size)

cell_size2=[10]*75723
car["cellsize"]=df(cell_size2)

cell_size3=[10]*33629
pedestrian["cellsize"]=df(cell_size3)

#print(bus)
#print(pedestrian)
###################Generation des maps on a choisi que les valeur de CQI au dessous de 6 seront affiché en rouge
# valeur entre 6 et 8 en jaune et si plus en vert et donc le cannal est bien. Hover template renvoie les donneés du point 
map_train = go.Scattermapbox(name="train",
                             customdata=train.loc[:,
                                      ['RSRQ', 'SNR', 'DL_bitrate', 'UL_bitrate', 'ServingCell_Distance']],
                                     lon=train['Longitude'],
                                     lat=train['Latitude'],
                                     text=train['CellID'],
                                     mode="markers",
                             hovertemplate="<b>CELL ID %{text}</b><br><br>"+
                                  "RSRQ:%{customdata[0]}<br>"+
                                  "SNR:%{customdata[1]}<br>"+
                                  "Dl_bitrate:%{customdata[2]}<br>"+
                                  "Ul_bitrate:%{customdata[3]}<br>"+
                                                   "ServingCell_distance:%{customdata[4]}<br>",
                                     marker=go.scattermapbox.Marker(
                                         size=train['cellsize'],
                                         color=pd.Series(np.where(train["CQI"] <6, 'alert',
                                                                  np.where(train["CQI"] > 7,
                                                                           'warning', 'normal'))
                                                         ).astype(str).map(
                                             {'alert': 1, 'warning': 0.5, 'normal': 0}),
                                         colorscale=[[0, 'green'], [0.5, 'yellow'], [1, 'red']],
                                         opacity=1
                                     )
                                     )
map_bus = go.Scattermapbox(name="Bus",
                             customdata=bus.loc[:,
                                      ['RSRQ', 'SNR', 'DL_bitrate', 'UL_bitrate', 'ServingCell_Distance']],
                                     lon=bus['Longitude'],
                                     lat=bus['Latitude'],
                                     text=bus['CellID'],
                                     mode="markers",
                             hovertemplate="<b>CELL ID %{text}</b><br><br>"+
                                  "RSRQ:%{customdata[0]}<br>"+
                                  "SNR:%{customdata[1]}<br>"+
                                  "Dl_bitrate:%{customdata[2]}<br>"+
                                  "Ul_bitrate:%{customdata[3]}<br>"+
                                                   "ServingCell_distance:%{customdata[4]}<br>",
                                     marker=go.scattermapbox.Marker(
                                         size=bus['cellsize'],
                                         color=pd.Series(np.where(bus["CQI"] <6, 'alert',
                                                                  np.where(bus["CQI"] > 7,
                                                                           'warning', 'normal'))
                                                         ).astype(str).map(
                                             {'alert': 1, 'warning': 0.5, 'normal': 0}),
                                         colorscale=[[0, 'green'], [0.5, 'yellow'], [1, 'red']],
                                         opacity=1
                                     )
                                     )

map_car = go.Scattermapbox(name="car",
                           customdata=car.loc[:,
                                      ['RSRQ', 'SNR', 'DL_bitrate', 'UL_bitrate', 'ServingCell_Distance']],

                           lon=car['Longitude'],
                                     lat=car['Latitude'],
                                     text=car['CellID'],
                           hovertemplate="<b>CELL ID %{text}</b><br><br>"+
                                  "RSRQ:%{customdata[0]}<br>"+
                                  "SNR:%{customdata[1]}<br>"+
                                  "Dl_bitrate:%{customdata[2]}<br>"+
                                  "Ul_bitrate:%{customdata[3]}<br>"+
                                                   "ServingCell_distance:%{customdata[4]}<br>",
                                     mode="markers",
                                     marker=go.scattermapbox.Marker(
                                         size=car['cellsize'],
                                         color=pd.Series(np.where(car["CQI"] <6, 'alert',
                                                                  np.where(car["CQI"] > 7,
                                                                           'warning', 'normal'))
                                                         ).astype(str).map(
                                             {'alert': 1, 'warning': 0.5, 'normal': 0}),
                                         colorscale=[[0, 'green'], [0.5, 'yellow'], [1, 'red']],
                                         opacity=1
                                     )
                                     )
map_pedestrian = go.Scattermapbox(name="pedestrian",
                                  customdata=pedestrian.loc[:,['RSRQ','SNR','DL_bitrate','UL_bitrate','ServingCell_Distance']],
                                     lon=pedestrian['Longitude'],
                                     lat=pedestrian['Latitude'],
                                     text=pedestrian['CellID'],
                                     hovertemplate="<b>CELL ID %{text}</b><br><br>"+
                                  "RSRQ:%{customdata[0]}<br>"+
                                  "SNR:%{customdata[1]}<br>"+
                                  "Dl_bitrate:%{customdata[2]}<br>"+
                                  "Ul_bitrate:%{customdata[3]}<br>"+
                                                   "ServingCell_distance:%{customdata[4]}<br>",
                                     mode="markers",
                                  showlegend=True,
                                     marker=go.scattermapbox.Marker(
                                         size=pedestrian['cellsize'],
                                         color=pd.Series(np.where(pedestrian["CQI"] <6, 'alert',
                                                                  np.where(pedestrian["CQI"] > 7,
                                                                           'warning', 'normal'))
                                                         ).astype(str).map(
                                             {'alert': 1, 'warning': 0.5, 'normal': 0}),
                                         colorscale=[[0, 'green'], [0.5, 'yellow'], [1, 'red']],
                                         opacity=1
                                     )
                                     )
#Le type de maps sur laquel on veut afficher
layout=go.Layout(mapbox_style="open-street-map",  paper_bgcolor='#E4EDE0',margin = dict(l = 0, r = 0, t = 0, b = 0),
    plot_bgcolor='white')

#Pour afficher les maps
data=[map_train,map_car,map_pedestrian,map_bus]
fig=go.Figure(data=data,layout=layout)

#pour styliser la legende
fig.update_layout(autosize=True,
    plot_bgcolor="#E4EDE0",
                  legend=dict(
                      yanchor="top",
                      y=0.99,
                      xanchor="left",
                      x=0.01,
                      orientation="h"
                  ))
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


#Structurer les division html et invoquer les deux fonction graph
app.layout = html.Div(
                      children=[
html.Img(
    src="https://upload.wikimedia.org/wikipedia/fr/c/cd/Logo_Sorbonne_Universit%C3%A9.png"
,height=100,style={'display':'center'}),
                          html.H1("Projet 11 CELL. Data Analytics with the LTE Dataset"),

                          html.Div(children=[
dcc.Dropdown(style={'width': '49%','display': 'inline-block'},
                id='Cell',
                options=[
                    {
                        'label':'Cell_ID'+str(i), 'value': i} for i in car['CellID'].unique()
                ],
             #la cellule qui va s'afficher par défaut
                value=1,
                multi=False
            ),
                dcc.Dropdown(style={'width': '49%'},
                    id='parameter',
                    options=[{'label': 'SNR', 'value': "SNR"},
                             {'label': 'CQI', 'value': "CQI"},
                             {'label': 'DL_bitrate', 'value': "DL_bitrate"},
                             {'label': 'RSRQ', 'value': "RSRQ"}],
                             # KPI par defaut
                             value='SNR'
                ),
#
        dcc.Graph(id='indicator-graphic', figure={}, style={'width': '49%','height':'49%','display': 'inline-block'}),
        dcc.Graph(figure=fig, style={'width': '49%','height':'49%','display': 'inline-block'})

    ])
])


# Fonction de callback qui permet de changer les parameteres dynamiquement selon ce que l'utilisateur
#aura choisit
@app.callback(

        Output(component_id='indicator-graphic', component_property='figure'),
[
    Input(component_id='Cell', component_property='value'),
    Input(component_id='parameter', component_property='value')]
)
#aprés une fonction de callback il faut toujours appeller cette fonction qui prend en compte
#les parametres que l'utilisateur aura choisi
def update_graph_scatter(idcell,parameter):
    dff=car
    dff = dff[dff['CellID'] == idcell]
    dff = dff.drop_duplicates(['ServingCell_Distance'])
    dff= dff.sort_values(by=['ServingCell_Distance'])
#Pour afficher la distance en kilométre
    fig = px.line(x=dff['ServingCell_Distance']/1000,
                     y=dff[parameter],labels = {'x':'ServiceCell_Distance Km','y':parameter})
    # Pour la couleur du graph
    fig.update_layout(
        autosize=True,
        plot_bgcolor="#E4EDE0"
    )

    return fig


#lancer le serveur
if __name__ == '__main__':
    app.run_server(debug=True)
