import streamlit as st
import numpy as np # tinh toan
import numpy.linalg as npl # tinh toan
from sympy import *
from sympy.abc import x
import random
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(page_title = "3.8 Orthonormal Basis")

st.header("Orthonormal Basis")  

st.markdown("We can use Gaussian elimination to find a basis for a vector space. If we are given a set of non-orthogonal, unnormalized basis vectors, we can concatenate them to construct an audmented matrix, and apply Gaussian elimination to it, to find an orthonormal basis. This process is called the Gram-Schmidt process.")

st.markdown("#### Example 3.8 (Orthonormal Basis)")

cosAngle = lambda x, y: (np.dot(x.T,y)/np.sqrt(np.dot(
                                np.dot(x.T,x),
                                np.dot(y.T,y))
                            ))[0][0]
angle = lambda cos: np.arccos(cos)*(180/np.pi)

b1 = np.vstack([1/np.sqrt(2),1/np.sqrt(2)])
b2 = np.vstack([1/np.sqrt(2),-1/np.sqrt(2)])

code = ''' 
    cosAngle = lambda x, y: (np.dot(x.T,y)/np.sqrt(np.dot(
                                np.dot(x.T,x),
                                np.dot(y.T,y))
                            ))[0][0]
angle = lambda cos: np.arccos(cos)*(180/np.pi)

b1 = np.vstack([1/np.sqrt(2),1/np.sqrt(2)])
b2 = np.vstack([1/np.sqrt(2),-1/np.sqrt(2)])

vectorB1 = go.Scatter(
    x=[0,1/np.sqrt(2)], 
    y=[0,1/np.sqrt(2)], 
    marker= dict(
        size=10,
        symbol= "arrow-bar-up", 
        angleref="previous"), 
    name="b1",
    mode="lines+markers")

vectorB2 = go.Scatter(
    x=[0,1/np.sqrt(2)], 
    y=[0,-1/np.sqrt(2)],
    marker= dict(
        size=10,
        symbol= "arrow-bar-up", 
        angleref="previous", 
        color=(255,0,0)), 
    marker_color="rgb(255,0,0)",
    name="b2")

layout = go.Layout(
    autosize=False,
    width=640,
    height=640
)
    
fig= go.Figure(data=[vectorB1, vectorB2], layout=layout)
fig.update_layout(yaxis_range=[-1,1.5])
fig.update_layout(xaxis_range=[-1.5,1.5])

fig.add_annotation(x=0.3,y=0,
        xref="x",
        yref="y",
        text=str(round(angle(cosAngle(b1,b2)),2))+"\u00B0",
        showarrow=False,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#000000"
            ))

fig.show()
'''

st.code(code,line_numbers=True)

vectorB1 = go.Scatter(
    x=[0,1/np.sqrt(2)], 
    y=[0,1/np.sqrt(2)], 
    marker= dict(size=10,symbol= "arrow-bar-up", angleref="previous"), 
    name="b1",
    mode="lines+markers")

vectorB2 = go.Scatter(
    x=[0,1/np.sqrt(2)], 
    y=[0,-1/np.sqrt(2)],
    marker= dict(size=10,symbol= "arrow-bar-up", angleref="previous", color=(255,0,0)), 
    marker_color="rgb(255,0,0)",
    name="b2")

layout = go.Layout(
    autosize=False,
    width=640,
    height=640
)

fig= go.Figure(data=[vectorB1, vectorB2], layout=layout)
fig.update_layout(yaxis_range=[-1,1.5])
fig.update_layout(xaxis_range=[-1.5,1.5])

fig.add_annotation(x=0.3,y=0,
        xref="x",
        yref="y",
        text=str(round(angle(cosAngle(b1,b2)),2))+"\u00B0",
        showarrow=False,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#000000"
            ))

st.plotly_chart(fig)