import streamlit as st
import numpy as np # tinh toan
import numpy.linalg as npl # tinh toan
from sympy import *
from sympy.abc import x
import random
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(page_title = "3.4 Angles and Orthogonality") 

st.header("Angles and Orthogonality") 

x = np.linspace(0,np.pi,100)
y = np.cos(x)

st.text("Figure 3.4")

st.text('Create set of data point with y = cos(x):')
code = '''
    x = np.linspace(0,np.pi,100)
    y = np.cos(x)
'''

st.code(code,line_numbers=True)

points = go.Scatter(
    x=x, y=y,
    name='cos',
    mode='markers'
)

layout = go.Layout(
    autosize=False,
    width=640,
    height=640
)

st.text("Create points object, setting layout and show figure:")

code = '''
    points = go.Scatter(
    x=x, y=y,
    name='cos',
    mode='markers'
    )

    layout = go.Layout(
        autosize=False,
        width=640,
        height=640
    )

    fig = go.Figure(data=[points], layout=layout)
    fig.show()
'''
st.code(code,line_numbers=True)
fig = go.Figure(data=[points], layout=layout)
st.plotly_chart(fig)

st.markdown("#### Example 3.6 (Angle between Vectors)")

st.latex(r'''
    \cos\omega = \frac{\langle x,y\rangle}{\sqrt{\langle x,x\rangle \langle y,y\rangle}}=\frac{x^{\top}y}{\sqrt{x^{\top}xy^{\top}y}}
''')

st.text("Create two vectors x and y:")

code = '''
    x = np.vstack([1,1])
    y = np.vstack([1,2])
'''

st.code(code,line_numbers=True)

st.text("Compute angle:")

x = np.vstack([1,1])
y = np.vstack([1,2])
cosAngle = lambda x, y: (np.dot(x.T,y)/np.sqrt(np.dot(
                                np.dot(x.T,x),
                                np.dot(y.T,y))
                            ))[0][0]
angle = lambda cos: np.arccos(cos)*(180/np.pi)

code = '''
    cosAngle = lambda x, y: (np.dot(x.T,y)/np.sqrt(np.dot(
                                np.dot(x.T,x),
                                np.dot(y.T,y))
                            ))[0][0]
    angle = lambda cos: np.arccos(cos)*(180/np.pi)

    resultXY = angle(cosAngle(x,y)) # 18.434948822922017
'''

st.code(code,line_numbers=True)

st.text("Figure 3.5:")

code = '''
    vector_A = go.Scatter(
        x=[0,1], 
        y=[0,1], 
        marker= dict(size=10,symbol= "arrow-bar-up", 
        angleref="previous"), 
        name="A", 
        mode="lines+markers")

    vector_B = go.Scatter(
        x=[0,1], 
        y=[0,2],
        marker= dict(size=10,symbol= "arrow-bar-up", 
        angleref="previous", 
        color=(255,0,0)), 
        marker_color="rgb(255,0,0)", name="B")

    layout = go.Layout(
        autosize=False,
        width=640,
        height=640
    )

    fig = go.Figure(data=[vector_A, vector_B], layout=layout)
    
    fig.update_layout(yaxis_range=[-1,2.5])
    fig.update_layout(xaxis_range=[-1,2.5])

    fig.add_annotation(
        x=0.75,
        y=1,
        xref="x",
        yref="y",
        text=str(round(angle(cosAngle(x,y)),2))+"\u00B0",
        showarrow=False,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#000000"))
    
    fig.show()
    
'''

st.code(code,line_numbers=True)

vector_A = go.Scatter(
    x=[0,1], 
    y=[0,1], 
    marker= dict(size=10,symbol= "arrow-bar-up", 
    angleref="previous"), 
    name="A", 
    mode="lines+markers")

vector_B = go.Scatter(
    x=[0,1], 
    y=[0,2],
    marker= dict(size=10,symbol= "arrow-bar-up", 
    angleref="previous", 
    color=(255,0,0)), 
    marker_color="rgb(255,0,0)", name="B")

layout = go.Layout(
    autosize=False,
    width=640,
    height=640
)

fig = go.Figure(data=[vector_A, vector_B], layout=layout)

fig.update_layout(yaxis_range=[-1,2.5])
fig.update_layout(xaxis_range=[-1,2.5])

fig.add_annotation(
    x=0.75,
    y=1,
    xref="x",
    yref="y",
    text=str(round(angle(cosAngle(x,y)),2))+"\u00B0",
    showarrow=False,
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#000000"))
    
st.plotly_chart(fig)

st.markdown("#### Example 3.7 (Orthogonal Vectors)")

st.text("Create two vectors x and y:")

code = '''
    x = np.vstack([1,1])
    y = np.vstack([-1,1])
'''
x = np.vstack([1,1])
y = np.vstack([-1,1])
st.code(code,line_numbers=True)

st.text("Compute angle:")

code = '''
    resultXY = angle(cosAngle(x,y)) # 90.0
'''

st.code(code,line_numbers=True)

st.text("Figure 3.6:")

code = '''
    vector_A = go.Scatter(
        x=[0,1], 
        y=[0,1], 
        marker= dict(size=10,symbol= "arrow-bar-up", 
        angleref="previous"), 
        name="A", 
        mode="lines+markers")

    vector_B = go.Scatter(
        x=[0,-1], 
        y=[0,1],
        marker= dict(size=10,symbol= "arrow-bar-up", 
        angleref="previous", 
        color=(255,0,0)), 
        marker_color="rgb(255,0,0)", name="B")

    layout = go.Layout(
        autosize=False,
        width=640,
        height=640
    )

    fig = go.Figure(data=[vector_A, vector_B], layout=layout)
    
    fig.update_layout(yaxis_range=[-2.5,2.5])
    fig.update_layout(xaxis_range=[-2.5,2.5])

    fig.add_annotation(
        x=0,
        y=0.3,
        xref="x",
        yref="y",
        text=str(round(angle(cosAngle(x,y)),2))+"\u00B0",
        showarrow=False,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#000000"))
    
    fig.show()
    
'''

st.code(code,line_numbers=True)

vector_A = go.Scatter(
    x=[0,1], 
    y=[0,1], 
    marker= dict(size=10,symbol= "arrow-bar-up", 
    angleref="previous"), 
    name="A", 
    mode="lines+markers")

vector_B = go.Scatter(
    x=[0,-1], 
    y=[0,1],
    marker= dict(size=10,symbol= "arrow-bar-up", 
    angleref="previous", 
    color=(255,0,0)), 
    marker_color="rgb(255,0,0)", name="B")

layout = go.Layout(
    autosize=False,
    width=640,
    height=640
)

fig = go.Figure(data=[vector_A, vector_B], layout=layout)

fig.update_layout(yaxis_range=[-2.5,2.5])
fig.update_layout(xaxis_range=[-2.5,2.5])

fig.add_annotation(
    x=0,
    y=0.3,
    xref="x",
    yref="y",
    text=str(round(angle(cosAngle(x,y)),2))+"\u00B0",
    showarrow=False,
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#000000"))

st.plotly_chart(fig)

st.text("By changing how inner is induced, such as with the matrix:")
st.latex(r'''
            \begin{bmatrix}
                2 & 0\\
                0 & 1
                \end{bmatrix}
         ''')
st.text("we find the vectors are no longer orthogonal, despite their being orthogonal \nwith respect to another inner product.")

code = '''
    x = np.vstack([1,1])
    y = np.vstack([-1,1])
    inner = np.vstack([[2,0],[0,1]])
    omega = (x.T@inner@y)/npl.norm(x.T@inner@x)
    newAngle = angle(omega) # 109.47122063
'''

st.code(code, line_numbers=True)
