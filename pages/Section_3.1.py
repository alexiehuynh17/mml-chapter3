import streamlit as st
import numpy as np # tinh toan
import numpy.linalg as npl # tinh toan
from sympy import *
from sympy.abc import x
import random
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from st_pages import add_page_title

st.set_page_config(page_title = "3.1 Norms")

st.header("Norm") 


st.markdown("#### Example 3.1 Manhattan and Euclidean Norm")

st.markdown("##### Manhattan Norm")
st.latex(r'''
    \|x\|_1 := \sum_{i=1}^{n}|x_i|=1 \\
    |x_2| + |x_1| = 1, x_2 = 1 - |x_1|
        ''')

st.markdown("##### Euclidean Norm")
st.latex(r'''        
    \|x\|_2 := \sqrt{\sum_{i=1}^n x_{i}^2} = \sqrt{x^{\top}x}=1 \\
    \sqrt{x_1^2 + x_2^2} = 1, x_2 = \sqrt{1-x_1^2}
     ''')

st.markdown("Where we state $= 1$, to produce a unit shape.")
st.markdown("1. For the $l_1$ norm, we are selecting $x_0$ and $x_1$ such that the sum of their absolute values is 1.")
st.markdown("2. For the $l_2$ norm, we are selecting $x_0$ and $x_1$ such that the square root of the sum of their squared values is 1.")

st.markdown("\n")
st.markdown("##### Plot $l_1$ and $l_2$ norm")

st.text("First we create set of data point:")

code = '''
    xRight = np.linspace(0,1,50)
    xLeft = np.linspace(-1,0,50)
    '''
st.code(code, language='python', line_numbers=True)

st.text("Manhattan: Building L1 (x,y) coordinates for 4 quadrants, where y in the first is \nof the form 1-x")

code = '''
    l1normData = np.hstack([
        np.vstack(
            np.vstack([xRight,xLeft,xLeft,xRight]).ravel()),
            np.vstack(np.vstack([1-xRight, xLeft+1, -xLeft-1, xRight-1]).ravel()
        )
    ])
    '''
st.code(code, language='python', line_numbers=True)

st.text("Euclidean: Building L1 (x,y) coordinates for 4 quadrants, where y in the first is \nof the form sqrt(1-x**2)")


xRight = np.linspace(0,1,50)
xLeft = np.linspace(-1,0,50) 
l1normData = np.hstack([np.vstack(np.vstack([xRight,xLeft,xLeft,xRight]).ravel()),
                       np.vstack(np.vstack([1-xRight, xLeft+1, -xLeft-1, xRight-1]).ravel())])
l2normData = np.hstack([np.vstack(np.vstack([xRight,xLeft,xLeft,xRight]).ravel()),
                       np.vstack(np.vstack([np.sqrt(1-xRight**2), np.sqrt(-xLeft**2+1), -np.sqrt(-xLeft**2+1), -np.sqrt(1-xRight**2)]).ravel())])

code = '''
    l2normData = np.hstack([
        np.vstack(
            np.vstack([xRight,xLeft,xLeft,xRight]).ravel()),
            np.vstack(
                np.vstack([
                    np.sqrt(1-xRight**2), 
                    np.sqrt(-xLeft**2+1), 
                    -np.sqrt(-xLeft**2+1), 
                    -np.sqrt(1-xRight**2)
                ]
            ).ravel()
        )
    ])
'''

st.code(code, language='python', line_numbers=True)

st.text("Create L1 and L2 data point object:")

code = '''
    l1_object = go.Scatter(
        x=l1normData[:,0], 
        y=l1normData[:, 1],
        name='Manhattan',
        mode='markers',
        marker_color='rgb(0,0,255)
    )

    l2_object = go.Scatter(
        x=l2normData[:,0], 
        y=l2normData[:, 1],
        name='Euclid',
        mode='markers',
        marker_color='rgb(255,0,0)'
    )
'''
st.code(code, language="python", line_numbers=True)

data1 = go.Scatter(
    x=l1normData[:,0], 
    y=l1normData[:, 1],
    name='Manhattan',
    mode='markers'
)
data2 = go.Scatter(
    x=l2normData[:,0], y=l2normData[:, 1],
    name='Euclid',
    mode='markers',
    marker_color='rgb(255,0,0)'
)

st.text("Setting layout for the same x-axis and y-axis:")

code = '''
    layout = go.Layout(
        autosize=False,
        width=800,
        height=800,
        xaxis=go.layout.XAxis(linecolor="black", linewidth=1, mirror=True),
        yaxis=go.layout.YAxis(linecolor="black", linewidth=1, mirror=True)
    )
'''
st.code(code,language="python", line_numbers=True)

st.text("Create figure object and show:")

code = '''
    fig = go.Figure(data=data, layout=layout)
    fig.show()
'''

st.code(code,language="python", line_numbers=True)

data = [data1, data2]

layout = go.Layout(
    autosize=False,
    width=800,
    height=800,
    xaxis=go.layout.XAxis(linecolor="black", linewidth=1, mirror=True),
    yaxis=go.layout.YAxis(linecolor="black", linewidth=1, mirror=True)
)
# margin=go.layout.Margin(l=50, r=50, b=100, t=100, pad=10),
fig = go.Figure(data=data, layout=layout)

st.text("\n")

st.plotly_chart(fig)