import streamlit as st
import numpy as np # tinh toan
import numpy.linalg as npl # tinh toan
import matplotlib.pyplot as plt # hien thi cac so do, do thi
from matplotlib.pyplot import figure # hien thi cac so do, do thi
from sympy import *
from sympy.abc import x
from mpl_toolkits.mplot3d import Axes3D # hien thi cac so do, do thi
from matplotlib import cm # hien thi cac so do, do thi
import random


plt.rcParams[ "figure.figsize" ] = (10,10)
origin2D = np.vstack([0,0])
origin3D = np.vstack([0,0,0])
scale = 10

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

st.markdown("First we create set of data point:")

code = '''
    xRight = np.linspace(0,1,50)
    xLeft = np.linspace(-1,0,50)
    '''
st.code(code, language='python', line_numbers=True)

st.markdown("Manhattan: Building L1 (x,y) coordinates for 4 quadrants, where y in the first is \nof the form 1-x")

code = '''
    l1normData = np.hstack([
        np.vstack(
            np.vstack([xRight,xLeft,xLeft,xRight]).ravel()),
            np.vstack(np.vstack([1-xRight, xLeft+1, -xLeft-1, xRight-1]).ravel()
        )
    ])
    '''
st.code(code, language='python', line_numbers=True)

st.markdown("Euclidean: Building L1 (x,y) coordinates for 4 quadrants, where y in the first is \nof the form sqrt(1-x**2)")


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

st.markdown("Data and figure initialization:")

code = '''
    plt.axis([-scale/5,scale/5,-scale/5,scale/5])
    plt.grid(alpha=.1)
    plt.title("L1, Manhattan Norm, in red and L2, Euclidean Norm, in green.")

    plt.scatter(l1normData[:,0],l1normData[:,1],s=.5,color="r")
    plt.scatter(l2normData[:,0],l2normData[:,1],s=.5,color="g")
'''
st.code(code, language="python", line_numbers=True)

plt.axis([-scale/5,scale/5,-scale/5,scale/5])
plt.grid(alpha=.1)
plt.title("L1, Manhattan Norm, in red and L2, Euclidean Norm, in green.")
# Data and figure initialization.

plt.scatter(l1normData[:,0],l1normData[:,1],s=.5,color="r")
plt.scatter(l2normData[:,0],l2normData[:,1],s=.5,color="g")

st.text("\n")

st.pyplot(plt)