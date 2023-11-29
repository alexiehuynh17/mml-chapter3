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

st.set_page_config(page_title = "3.5 Orthonormal Basis")

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

angleb1b2 = angle(cosAngle(b1,b2))

code = ''' 
    cosAngle = lambda x, y: (np.dot(x.T,y)/np.sqrt(np.dot(
                                np.dot(x.T,x),
                                np.dot(y.T,y))
                            ))[0][0]
    angle = lambda cos: np.arccos(cos)*(180/np.pi)

    plt.grid(alpha=.1)
    b1 = np.vstack([1,1])*(1/np.sqrt(2))
    b2 = np.vstack([1,-1])*(1/np.sqrt(2))
    plt.title(r"Orthonormal basis since: $angle(dot(b_1^T,b_2))$ = %1.2f," %angleb1b2)

    plt.quiver(*origin2D,*b1,scale=scale, color = "lightblue")
    plt.quiver(*origin2D,*b2,scale=scale, color = "gold")
    plt.legend([r"$b_1$",r"$b_2$"])
'''

st.code(code, line_numbers=True)

cosAngle = lambda x, y: (np.dot(x.T,y)/np.sqrt(np.dot(
                                np.dot(x.T,x),
                                np.dot(y.T,y))
                            ))[0][0]
angle = lambda cos: np.arccos(cos)*(180/np.pi)

plt.grid(alpha=.1)
b1 = np.vstack([1,1])*(1/np.sqrt(2))
b2 = np.vstack([1,-1])*(1/np.sqrt(2))
plt.title(r"Orthonormal basis since: $angle(dot(b_1^T,b_2))$ = %1.2f," %angleb1b2)

plt.quiver(*origin2D,*b1,scale=scale, color = "lightblue")
plt.quiver(*origin2D,*b2,scale=scale, color = "gold")
plt.legend([r"$b_1$",r"$b_2$"])

st.pyplot(plt)