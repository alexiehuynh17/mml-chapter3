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

st.set_page_config(page_title = "3.6 Orthogonal Complement")

st.header("Orthonormal Complement")  

st.markdown(r"Any plane in $\mathbb{R}^3$ can be described by some vector orthogonal to it, \nthis vector is said to span it's orthogonal complement, and is known as the normal vector of a plane.")

st.text("Figure 3.7:")


code = '''
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    scale3D = 10
    ax.set_xlim3d(-scale3D,scale3D)
    ax.set_ylim3d(scale3D,-scale3D)
    ax.set_zlim3d(-scale3D,scale3D)
    point  = origin3D.T[0]
    ax.grid(b=None)

    colors = ["r","g","b"]
    normalVecs = np.vstack([[-1,1,1],[1,-1,1],[1,1,-1]])
    for c,i in enumerate(normalVecs): # Using index of vectors to iterate colors.
        d = np.dot(-point,i)
        xx, yy = np.meshgrid(np.linspace(-scale3D/5,scale3D/5,10), 
                            np.linspace(-scale3D/5,scale3D/5,10))
        z = (d-i[0] * xx - i[1] * yy) /i[2] #ax+by+cz = d ∴ z = (d-ax-by)/c
        ax.plot_surface(xx, yy, z, alpha=0.2,color = colors[c])
        ax.quiver(*point,*i, length=scale3D/2,color=colors[c])

    ax.set_title("Normal Vectors with their Planes");
'''

st.code(code,line_numbers=True)

# Figure setup.
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
scale3D = 10
ax.set_xlim3d(-scale3D,scale3D)
ax.set_ylim3d(scale3D,-scale3D)
ax.set_zlim3d(-scale3D,scale3D)
point  = origin3D.T[0]
ax.grid(b=None)
# Figure setup.

colors = ["r","g","b"]
normalVecs = np.vstack([[-1,1,1],[1,-1,1],[1,1,-1]])
for c,i in enumerate(normalVecs): # Using index of vectors to iterate colors.
    d = np.dot(-point,i)
    xx, yy = np.meshgrid(np.linspace(-scale3D/5,scale3D/5,10), np.linspace(-scale3D/5,scale3D/5,10))
    z = (d-i[0] * xx - i[1] * yy) /i[2] #ax+by+cz = d ∴ z = (d-ax-by)/c
    ax.plot_surface(xx, yy, z, alpha=0.2,color = colors[c])
    ax.quiver(*point,*i, length=scale3D/2,color=colors[c])

ax.set_title("Normal Vectors with their Planes");

st.pyplot(fig)