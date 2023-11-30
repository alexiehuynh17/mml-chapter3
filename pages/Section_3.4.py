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

st.set_page_config(page_title = "3.4 Angles and Orthogonality") 

st.header("Angles and Orthogonality") 

x = np.linspace(0,np.pi,100)
y = np.cos(x)

st.markdown("Figure 3.4")

st.markdown('Create set of data point with y = cos(x):')
code = '''
    x = np.linspace(0,np.pi,100)
    y = np.cos(x)
'''

st.code(code,line_numbers=True)

st.markdown("Create figure:")

code = '''
    plt.scatter(x, y)
    plt.grid(alpha=.1)
    plt.title(r"When restricted to $[0,\pi]$, then $f(w) = cos(w)$ returns a \\
                unique number in the interval $[-1,1]$.");
'''
st.code(code,line_numbers=True)

plt.scatter(x, y)
plt.grid(alpha=.1)
plt.title(r"When restricted to $[0,\pi]$, then $f(w) = cos(w)$ returns a unique number in the interval $[-1,1]$.")

st.pyplot(plt)

st.markdown("#### Example 3.6 (Angle between Vectors)")

st.latex(r'''
    \cos\omega = \frac{\langle x,y\rangle}{\sqrt{\langle x,x\rangle \langle y,y\rangle}}=\frac{x^{\top}y}{\sqrt{x^{\top}xy^{\top}y}}
''')

st.markdown("Create two vectors x and y:")

code = '''
    x = np.vstack([1,1])
    y = np.vstack([1,2])
'''

st.code(code,line_numbers=True)

st.markdown("Compute angle using inner product:")

x = np.vstack([1,1])
y = np.vstack([1,2])
cosAngle = lambda x, y: (np.dot(x.T,y)/np.sqrt(np.dot(
                                np.dot(x.T,x),
                                np.dot(y.T,y))
                            ))[0][0]
angle = lambda cos: np.arccos(cos)*(180/np.pi)

angleXY = angle(cosAngle(x,y))

code = '''
    cosAngle = lambda x, y: (np.dot(x.T,y)/np.sqrt(np.dot(
                                np.dot(x.T,x),
                                np.dot(y.T,y))
                            ))[0][0]
    angle = lambda cos: np.arccos(cos)*(180/np.pi)

    angleXY = angle(cosAngle(x,y))
'''

st.code(code,line_numbers=True)

code = f'''
    {angleXY}
'''

st.code(code)

st.markdown("Figure 3.5:")

code = '''
    plt.grid(alpha=.1)
    plt.legend(["x","y"])
    plt.axis([-scale/5,scale/5,-scale/5,scale/5])
    plt.title(r"Angle of %1.2f$^\circ$ for vectors $x$ and $y$." %round(angle(cosAngle(x,y)),2))

    plt.quiver(*origin2D, *x,scale=scale,color="lightblue", width=.005)
    plt.quiver(*origin2D, *y,scale=scale,color="blue", width=.005)
    plt.annotate(str(round(angle(cosAngle(x,y)),2))+r"$^\circ$",xy=(scale/50,scale/30)); 
'''

st.code(code,line_numbers=True)

plt.clf()

plt.grid(alpha=.1)
plt.legend(["x","y"])
plt.axis([-scale/5,scale/5,-scale/5,scale/5])
plt.title(r"Angle of %1.2f$^\circ$ for vectors $x$ and $y$." %round(angle(cosAngle(x,y)),2))

plt.quiver(*origin2D, *x,scale=scale,color="lightblue", width=.005)
plt.quiver(*origin2D, *y,scale=scale,color="blue", width=.005)
plt.annotate(str(round(angle(cosAngle(x,y)),2))+r"$^\circ$",xy=(scale/50,scale/30)); 


st.pyplot(plt)

st.markdown("#### Example 3.7 (Orthogonal Vectors)")

st.markdown("Create two vectors x and y:")

code = '''
    x = np.vstack([1,1])
    y = np.vstack([-1,1])
'''
x = np.vstack([1,1])
y = np.vstack([-1,1])
angleXY = angle(cosAngle(x,y))
st.code(code,line_numbers=True)
st.markdown("Compute angle:")

code = '''
    angleXY = angle(cosAngle(x,y))
'''

st.code(code,line_numbers=True)

code = f'''
    {angleXY}
'''

st.code(code)
st.markdown("Figure 3.6:")

code = '''
    plt.grid(alpha=.1)
    plt.legend(["x","y"])
    plt.axis([-scale/5,scale/5,-scale/5,scale/5])
    plt.title(r"Angle of %1.2f$^\circ$ for vectors $x$ and $y$." %round(angle(cosAngle(x,y)),2))

    plt.quiver(*origin2D, *x,scale=scale,color="lightblue", width=.005)
    plt.quiver(*origin2D, *y,scale=scale,color="blue", width=.005)
    plt.annotate(str(round(angleXY),2))+r"$^\circ$",xy=origin2D);
'''

st.code(code,line_numbers=True)

plt.clf()
plt.grid(alpha=.1)
plt.legend(["x","y"])
plt.axis([-scale/5,scale/5,-scale/5,scale/5])
plt.title(r"Angle of %1.2f$^\circ$ for vectors $x$ and $y$." %round(angle(cosAngle(x,y)),2))

plt.quiver(*origin2D, *x,scale=scale,color="lightblue", width=.005)
plt.quiver(*origin2D, *y,scale=scale,color="blue", width=.005)
plt.annotate(str(round(angleXY,2))+r"$^\circ$",xy=origin2D);

st.pyplot(plt)


st.markdown("By changing how inner is induced, such as with the matrix:")
st.latex(r'''
            \begin{bmatrix}
                2 & 0\\
                0 & 1
                \end{bmatrix}
         ''')
st.markdown("We find the vectors are no longer orthogonal, despite their being orthogonal \nwith respect to another inner product.")

code = '''
    x = np.vstack([1,1])
    y = np.vstack([-1,1])
    inner = np.vstack([[2,0],[0,1]])
    omega = (x.T@inner@y)/npl.norm(x.T@inner@x)
    newAngle = angle(omega) # 109.47122063
'''

st.code(code, line_numbers=True)
