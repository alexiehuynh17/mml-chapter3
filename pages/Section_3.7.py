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

st.set_page_config(page_title = "3.7 Inner Product of Functions")

st.header("Inner Product of Functions")  

st.markdown("An inner product of two functions u and v mapping from  R  to  R , can be defined as the following.")

st.latex(r'''
    \langle u, v \rangle := \int^b_au(x)v(x)dx
''')

st.markdown("Example 3.9 (Inner Product of Functions)")

st.latex(r'''
    u = sin(x)
''')
st.latex(r'''
    v = cos(x)
''')

st.latex(r'''f(x)=u(x)v(x)''')

st.markdown("Figure 3.8")

st.markdown("Create x and y:")

code = '''
    x = np.linspace(-np.pi,np.pi,100)
    y = lambda x: np.sin(x)*np.cos(x)
'''
st.code(code,line_numbers=True)


x = np.linspace(-np.pi,np.pi,100)
y = lambda x: np.sin(x)*np.cos(x)

st.markdown("Figure setup:")

code = '''
    plt.plot(x,y(x))
    plt.grid(alpha=.1)
    plt.title("sin(x)cos(x)")
'''
st.code(code,line_numbers=True)

plt.plot(x,y(x))
plt.grid(alpha=.1)
plt.title("sin(x)cos(x)")

st.pyplot(plt)

st.markdown("This function is odd, f(-x) = -f(x), so the integral from -_ to _ will evaluate \nto 0, thus sine and cosine are orthogonal.")

code = '''
    y(-2), -y(2)
    # (0.37840124765396416, 0.37840124765396416)
'''

st.code(code,line_numbers=True)

st.markdown("An integration of the function sin(x)cos(x) with respect to x, over -pi and pi, \nas stated above, evaluates to 0.")

code = '''
    symbX = Symbol("x")
    integrate(sin(symbX)*cos(symbX),(symbX,-pi,pi)) # 0
'''

st.code(code,line_numbers=True)

code = '''
    round(sum(np.sin(x)*np.cos(x))) # 0
'''

st.code(code,line_numbers=True) 


st.markdown("It also holds that the collection of functions")

st.latex(r'''{1,cos(x)...cos(n\times x)}''')

st.markdown("is orthogonal if we integrate from  −π  to  π , i.e. any pair are orthogonal.")
code = '''
    integrate(cos(symbX)*cos(3*symbX),(symbX,-pi,pi)) # 0
'''
