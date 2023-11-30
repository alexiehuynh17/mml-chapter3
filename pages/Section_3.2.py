import streamlit as st
import numpy as np # tinh toan
import numpy.linalg as npl # tinh toan
from sympy import *
from sympy.abc import x
import random
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff


st.set_page_config(page_title = "3.2 Inner Products") 

st.header("Inner Products") 

st.markdown("#### Example 3.3 (Inner Product That Is Not the Dot Product)")
st.markdown("##### Consider $V = R^2$ . If we define:")
st.latex(r'''
    \langle x,y\rangle := x_1 y_1 - (x_1 y_2 + x_2 y_1) + 2x_2 y_2
''')

st.markdown("##### then ⟨·, ·⟩ is an inner product but different from the dot product.")

st.markdown("Create two vector x and y:")

code = '''
    x = np.vstack([1,2])
    y = np.vstack([3,4])
'''

st.code(code,line_numbers=True)

x = np.vstack([1,2])
y = np.vstack([3,4])

dotProduct = np.dot(x,y.T)
innerProduct = x[0]*y[0] - (x[0]*y[1]+x[1]*y[0])+2*x[1]*y[1] 

st.markdown("Compute dot product and inner product:")
code = '''
    dotProduct = np.dot(x,y.T)
    innerProduct = x[0]*y[0]-(x[0]*y[1]+x[1]*y[0])+2*x[1]*y[1]
'''
st.code(code,line_numbers=True)

code = f'''
    dotProduct: {list(dotProduct)}
    innerProduct: {innerProduct}
'''

st.code(code)


st.markdown("#### Example 3.4 (Symmetric, Positive Definite Matrices)")

st.markdown("Create two matrixs A1 and A2:")

code = '''
    A1 = np.vstack([[9,6],[6,5]])
    A2 = np.vstack([[9,6],[6,3]])
'''
A1 = np.vstack([[9,6],[6,5]])
A2 = np.vstack([[9,6],[6,3]])
st.code(code,line_numbers=True)

st.markdown("x and y is belong V with respect to B:")

code = '''
    x = symbols('x')
    y = symbols('y')
'''

x = symbols('x')
y = symbols('y')

st.code(code,line_numbers=True)

st.markdown("Create objective function f for A1:")
code = '''
    A1 = Matrix(A1)
    A1_express = simplify(Matrix([x,y]).T*A1*Matrix([x,y]))[0] 
    # Matrix function can be used for create object matrix
    f = lambdify([x,y],A1_express)
'''

A1 = Matrix(A1)
A1_express = simplify(Matrix([x,y]).T*A1*Matrix([x,y]))[0] 
# Matrix function can be used for create object matrix
f = lambdify([x,y],A1_express)

st.code(code,line_numbers=True)

st.markdown("f funtions is:")
st.latex("9x^2 + 12xy + 5y^2")

st.markdown("Checking if A1 is symmetric, positive definite:")

code = '''
    checkingA1 = (np.array([f(-1,-1),f(-1,1),f(1,-1),f(1,1)]) > 0)
'''

st.code(code,line_numbers=True)

checkingA1 = (np.array([f(-1,-1),f(-1,1),f(1,-1),f(1,1)]) > 0)
code = f'''
    {checkingA1}
'''

st.code(code)

st.markdown("Matrix A1 is positive definite because it is symmetric and the expression output \nis always greater than 0.")

st.markdown("Create objective function g for A2:")
code = '''
    A2 = Matrix(A2)
    A2_express = simplify(Matrix([x,y]).T*A2*Matrix([x,y]))[0] 
    # Matrix function can be used for create object matrix
    g = lambdify([x,y],A2_express)
'''

A2 = Matrix(A2)
A2_express = simplify(Matrix([x,y]).T*A2*Matrix([x,y]))[0] 
# Matrix function can be used for create object matrix
g = lambdify([x,y],A2_express)

st.code(code,line_numbers=True)

st.markdown("g funtions is:")
st.latex("9x^2 + 12xy + 3y^2")

st.markdown("Checking if A2 is symmetric, positive definite:")

checkingA2 = (np.array([g(-1,-1),g(-1,1),g(1,-1),g(1,1)]) > 0)
code = '''
    checkingA2 = (np.array([g(-1,-1),g(-1,1),g(1,-1),g(1,1)]) > 0)
    # [True, False, False, True]
'''

st.code(code,line_numbers=True)

code = f'''
    {checkingA2}
'''

st.code(code)


st.markdown("The above expression is symmetric, but not positive definite because some \noutputs are less than 0.")

st.markdown(r"Change expression from $> 0$ to $\geq 0$")
code = '''
    checkingA2 = (np.array([g(-1,-1),g(-1,1),g(1,-1),g(1,1)]) >= 0)
'''


st.code(code,line_numbers=True)

code = f'''
    {checkingA2}
'''

st.code(code)

st.markdown("We see that some values are equal to 0 though, so we say only ≥ holds, and \nthus is symmetric, positive semidefinite.")

