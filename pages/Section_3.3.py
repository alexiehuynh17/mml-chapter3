import streamlit as st
import numpy as np # tinh toan
import numpy.linalg as npl # tinh toan
from sympy import *
from sympy.abc import x
import random
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(page_title = "3.3 Lengths and Distances") 

st.header("Lengths and Distances")  

st.markdown("Norms can be used to find the length of a vector, to find some norms, e.g. Manhattan Norm does not require an inner product. A norm can be found by taking the square root of the inner product.")

st.markdown("Create vector x0:")

code = '''
    x0 = np.vstack([1,1])
'''

x0 = np.vstack([1,1])
st.code(code,line_numbers=True)

st.markdown("Compute length using dot product and inner product")
code = '''
    dotLength = np.sqrt(np.dot(x0.T,x0)) 
    innerLenght = x0[0]*x0[0] - (x0[0]*x0[1]+x0[1]*x0[0])+2*x0[1]*x0[1]
'''

st.code(code,line_numbers=True)

dotLength = np.sqrt(np.dot(x0.T,x0)) 

innerLength = x0[0]*x0[0] - (x0[0]*x0[1]+x0[1]*x0[0])+2*x0[1]*x0[1]

code = f'''
    dotLength: {dotLength}
    innerLength: {innerLength}
'''

st.code(code)

st.markdown("Using the inner product noted above, we get a \"shorter\" length.")

