import numpy as np # tinh toan
import numpy.linalg as npl # tinh toan
import matplotlib.pyplot as plt # hien thi cac so do, do thi
from matplotlib.pyplot import figure # hien thi cac so do, do thi
from sympy import *
from sympy.abc import x
from matplotlib import cm # hien thi cac so do, do thi
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

st.set_page_config(layout="wide")

st.header("Analytic Geometry")

st.markdown("<div style=\"text-align: justify\"> In this chapter, we will add some geometric interpretation and intuition to all of these concepts. In particular, we will look at geometric vectors and compute their lengths and distances or angles between two vectors. To be able to do this, we equip the vec- tor space with an inner product that induces the geometry of the vector space. Inner products and their corresponding norms and metrics capture the intuitive notions of similarity and distances, which we use to develop the support vector machine in Chapter 12. We will then use the concepts of lengths and angles between vectors to discuss orthogonal projections, which will play a central role when we discuss principal component analysis in Chapter 10 and regression via maximum likelihood estimation in Chapter 9. Figure 3.1 gives an overview of how concepts in this chapter are related and how they are connected to other chapters of the book.</div>", unsafe_allow_html=True)

image = Image.open('pages/images/ag_chapter3.png')

st.image(image, caption='A mind map of the concepts introduced in this chapter, along with when they are used in other parts of the book.')