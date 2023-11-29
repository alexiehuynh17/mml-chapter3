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


plt.rcParams[ "figure.figsize" ] = (7,7)
origin2D = np.vstack([0,0])
origin3D = np.vstack([0,0,0])
scale = 10

st.set_page_config(page_title = "3.8 Orthogonal Projections")

st.text("This section begins with an implementation, and procedes with an explanation and reimplementation.")

st.text("Figure 3.9")

st.text("Initializing data and using least squares regression as fit line:")

code = '''
    xs = np.vstack([-3.2,-2.2,-2,-1.7,-1.7,-1.5,-1.3,-.4,-.3,0,.5,.6,.8,1,1.1,1.2,1.3,1.7,2.3,3.2])
    ys = np.vstack(np.random.normal(0,.8,len(xs)))
    points = np.hstack([xs,ys])
    lsq = npl.lstsq(points,np.random.randn(20), rcond=-1)[0]

    posPts = [p for p in points if p[0]>=0] # Splitting into positive and negative to keep a 0,0 origin.
    negPts = [p for p in points if p[0]<0]
    maxNeg = np.hstack([xs[0],[xs*(lsq[0]/lsq[1])][0][0]]) # -Line for data to project onto.
    maxPos = np.hstack([xs[::-1][0],[xs*(lsq[0]/lsq[1])][0][::-1][0]]) # +Line for data to project onto.
'''

st.code(code, line_numbers=True)

xs = np.vstack([-3.2,-2.2,-2,-1.7,-1.7,-1.5,-1.3,-.4,-.3,0,.5,.6,.8,1,1.1,1.2,1.3,1.7,2.3,3.2])
ys = np.vstack(np.random.normal(0,.8,len(xs)))
points = np.hstack([xs,ys])
lsq = npl.lstsq(points,np.random.randn(20), rcond=-1)[0]

posPts = [p for p in points if p[0]>=0] # Splitting into positive and negative to keep a 0,0 origin.
negPts = [p for p in points if p[0]<0]
maxNeg = np.hstack([xs[0],[xs*(lsq[0]/lsq[1])][0][0]]) # -Line for data to project onto.
maxPos = np.hstack([xs[::-1][0],[xs*(lsq[0]/lsq[1])][0][::-1][0]]) # +Line for data to project onto.


st.text("Figure setup:")

code = '''
    plt.axis([-4,4,-4,4])
    plt.plot(xs,xs*(lsq[0]/lsq[1]),c="k")
    plt.scatter(points[:,0],points[:,1])
    plt.grid(alpha=.1)

    # Iteratively finding each projection and plotting it.
    for i in negPts:
        newQ = (np.dot(i,maxNeg)/npl.norm(maxNeg)**2)*maxNeg # Projection step: p = v2 * (dot(v1,v2)/norm(v2)**2).
        plt.quiver(*origin2D,*newQ,scale=8,width=.003,color="orange")  # Projection vector.
        currV = newQ-i
        plt.quiver(*i,*currV,scale=8,width=.003,color="orange")
        plt.annotate("   " + str(round(npl.norm(currV),1)),i)
    for i in posPts:
        newQ = (np.dot(i,maxPos)/npl.norm(maxPos)**2)*maxPos # Projection step: p = v2 * (dot(v1,v2)/norm(v2)**2).
        plt.quiver(*origin2D,*newQ,scale=8,width=.003,color="orange")  # Projection vector.
        currV = newQ-i
        plt.quiver(*i,*currV,scale=8,width=.003,color="orange")
        plt.annotate("   " + str(round(npl.norm(currV),1)),i)
    # Iteratively finding each projection and plotting it.

    plt.title("Orthogonal projection of a two-dimensional dataset onto a one-dimensional subspace.")
'''

st.code(code, line_numbers=True)

# Figure setup.
plt.axis([-4,4,-4,4])
plt.plot(xs,xs*(lsq[0]/lsq[1]),c="k")
plt.scatter(points[:,0],points[:,1])
plt.grid(alpha=.1)
# Figure setup.
print()
# Iteratively finding each projection and plotting it.
for i in negPts:
    newQ = (np.dot(i,maxNeg)/npl.norm(maxNeg)**2)*maxNeg # Projection step: p = v2 * (dot(v1,v2)/norm(v2)**2).
    plt.quiver(*origin2D,*newQ,scale=8,width=.003,color="red")  # Projection vector.
    currV = newQ-i
    plt.quiver(*i,*currV,scale=8,width=.003,color="orange")
    plt.annotate("   " + str(round(npl.norm(currV),1)),i)
for i in posPts:
    newQ = (np.dot(i,maxPos)/npl.norm(maxPos)**2)*maxPos # Projection step: p = v2 * (dot(v1,v2)/norm(v2)**2).
    plt.quiver(*origin2D,*newQ,scale=8,width=.003,color="red")  # Projection vector.
    currV = newQ-i
    plt.quiver(*i,*currV,scale=8,width=.003,color="orange")
    plt.annotate("   " + str(round(npl.norm(currV),1)),i)
# Iteratively finding each projection and plotting it.

plt.title("Orthogonal projection of a two-dimensional dataset onto a one-dimensional subspace.")

st.pyplot(plt)

st.text("Figure 3.10: Examples of projections onto one-dimensional subspaces.")

code = '''
    v1 = np.vstack([1,2])
    v2 = np.vstack([2,1])
    projec = lambda v1,v2: (np.dot(v2.T,v1.T)/npl.norm(v1.T)**2) * v2 # v2 * (dot(v1,v2)/norm(v2)**2)
    v3 = projec(v1.ravel(),v2.ravel()) # Projection step.
    v4 = v3.ravel() - v1.ravel()

    cosAngle = lambda x, y: (np.dot(x.T,y)/np.sqrt(np.dot(
                                np.dot(x.T,x),
                                np.dot(y.T,y))
                            ))[0][0]
    angle = lambda cos: np.arccos(cos)*(180/np.pi)
'''

v1 = np.vstack([1,2])
v2 = np.vstack([2,1])
projec = lambda v1,v2: (np.dot(v2.T,v1.T)/npl.norm(v1.T)**2) * v2 # v2 * (dot(v1,v2)/norm(v2)**2)
v3 = projec(v1.ravel(),v2.ravel()) # Projection step.
v4 = v3.ravel() - v1.ravel()

st.text("Figure setup:")

cosAngle = lambda x, y: (np.dot(x.T,y)/np.sqrt(np.dot(
                                np.dot(x.T,x),
                                np.dot(y.T,y))
                            ))[0][0]
angle = lambda cos: np.arccos(cos)*(180/np.pi)

code = '''
    plt.axis([-scale/2,scale/2,-scale/2,scale/2])
    plt.suptitle(r"(a) Projection of $x \in \mathbb{R}^2$ onto a subspace $U$ with basis vector $b$.",size=15)
    plt.title("Displacement of %1.1f." %npl.norm(v3))
    plt.grid(alpha=.1)

    # Plotting projection.
    plt.quiver(*origin2D, *v1, scale=scale, color = "k")
    plt.quiver(*origin2D, *v2, scale=scale, color = "orange")
    plt.scatter(*v3,color="k")
    plt.quiver(*origin2D,*v3,scale=scale, width = .005,color="blue")
    plt.quiver(*v1,*v4,scale=scale,width=.002,color="red")
    # Plotting projection.

    plt.legend(["x","U",r"$\pi_U(x)$","b"])
'''

st.code(code,line_numbers=True)

plt.clf()
plt.axis([-scale/2,scale/2,-scale/2,scale/2])
plt.suptitle(r"(a) Projection of $x \in \mathbb{R}^2$ onto a subspace $U$ with basis vector $b$.",size=15)
plt.title("Displacement of %1.1f." %npl.norm(v3))
plt.grid(alpha=.1)

# Plotting projection.
plt.quiver(*origin2D, *v1, scale=scale, color = "k")
plt.quiver(*origin2D, *v2, scale=scale, color = "orange")
plt.scatter(*v3,color="k")
plt.quiver(*origin2D,*v3,scale=scale, width = .005,color="blue")
plt.quiver(*v1,*v4,scale=scale,width=.002,color="red")
# Plotting projection.

# plt.annotate("w = " + str(round(angle(cosAngle(v1,v2)),2))+r"$^\circ$",xy=origin2D)
plt.legend(["x","U",r"$\pi_U(x)$","b"])

st.pyplot(plt)

st.markdown(r"1. The projection $\pi_U(x)$ is closest to $x$, where \"closest\" implies that the distance $||x-\pi_U(x)||$ is minimal. It follows that the (red) segment is orthogonal to $U$, and therefore the basis vector $b$ of $U$. The orthogonality condition yields $\langle \pi_U(x) - x, b\rangle = 0$, since angles between vectors are defined via the inner product.")

st.markdown("Checking orthogonality condition:")

code = '''round(np.inner(v3 - v1.T,v2.T)[0][0],5) # 0.0'''

st.code(code, line_numbers=True)

st.markdown("2. The projection $\pi_U(x)$ of $x$ onto $U$ must be an lement of $U$ and, therefore, a multiple of the basis vector $b$ that spansU. Hence, $\pi_U(x) = \lambda b$, for some $\lambda \in \mathbb{R}$.")

code = '''
    v1 = np.vstack(np.array([1,0]))
    v2 = np.vstack(l2normData[34:35][0])
    # [0.693877551020408, 0.720093010790912]

    v3 = projec(v2.ravel(),v1.ravel()) # Projection step.
    # [0.693877551020408, 0]

    v4 = v3.ravel()-v2.ravel()
    # [0, −0.720093010790912]
    '''

st.code(code,line_numbers=True)


xRight = np.linspace(0,1,50)
xLeft = np.linspace(-1,0,50) 
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

v1 = np.vstack(np.array([1,0]))
v2 = np.vstack(l2normData[34:35][0])
v3 = projec(v2.ravel(),v1.ravel()) # Projection step.
v4 = v3.ravel()-v2.ravel()

st.markdown("Figure setup")

code = '''
    # Figure setup.
    plt.axis([-scale/2,scale/2,-scale/2,scale/2])
    plt.grid(alpha=.1)
    plt.scatter(l2normData[:,0],l2normData[:,1],s=.5,color="k") # Circle data from prior figure on the L2 norm.
    plt.title("(b) Projection of a two-dimensional vector (black) with ||x|| = 1 \n onto a one-dimensional subspace spanned by b (orange)." + " \n Projection magnitude %1.1f." %npl.norm(v3));
    # Figure setup.

    # Projection plotting.
    plt.quiver(*origin2D, *v1,scale=scale, color="orange", width=.005)
    plt.quiver(*origin2D, *v2,scale=scale, color="k", width=.005)
    plt.annotate("w = " + str(round(angle(cosAngle(v1,v2)),2))+r"$^\circ$",xy=origin2D-.2, size = 8)
    plt.quiver(*origin2D,*v3,scale=scale,width=.002,color="red")
    plt.quiver(*v2,*v4,scale=scale,width=.002,color="blue");
    # Projection plotting.
'''

st.code(code,line_numbers=True)

# Figure setup.
plt.clf()
plt.axis([-scale/2,scale/2,-scale/2,scale/2])
plt.grid(alpha=.1)
plt.scatter(l2normData[:,0],l2normData[:,1],s=.5,color="k") # Circle data from prior figure on the L2 norm.
plt.title("(b) Projection of a two-dimensional vector (black) with ||x|| = 1 \n onto a one-dimensional subspace spanned by b (orange)." + " \n Projection magnitude %1.1f." %npl.norm(v3));
# Figure setup.

# Projection plotting.
plt.quiver(*origin2D, *v1,scale=scale, color="orange", width=.005)
plt.quiver(*origin2D, *v2,scale=scale, color="k", width=.005)
plt.annotate("w = " + str(round(angle(cosAngle(v1,v2)),2))+r"$^\circ$",xy=origin2D-.2, size = 8)
plt.quiver(*origin2D,*v3,scale=scale,width=.002,color="red")
plt.quiver(*v2,*v4,scale=scale,width=.002,color="blue");
# Projection plotting.

st.pyplot(plt)

st.markdown(r"Three steps to determine a projection between any vector x and some basis vector, b, for a subspace, U.\
1. Find the scalar, $\lambda$ for b. Given that some projection, p, exists, the inner product between x-p and b, will equal 0.\
    a. We know that the projection, p, can also be written as a scalar operation of our basis vector, b, so we say $\langle x-\lambda b,b \rangle= 0$.\
    b. We can then isolate the scalar, $\lambda$, by seeing that the above is equal to the inner product of x and b minus $\lambda \langle b,b \rangle = 0$ \
    c. $\lambda = \frac{\langle x,b \rangle}{\langle b,b \rangle}$ which can also be written as the inner product between b and x divided by the squared norm of b, $\frac{\langle b,x \rangle}{||b||^2}$.\
2. Find the point on U that the projection, p, will create. Seeing as $p = \lambda b$, we take (c) from above to find $p = \frac{\langle x,b \rangle}{\langle b,b \rangle} b$.\
3. Find a projection matrix $P_p$. We know a projection is simply a linear mapping, so we should be able to find a matrix, $P$, that takes any vector, $x$, and maps it, or creates a new projected vector, $p$, to our subspace or line, such that $p = P_p x$, and $P_p = \frac{p}{x}$.\
    a. Seeing as $p = \frac{\langle x,b \rangle}{\langle b,b \rangle} b$, from 2. above, $P_p  = \frac{1}{x}\frac{\langle x,b \rangle}{\langle b,b \rangle} b = \frac{bb^T}{\langle b,b \rangle}$\
**Applying these steps to Figure 3.9.**")
