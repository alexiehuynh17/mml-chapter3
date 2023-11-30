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
    a. Seeing as $p = \frac{\langle x,b \rangle}{\langle b,b \rangle} b$, from 2. above, $P_p  = \frac{1}{x}\frac{\langle x,b \rangle}{\langle b,b \rangle} b = \frac{bb^T}{\langle b,b \rangle}$")

st.markdown("**Applying these steps to Figure 3.9.**")

# Initializing data and using least squares regression as fit line.
xs = np.vstack([-3.2,-2.2,-2,-1.7,-1.7,-1.5,-1.3,-.4,-.3,0,.5,.6,.8,1,1.1,1.2,1.3,1.7,2.3,3.2])
ys = np.vstack(np.random.normal(0,.8,len(xs)))
points = np.hstack([xs,ys])
lsq = npl.lstsq(points,np.random.randn(20), rcond=-1)[0]
# Initializing data and using least squares regression as fit line.

posPts = [p for p in points if p[0]>=0] # Splitting into positive and negative to keep a 0,0 origin.
negPts = [p for p in points if p[0]<0]
maxNeg = np.hstack([xs[0],[xs*(lsq[0]/lsq[1])][0][0]]) # -Line for data to project onto.
maxPos = np.hstack([xs[::-1][0],[xs*(lsq[0]/lsq[1])][0][::-1][0]]) # +Line for data to project onto.

code = '''
    data = np.vstack([negPts,posPts])
    pB = np.vstack(maxPos) # basis vector
    nB = np.vstack(maxNeg)

    sampleX = np.vstack(data[0])
'''

st.code(code, line_numbers=True)

data = np.vstack([negPts,posPts])
pB = np.vstack(maxPos) # basis vector
nB = np.vstack(maxNeg)

sampleX = np.vstack(data[0])

# st.markdown("Output:")
code = f'''
    pB: {list(pB)}
    nB: {list(nB)}
    sampleX: {list(sampleX)}
'''

st.code(code)

st.markdown("**Step 1.** Find lambda. Recall lambda is the scalar applied to B that represents the coordinates of the projection.")

code = '''
    sampleLam = np.dot(nB.T,sampleX)/npl.norm(nB)**2
    # b^T x / ||b||^2; recall inner(x,b) = dot(b^T,x)
'''

st.code(
    code, line_numbers=True)

sampleLam = np.dot(nB.T,sampleX)/npl.norm(nB)**2
code = f'''
    sampleLam: {list(sampleLam)}
'''

st.code(code)

st.markdown("**Step 2.** Find projected point using lambda.")

code = '''
    sampleP = nB*sampleLam
'''

st.code(code, line_numbers=True)

sampleP = nB*sampleLam

code = f'''
    sampleP: {list(sampleP)}
'''

st.code(code)

st.markdown("**Step 3.** Find projection matrix, P. For any data point, x, the dot product between P and x will yield the new data point projected on the line.")

pMatrix = np.dot(nB,nB.T)/npl.norm(nB)**2 

code = '''
    pMatrix = np.dot(nB,nB.T)/npl.norm(nB)**2 
    # bb^T/||b||^2
'''

st.code(code, line_numbers=True)

code = f'''
    pMatrix: {list(pMatrix)}
'''
st.code(code)

st.markdown("Figure setup:")

code = '''
    plt.axis([-scale/2,scale/2,-scale/2,scale/2])
    plt.grid(alpha=.1)
    plt.scatter(data[:,0],data[:,1],color="orange")
    plt.plot(xs,xs*(lsq[0]/lsq[1]),c="k")
    plt.quiver(*origin2D,*maxNeg,scale=scale)
    plt.quiver(*origin2D,*maxPos,scale=scale)
    plt.quiver(*origin2D,*sampleP,scale=scale,color="r")
'''

st.code(code, line_numbers=True)

plt.clf()
plt.axis([-scale/2,scale/2,-scale/2,scale/2])
plt.grid(alpha=.1)
plt.scatter(data[:,0],data[:,1],color="orange")
plt.plot(xs,xs*(lsq[0]/lsq[1]),c="k")
plt.quiver(*origin2D,*maxNeg,scale=scale)
plt.quiver(*origin2D,*maxPos,scale=scale)
plt.quiver(*origin2D,*sampleP,scale=scale,color="r")

st.pyplot(plt)

st.markdown("### Example 3.10 (Projection onto a Line)")

b = np.vstack([1,2,2])

code = '''
    b = np.vstack([1,2,2])
    P = np.dot(b,b.T)/npl.norm(b)**2
    # bb^T/||b||^2
    x = np.vstack([1,1,1])
'''
st.code(code, line_numbers=True)
b = np.vstack([1,2,2])
P = np.dot(b,b.T)/npl.norm(b)**2
# bb^T/||b||^2
x = np.vstack([1,1,1])

code = f'''
    P: {P}
'''
st.code(code)


plt.rcParams[ "figure.figsize" ] = (10,10)
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.grid(b=None)
scale3D = 10
ax.set_xlim3d(scale3D,-scale3D) # Inverted graph orientation for figure clarity.
ax.set_ylim3d(-scale3D,scale3D)
ax.set_zlim3d(scale3D,-scale3D)
point  = origin3D.T[0]
ax.quiver(*origin3D, *b,length=scale3D, color = "lightblue",lw=3)
ax.quiver(*origin3D, *x,length=scale3D, color = "orange",lw=3)
ax.quiver(*origin3D, *np.dot(P,x), length=scale3D, color="lightgreen", lw=2)
ax.set_title("Example 3.10", size=15)

code = '''
    plt.rcParams[ "figure.figsize" ] = (10,10)
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.grid(b=None)
    scale3D = 10
    ax.set_xlim3d(scale3D,-scale3D) # Inverted graph orientation for figure clarity.
    ax.set_ylim3d(-scale3D,scale3D)
    ax.set_zlim3d(scale3D,-scale3D)
    point  = origin3D.T[0]
    ax.quiver(*origin3D, *b,length=scale3D, color = "lightblue",lw=3)
    ax.quiver(*origin3D, *x,length=scale3D, color = "orange",lw=3)
    ax.quiver(*origin3D, *np.dot(P,x), length=scale3D, color="lightgreen", lw=2)
    ax.set_title("Example 3.10", size=15)
'''

st.code(code, line_numbers=True)

st.pyplot(fig)

st.markdown("**Steps 1-3 revisited for projection onto a general subspace.**")
st.markdown(r"1. Find lambdas of the projection with respect to the basis of the subspace.\
    a. $\lambda = (B^TB)^{-1}B^Tx$. Where $\lambda$ is the set of all lambdas, and $B$ is the set of all bases.\
2. Find projection by multiplying lambda by B.\
3. Find the general projection matrix by dividing the projection in 2 by x.")

st.markdown("### Example 3.11 (Projection onto a Two-dimensional Subspace)")

basis1 = np.vstack([1,1,1])
basis2 = np.vstack([0,1,2])
B = np.hstack([basis1,basis2])
x = np.vstack([6,0,0])
proj = np.vstack([5,2,-1])
projDisplacement = x - proj

code = '''
    basis1 = np.vstack([1,1,1])
    basis2 = np.vstack([0,1,2])
    B = np.hstack([basis1,basis2])
    x = np.vstack([6,0,0])
    proj = np.vstack([5,2,-1])
    projDisplacement = x - proj
'''

st.code(code, line_numbers=True)

code = f'''
    B: {list(B)}
    projDisplacement: {list(projDisplacement)}
'''

st.code(code)

st.markdown(r"The projection was found using the steps noted above.")
st.markdown(r"1. $\lambda = (B^TB)^{-1}B^Tx$.")
st.markdown(r"2. $\pi = B\lambda$")
st.markdown(r"3. $\mathbb{P} = \frac{B}{x}\lambda$")

code = '''
    lam = npl.inv(B.T@B)@B.T@x     # 1
    pi = B@lam                     # 2
    P = B@npl.inv(B.T@B)@B.T       # 3
'''

st.code(code,line_numbers=True)

lam = npl.inv(B.T@B)@B.T@x     # 1
pi = B@lam                     # 2
P = B@npl.inv(B.T@B)@B.T       # 3

code = f'''
    lam: {list(lam)}
    pi: {list(pi)}
    P: {list(P)}
'''

st.code(code)

code = '''
    xTest1 = np.vstack([1,-.5,3])
    xTest2 = np.vstack([-2,1,0])
    pTest1 = P@xTest1
    pTest2 = P@xTest2
    displ1 = xTest1 - pTest1
    displ2 = xTest2 - pTest2
'''

st.code(code, line_numbers=True)

xTest1 = np.vstack([1,-.5,3])
xTest2 = np.vstack([-2,1,0])
pTest1 = P@xTest1
pTest2 = P@xTest2
displ1 = xTest1 - pTest1
displ2 = xTest2 - pTest2

code = f'''
    pTest1: {list(pTest1)}
    pTest2: {list(pTest2)}
    displ1: {list(displ1)}
    displ2: {list(displ2)}
'''

st.code(code)

st.markdown("Figure setup:")

code = '''
    fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.grid(b=None)
scale3D = 5
ax.set_xlim3d(-scale3D/1,scale3D/1)
ax.set_ylim3d(scale3D/1,-scale3D/1)
ax.set_zlim3d(-scale3D/1,scale3D/1)
ax.set_title(r"Projection onto the span of bases $[1 , 1 , 1]^T [0 , 1 , 2]^T$, for vector $[6,0,0]^T$ (in red)." + "\n Aswell as other vectors.")
# 3D Figure Formatting.

# Span visualization.
quiveropts = dict(color='lightblue', linewidths = .5,arrow_length_ratio = .05, pivot='tail')
spanScalar = 6
ax.quiver(*origin3D,*B[:,0]*spanScalar,**quiveropts)
ax.quiver(*origin3D,*B[:,1]*spanScalar,**quiveropts)
ax.quiver(*origin3D,*-B[:,0]*spanScalar,**quiveropts)
ax.quiver(*origin3D,*-B[:,1]*spanScalar,**quiveropts)
ax.quiver(*B[:,0]*spanScalar, *((B[:,0]+B[:,1])-B[:,0])*spanScalar,**quiveropts)
ax.quiver(*B[:,0]*spanScalar, *((B[:,0]+B[:,1])-B[:,0])*-spanScalar,**quiveropts)
ax.quiver(*-B[:,0]*spanScalar, *(-((B[:,0]+B[:,1])-B[:,0]))*spanScalar,**quiveropts)
ax.quiver(*-B[:,0]*spanScalar, *(-((B[:,0]+B[:,1])-B[:,0]))*-spanScalar,**quiveropts)
# Span visualization.

# Projections.
ax.quiver(*origin3D, *x, color='red', linewidths = 2,arrow_length_ratio = .05) # <-- Initial vector.
ax.quiver(*origin3D, *xTest1, color='pink', linewidths = 2,arrow_length_ratio = .05)
ax.quiver(*origin3D, *xTest2, color='pink', linewidths = 2,arrow_length_ratio = .05)

ax.quiver(*origin3D, *proj, color='pink', linewidths = .5,arrow_length_ratio = .05); # <-- Projected vector.
ax.quiver(*origin3D, *pTest1, color='pink', linewidths = .5,arrow_length_ratio = .05)
ax.quiver(*origin3D, *pTest2, color='pink', linewidths = .5,arrow_length_ratio = .05)

ax.quiver(*proj, *projDisplacement, color='pink', linewidths = .5,arrow_length_ratio = .05); # <-- Displacement.
ax.quiver(*pTest1, *displ1, color='pink', linewidths = .5,arrow_length_ratio = .05)
ax.quiver(*pTest2, *displ2, color='pink', linewidths = .5,arrow_length_ratio = .05);
    # Projections.
'''

# 3D Figure Formatting.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.grid(b=None)
scale3D = 5
ax.set_xlim3d(-scale3D/1,scale3D/1)
ax.set_ylim3d(scale3D/1,-scale3D/1)
ax.set_zlim3d(-scale3D/1,scale3D/1)
ax.set_title(r"Projection onto the span of bases $[1 , 1 , 1]^T [0 , 1 , 2]^T$, for vector $[6,0,0]^T$ (in red)." + "\n Aswell as other vectors.")
# 3D Figure Formatting.

# Span visualization.
quiveropts = dict(color='lightblue', linewidths = .5,arrow_length_ratio = .05, pivot='tail')
spanScalar = 6
ax.quiver(*origin3D,*B[:,0]*spanScalar,**quiveropts)
ax.quiver(*origin3D,*B[:,1]*spanScalar,**quiveropts)
ax.quiver(*origin3D,*-B[:,0]*spanScalar,**quiveropts)
ax.quiver(*origin3D,*-B[:,1]*spanScalar,**quiveropts)
ax.quiver(*B[:,0]*spanScalar, *((B[:,0]+B[:,1])-B[:,0])*spanScalar,**quiveropts)
ax.quiver(*B[:,0]*spanScalar, *((B[:,0]+B[:,1])-B[:,0])*-spanScalar,**quiveropts)
ax.quiver(*-B[:,0]*spanScalar, *(-((B[:,0]+B[:,1])-B[:,0]))*spanScalar,**quiveropts)
ax.quiver(*-B[:,0]*spanScalar, *(-((B[:,0]+B[:,1])-B[:,0]))*-spanScalar,**quiveropts)
# Span visualization.

# Projections.
ax.quiver(*origin3D, *x, color='red', linewidths = 2,arrow_length_ratio = .05) # <-- Initial vector.
ax.quiver(*origin3D, *xTest1, color='pink', linewidths = 2,arrow_length_ratio = .05)
ax.quiver(*origin3D, *xTest2, color='pink', linewidths = 2,arrow_length_ratio = .05)

ax.quiver(*origin3D, *proj, color='pink', linewidths = .5,arrow_length_ratio = .05); # <-- Projected vector.
ax.quiver(*origin3D, *pTest1, color='pink', linewidths = .5,arrow_length_ratio = .05)
ax.quiver(*origin3D, *pTest2, color='pink', linewidths = .5,arrow_length_ratio = .05)

ax.quiver(*proj, *projDisplacement, color='pink', linewidths = .5,arrow_length_ratio = .05); # <-- Displacement.
ax.quiver(*pTest1, *displ1, color='pink', linewidths = .5,arrow_length_ratio = .05)
ax.quiver(*pTest2, *displ2, color='pink', linewidths = .5,arrow_length_ratio = .05);
# Projections.

st.code(code,line_numbers=True)

st.pyplot(fig)

st.markdown("#### Figure 3.11")

st.markdown(r"*Projection onto a two-dimensional subspace $U$ with basis $b_1$, $b_2$. The projection $\pi_U(x)$ of $x \in \mathbb{R}^3$ onto $U$ can be expressed as a linear combination of $b_1$, $b_2$ and the displacement vector $x-\pi_U(x)$ is orthogonal to both $b_1$ and $b_2$.*")

code = '''
    angle(cosAngle(projDisplacement,B[:,0])) # 90.0
    angle(cosAngle(projDisplacement,B[:,1])) # 90.0
'''

st.code(code, line_numbers=True)

st.markdown("The norm of the displacement is sometimes called the projection error, and simply reprsents the distance from the original vector to it's projection.")

code = '''
    npl.norm(projDisplacement),np.sqrt(6) 
    # np.sqrt added for continuity with book.
'''

st.code(code, line_numbers=True)

code = f'''
    {npl.norm(projDisplacement)}, {np.sqrt(6)}
'''

st.code(code)

st.markdown(r"An important potential trick to note is that if the basis is orthonormal, that is, it is orthogonal and unitary, then $B^TB=I$, meaning that $\lambda = B^Tx$ and the projection $\pi_U(x) = BB^Tx$, which in practice on a problem wherein scale is a factor, could save computation time. This process leaves the span unaltered, that is the span of the original basis and the orthonormal basis are the same.")

st.markdown("### Example 3.12 (Gram-Schmidt Orthogonalization)")
st.markdown("In the event that we were given a bases and simply wanted to find the orthogonal version, we could do the following.In the event that we were given a bases and simply wanted to find the orthogonal version, we could do the following.")

st.markdown(r"1. Set the first orthogonal vector as the first basis vector. $u_1 = b_1$.")
st.markdown(r"2. Project the second basis onto the new vector. $p = \frac{u_1 u_1^T}{||u_1||^2}b_2$.")
st.markdown(r"3. Subtract the projection from the second basis, to find the second orthogonal vector. $u_2 = b_2 - p$.")

st.markdown("#### Figure 3.12")

code = '''
    b1 = np.vstack([2,0])
    b2 = np.vstack([1,1])

    u1 = b1
    p = ((u1@u1.T)/(npl.norm(u1)**2))@b2 # Projection step.
    u2 = b2-p
'''

st.code(code, line_numbers=True)

b1 = np.vstack([2,0])
b2 = np.vstack([1,1])

u1 = b1
p = ((u1@u1.T)/(npl.norm(u1)**2))@b2 # Projection step.
u2 = b2-p

code = f'''
    p: {list(p)}
    u2: {list(u2)}
'''

st.code(code)

st.markdown("Figure setup")

code = '''
    # Figure setup.
fig,a = plt.subplots(1,3, figsize=(15,5))
plt.setp(a, xlim=(-scale/2,scale/2), ylim=(-scale/2,scale/2))
a[0].set_title(r"(a) non-orthogonal basis $(b_1,b_2)$ of $\mathbb{R}^2$")
a[1].set_title(r"(b) first constructed basis vector $u_1$"+"\n"+r" and orthogonal projection of $b_2$ onto $span[u_1]$")
a[2].set_title(r"(c) orthogonal basis $(u_1, u_2)$ of $\mathbb{R}^2$.")
# Figure setup.


a[0].quiver(*origin2D, *b1,scale=scale, width = .005)
a[0].quiver(*origin2D, *b2,scale=scale, width = .005)
a[0].legend([r"$b_1$",r"$b_2$"])

a[1].quiver(*origin2D, *u1,scale=scale, width = .005, color = "b")
a[1].quiver(*origin2D, *b2,scale=scale, width = .005)
a[1].quiver(*origin2D, *p,scale=scale, width = .007, color = "orange")
a[1].quiver(*p, *b2-p, scale = scale, width= .002, color = "orange")
a[1].legend([r"$u_1$",r"$b_2$",r"$\pi b_2$","Displacement"])

a[2].quiver(*origin2D, *u1, scale = scale, width = .005, color = "b")
a[2].quiver(*origin2D, *b2, scale = scale, width = .005)
a[2].quiver(*origin2D, *p, scale = scale, width = .007, color = "orange")
a[2].quiver(*p, *b2-p, scale = scale, width= .002, color = "orange")
a[2].quiver(*b2,*-p, scale = scale, width= .005, color = "orange")
a[2].quiver(*origin2D, *b2-p, scale = scale, width= .007, color = "b")
a[2].legend([r"$u_1$",r"$b_2$",r"$\pi b_2$","Displacement",r"$b_2$ --> -p",r"$u_2$"]);
'''

st.code(code, line_numbers=True)

# Figure setup.
fig,a = plt.subplots(1,3, figsize=(15,5))
plt.setp(a, xlim=(-scale/2,scale/2), ylim=(-scale/2,scale/2))
a[0].set_title(r"(a) non-orthogonal basis $(b_1,b_2)$ of $\mathbb{R}^2$")
a[1].set_title(r"(b) first constructed basis vector $u_1$"+"\n"+r" and orthogonal projection of $b_2$ onto $span[u_1]$")
a[2].set_title(r"(c) orthogonal basis $(u_1, u_2)$ of $\mathbb{R}^2$.")
# Figure setup.


a[0].quiver(*origin2D, *b1,scale=scale, width = .005)
a[0].quiver(*origin2D, *b2,scale=scale, width = .005)
a[0].legend([r"$b_1$",r"$b_2$"])

a[1].quiver(*origin2D, *u1,scale=scale, width = .005, color = "b")
a[1].quiver(*origin2D, *b2,scale=scale, width = .005)
a[1].quiver(*origin2D, *p,scale=scale, width = .007, color = "orange")
a[1].quiver(*p, *b2-p, scale = scale, width= .002, color = "orange")
a[1].legend([r"$u_1$",r"$b_2$",r"$\pi b_2$","Displacement"])

a[2].quiver(*origin2D, *u1, scale = scale, width = .005, color = "b")
a[2].quiver(*origin2D, *b2, scale = scale, width = .005)
a[2].quiver(*origin2D, *p, scale = scale, width = .007, color = "orange")
a[2].quiver(*p, *b2-p, scale = scale, width= .002, color = "orange")
a[2].quiver(*b2,*-p, scale = scale, width= .005, color = "orange")
a[2].quiver(*origin2D, *b2-p, scale = scale, width= .007, color = "b")
a[2].legend([r"$u_1$",r"$b_2$",r"$\pi b_2$","Displacement",r"$b_2$ --> -p",r"$u_2$"]);

st.pyplot(fig)

st.markdown(r"From here we can see that the new bases, $u_1, u_2$, are orthogonal, although not orthonormal as $u_1$ is not unitary.")

st.markdown("### 3.8.4 Projection onto Affine Subspaces")

st.markdown(r"*Recall affine subspaces from Chapter 2. In $\mathbb{R}^2$, a line is defined by a support point $x_0$ and a vector $x_1$, with the equation $L = x_0 + \lambda x_1$. We can intuit if the value of $\lambda$ is 0 or 1, we have a point either at $x_0$ or $\lambda x_1$, our support point however, will always \"hold\" our line at a certain point.*")

st.markdown("Figure setup:")

code = '''
    # Figure setup.
scale = 10
origin = np.array([0,0])
plt.axis([-scale/2,scale/2,-scale/2,scale/2])
plt.title("An affine subspace L. A scaling of the black vector by some value lambda will \n move the orange vector parallel to the black, 'supported' by the purple.", size=15)
# Figure setup.

# Data setup.
lam = 1
x0 = np.array([-1,1])
x1 = np.array([3,.5])
val = x0+lam*x1
x1s = np.vstack([l*x1 for l in np.linspace(-4,4,25)]) # Grey lines
# Data setup.

# Plotting vectors.


plt.quiver(*origin, *x1, scale=scale, color="k")
plt.quiver(*origin, *x0, scale=scale, color="blueviolet")
plt.quiver(*origin, *val, scale=scale, color="orange") #L = x_0 + lambda * x_1 ; Affine subspace
# Plotting vectors.

# Plotting subspace.
# Arbitrary choice of 250 points on the subspace. Subspace is any infinite amount of points along the line.
val = np.vstack([x0+l*x1 for l in np.linspace(-4,4,250)])
plt.scatter(val[:,0],val[:,1],color="r", s = .5)
# Plotting subspace.
for idx in range(len(x1s)):
    plt.quiver(*origin, *x1s[idx], width=.003, scale=scale, color="grey")
plt.legend(["$x_1*\lambda$",r"$x_1$",r"$x_0$","y",r"$L = x_0 + \lambda x_1$"]);
'''


plt.clf()


# Figure setup.
scale = 10
origin = np.array([0,0])
plt.axis([-scale/2,scale/2,-scale/2,scale/2])
plt.title("An affine subspace L. A scaling of the black vector by some value lambda will \n move the orange vector parallel to the black, 'supported' by the purple.", size=15)
# Figure setup.

# Data setup.
lam = 1
x0 = np.array([-1,1])
x1 = np.array([3,.5])
val = x0+lam*x1
x1s = np.vstack([l*x1 for l in np.linspace(-4,4,25)]) # Grey lines
# Data setup.
print(x1s)
# Plotting vectors.


plt.quiver(*origin, *x1, scale=scale, color="k")
plt.quiver(*origin, *x0, scale=scale, color="blueviolet")
plt.quiver(*origin, *val, scale=scale, color="orange") #L = x_0 + lambda * x_1 ; Affine subspace
# Plotting vectors.

# Plotting subspace.
# Arbitrary choice of 250 points on the subspace. Subspace is any infinite amount of points along the line.
val = np.vstack([x0+l*x1 for l in np.linspace(-4,4,250)])
plt.scatter(val[:,0],val[:,1],color="r", s = .5)
# Plotting subspace.
for idx in range(len(x1s)):
    plt.quiver(*origin, *x1s[idx], width=.003, scale=scale, color="grey")
plt.legend(["$x_1*\lambda$",r"$x_1$",r"$x_0$","y",r"$L = x_0 + \lambda x_1$"])

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

st.code(code,line_numbers=True)


st.pyplot(plt)

st.markdown("Figure 3.13")
st.markdown(r"We replicate the above, this time in $\mathbb{R}^3$ and with projection onto a plane.")

code = '''
basis1 = np.vstack([1,0,0])
basis2 = np.vstack([0,1,0])
B = np.hstack([basis1,basis2])
support = np.vstack([-2,-1,3])
x = np.vstack([1,-3,4.5])
P = B@npl.inv(B.T@B)@B.T # Projection matrix onto basis.
proj = P@(x-support) # Projection of x onto basis.
displ = x - support - proj
'''

st.code(code,line_numbers=True)

basis1 = np.vstack([1,0,0])
basis2 = np.vstack([0,1,0])
B = np.hstack([basis1,basis2])
support = np.vstack([-2,-1,3])
x = np.vstack([1,-3,4.5])
P = B@npl.inv(B.T@B)@B.T # Projection matrix onto basis.
proj = P@(x-support) # Projection of x onto basis.
displ = x - support - proj


code = f'''
    P: {list(P)}
    proj: {list(proj)}
    displ: {list(displ)}
'''

st.code(code)


plt.rcParams[ "figure.figsize" ] = (12,12)

st.markdown("Figure setup:")

code = '''
def fig313(a = True, b = True, c = True):# 3D Figure Formatting.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    scale3D = 5
    ax.set_xlim3d(-scale3D/1,scale3D/1)
    ax.set_ylim3d(scale3D/1,-scale3D/1)
    ax.set_zlim3d(-scale3D/1,scale3D/1)
    ax.grid(b=None)
    ax.set_title(r"Figure 3.13. Affine Projection in $\mathbb{R}^3$.", size = 20)
    # 3D Figure Formatting.
    if (a):
        # Basis span visualization.
        spanScalar = 6
        ax.quiver(*origin3D,*B[:,0]*spanScalar, color='lightblue', linewidths = .5,arrow_length_ratio = .05)
        quiveropts = dict(color='lightblue', linewidths = .5,arrow_length_ratio = .05, label='_nolegend_')
        ax.quiver(*origin3D,*B[:,1]*spanScalar,**quiveropts)
        ax.quiver(*origin3D,*-B[:,0]*spanScalar,**quiveropts)
        ax.quiver(*origin3D,*-B[:,1]*spanScalar,**quiveropts)
        ax.quiver(*B[:,0]*spanScalar, *((B[:,0]+B[:,1])-B[:,0])*spanScalar,**quiveropts)
        ax.quiver(*B[:,0]*spanScalar, *((B[:,0]+B[:,1])-B[:,0])*-spanScalar,**quiveropts)
        ax.quiver(*-B[:,0]*spanScalar, *(-((B[:,0]+B[:,1])-B[:,0]))*spanScalar,**quiveropts)
        ax.quiver(*-B[:,0]*spanScalar, *(-((B[:,0]+B[:,1])-B[:,0]))*-spanScalar,**quiveropts)
        # Basis span visualization.

        # L span visualization.
        ax.quiver(*origin3D,*support,color='purple', linewidths = 1.5,arrow_length_ratio = .05)
        ax.quiver(*support, *B[:,0]*spanScalar, color='grey', linewidths = .75,arrow_length_ratio = .05)
        quiveropts = dict(color='grey', linewidths = .75,arrow_length_ratio = .05, label='_nolegend_')
        ax.quiver(*support, *B[:,1]*spanScalar,**quiveropts)
        ax.quiver(*support, *-B[:,0]*spanScalar,**quiveropts)
        ax.quiver(*support, *-B[:,1]*spanScalar,**quiveropts)
        ax.quiver(*support+np.vstack(B[:,0]*spanScalar), *-B[:,1]*spanScalar, **quiveropts)
        ax.quiver(*support+np.vstack(B[:,0]*spanScalar), *B[:,1]*spanScalar, **quiveropts)
        ax.quiver(*support+np.vstack(-B[:,0]*spanScalar),*-B[:,1]*spanScalar, **quiveropts)
        ax.quiver(*support+np.vstack(-B[:,0]*spanScalar),*B[:,1]*spanScalar, **quiveropts)
        # L span visualization.s

    ax.quiver(*origin3D, *x, color='orange', linewidths = 1.5,arrow_length_ratio = .05) # Initial vector.

    if (b):
    # Projections.
        ax.quiver(*origin3D, *x-support, color='red', linewidths = 1.5,arrow_length_ratio = .05)
        ax.quiver(*origin3D, *proj, color='pink', linewidths = .75,arrow_length_ratio = .05)
        ax.quiver(*proj, *displ, color='pink', linewidths = .75,arrow_length_ratio = .05, label='_nolegend_')
    ## "Add support point back in"
    if (c):
        ax.quiver(*proj, *support, color='purple', linewidths = .75,arrow_length_ratio = .05,label='_nolegend_')
        ax.quiver(*origin3D+support, *proj, color='pink', linewidths = .75,arrow_length_ratio = .05, label='_nolegend_')
        ax.quiver(*x, *-displ, color='pink', linewidths = .75,arrow_length_ratio = .05, label='_nolegend_')
        ax.quiver(*support, *x-support, color='pink', linewidths = .75,arrow_length_ratio = .05, label='_nolegend_')
    # Projections.

    ax.legend(["Basis","'Support'","Linear Manifold, L","x","x-Support","projection"]);
'''

st.code(code, line_numbers=True)

def fig313(a = True, b = True, c = True):# 3D Figure Formatting.
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    scale3D = 5
    ax.set_xlim3d(-scale3D/1,scale3D/1)
    ax.set_ylim3d(scale3D/1,-scale3D/1)
    ax.set_zlim3d(-scale3D/1,scale3D/1)
    ax.grid(b=None)
    ax.set_title(r"Figure 3.13. Affine Projection in $\mathbb{R}^3$.", size = 20)
    # 3D Figure Formatting.
    if (a):
        # Basis span visualization.
        spanScalar = 6
        ax.quiver(*origin3D,*B[:,0]*spanScalar, color='lightblue', linewidths = .5,arrow_length_ratio = .05)
        quiveropts = dict(color='lightblue', linewidths = .5,arrow_length_ratio = .05, label='_nolegend_')
        ax.quiver(*origin3D,*B[:,1]*spanScalar,**quiveropts)
        ax.quiver(*origin3D,*-B[:,0]*spanScalar,**quiveropts)
        ax.quiver(*origin3D,*-B[:,1]*spanScalar,**quiveropts)
        ax.quiver(*B[:,0]*spanScalar, *((B[:,0]+B[:,1])-B[:,0])*spanScalar,**quiveropts)
        ax.quiver(*B[:,0]*spanScalar, *((B[:,0]+B[:,1])-B[:,0])*-spanScalar,**quiveropts)
        ax.quiver(*-B[:,0]*spanScalar, *(-((B[:,0]+B[:,1])-B[:,0]))*spanScalar,**quiveropts)
        ax.quiver(*-B[:,0]*spanScalar, *(-((B[:,0]+B[:,1])-B[:,0]))*-spanScalar,**quiveropts)
        # Basis span visualization.

        # L span visualization.
        ax.quiver(*origin3D,*support,color='purple', linewidths = 1.5,arrow_length_ratio = .05)
        ax.quiver(*support, *B[:,0]*spanScalar, color='grey', linewidths = .75,arrow_length_ratio = .05)
        quiveropts = dict(color='grey', linewidths = .75,arrow_length_ratio = .05, label='_nolegend_')
        ax.quiver(*support, *B[:,1]*spanScalar,**quiveropts)
        ax.quiver(*support, *-B[:,0]*spanScalar,**quiveropts)
        ax.quiver(*support, *-B[:,1]*spanScalar,**quiveropts)
        ax.quiver(*support+np.vstack(B[:,0]*spanScalar), *-B[:,1]*spanScalar, **quiveropts)
        ax.quiver(*support+np.vstack(B[:,0]*spanScalar), *B[:,1]*spanScalar, **quiveropts)
        ax.quiver(*support+np.vstack(-B[:,0]*spanScalar),*-B[:,1]*spanScalar, **quiveropts)
        ax.quiver(*support+np.vstack(-B[:,0]*spanScalar),*B[:,1]*spanScalar, **quiveropts)
        # L span visualization.s

    ax.quiver(*origin3D, *x, color='orange', linewidths = 1.5,arrow_length_ratio = .05) # Initial vector.

    if (b):
    # Projections.
        ax.quiver(*origin3D, *x-support, color='red', linewidths = 1.5,arrow_length_ratio = .05)
        ax.quiver(*origin3D, *proj, color='pink', linewidths = .75,arrow_length_ratio = .05)
        ax.quiver(*proj, *displ, color='pink', linewidths = .75,arrow_length_ratio = .05, label='_nolegend_')
    ## "Add support point back in"
    if (c):
        ax.quiver(*proj, *support, color='purple', linewidths = .75,arrow_length_ratio = .05,label='_nolegend_')
        ax.quiver(*origin3D+support, *proj, color='pink', linewidths = .75,arrow_length_ratio = .05, label='_nolegend_')
        ax.quiver(*x, *-displ, color='pink', linewidths = .75,arrow_length_ratio = .05, label='_nolegend_')
        ax.quiver(*support, *x-support, color='pink', linewidths = .75,arrow_length_ratio = .05, label='_nolegend_')
    # Projections.

    ax.legend(["Basis","'Support'","Linear Manifold, L","x","x-Support","projection"])

    st.pyplot(fig)

fig313(b=False,c=False)

st.markdown(r"$(a)$ Original setting: we're given a basis, a support, an affine subspace created by the support, and a vector x.")
fig313(c=False)

st.markdown(r"$(b)$ Reduce problem to projection $\pi_U$ onto basis. Original setting shifted by `-support` so that `x-support` can be projected onto the direction space $U$, our basis.")

fig313()

st.markdown(r"$(c)$ Add support point back in to get affine projection $\pi_L$. Projection is translated back to `support`$+\pi_U$ `(x-support)`, which gives the final orthogonal projection $\pi_L(x)$.")
