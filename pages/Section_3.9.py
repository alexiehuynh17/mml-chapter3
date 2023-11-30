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
scale = 8

st.set_page_config(page_title = "3.9 Rotations")

st.header("Rotations")

st.markdown("Figure 3.14:")

code = '''
plt.grid(visible=True)
plt.axis([-2.5,2.5,-2.5,2.5])
colors = np.vstack([[i,0,j] for i in np.linspace(0,1,20) for j in np.linspace(1,0,20)])
plt.title("Figure 3.14 A rotation rotates objects in a plane about the origin. If the rotation angle is positive, we rotate counterclockwise.", size = 15)

R = np.vstack([[-.38,-.92],[.92,-.38]]) # Rotation matrix.
original = np.vstack([[i,j] for i in np.linspace(.25,1,20) for j in np.linspace(.25,1,20)]) # Original coordinates.
rotated = np.dot(original,R.T) # Rotated coordinates. # Rotation step.

plt.scatter(original[:,0],original[:,1], c = colors)
plt.scatter(rotated[:,0],rotated[:,1], c = colors)
'''

st.code(code, line_numbers=True)

st.markdown("Rotation matrix")

st.latex(r'''
    \begin{bmatrix}
                -0.38 & -0.92\\
                0.92 & -0.38
                \end{bmatrix}
''')

plt.grid(visible=True)
plt.axis([-2.5,2.5,-2.5,2.5])
colors = np.vstack([[i,0,j] for i in np.linspace(0,1,20) for j in np.linspace(1,0,20)])
plt.title("Figure 3.14 A rotation rotates objects in a plane about the origin. \n If the rotation angle is positive, we rotate counterclockwise.", size = 15)

R = np.vstack([[-.38,-.92],[.92,-.38]]) # Rotation matrix.
original = np.vstack([[i,j] for i in np.linspace(.25,1,20) for j in np.linspace(.25,1,20)]) # Original coordinates.
rotated = np.dot(original,R.T) # Rotated coordinates. # Rotation step.

plt.scatter(original[:,0],original[:,1], c = colors)
plt.scatter(rotated[:,0],rotated[:,1], c = colors)

st.pyplot(plt)

st.markdown("Figure 3.16:")
st.markdown(r"*Rotation of the standard basis in $\mathbb{R}^2$ by an angle $\theta$.*")

code = '''
    e1 = np.vstack([1,0])
    e2 = np.vstack([0,1])
    someVec = np.vstack([3,1])

    plt.quiver(*origin2D, *e1, scale = scale, width = .002)
    plt.quiver(*origin2D, *e2, scale = scale, width = .002)
    plt.quiver(*origin2D, *someVec, scale = scale, width = .002)
    plt.axis([-scale/2,scale/2,-scale/2,scale/2])
    plt.title("Given a basis, we are looking to perform a basis change, via a rotation matrix.", size = 15);
'''

st.code(code,line_numbers=True)

e1 = np.vstack([1,0])
e2 = np.vstack([0,1])
someVec = np.vstack([3,1])

plt.clf()
plt.quiver(*origin2D, *e1, scale = scale, width = .002)
plt.quiver(*origin2D, *e2, scale = scale, width = .002)
plt.quiver(*origin2D, *someVec, scale = scale, width = .002)
plt.axis([-scale/2,scale/2,-scale/2,scale/2])
plt.title("Given a basis, we are looking to perform a basis change, via a rotation matrix.", size = 15);

st.pyplot(plt)

st.markdown("Trigonometric identities for sine and cosine work conveniently for rotations.")

code = '''
    E = np.hstack([e1,e2])
    phiE1 = lambda theta: np.vstack([np.cos(theta),np.sin(theta)])
    phiE2 = lambda theta: np.vstack([-np.sin(theta),np.cos(theta)])
    Phi = lambda theta: np.hstack([phiE1(theta), phiE2(theta)]) # Constructed transformation matrix.
    R = Phi(np.pi/4)

    plt.quiver(*origin2D, *e1, scale = scale, width = .002)
    plt.quiver(*origin2D, *e2, scale = scale, width = .002, label = '_nolegend_')
    plt.quiver(*origin2D, *E@R[:,0], scale = scale, width = .002, color = "b")
    plt.quiver(*origin2D, *E@R[:,1], scale = scale, width = .002, color = "b", label = '_nolegend_')
    plt.quiver(*origin2D, *someVec, scale = scale, width = .002)
    plt.quiver(*origin2D, *np.dot(someVec.flatten(),R.T), scale = scale, width = .002, color = "b")
    plt.axis([-scale/2,scale/2,-scale/2,scale/2])
    plt.legend([r"$e_{1,2}$",r"$\phi_{1,2}$"])
'''

st.code(code, line_numbers=True)

E = np.hstack([e1,e2])
phiE1 = lambda theta: np.vstack([np.cos(theta),np.sin(theta)])
phiE2 = lambda theta: np.vstack([-np.sin(theta),np.cos(theta)])
Phi = lambda theta: np.hstack([phiE1(theta), phiE2(theta)]) # Constructed transformation matrix.
R = Phi(np.pi/4)

plt.clf()
plt.quiver(*origin2D, *e1, scale = scale, width = .002)
plt.quiver(*origin2D, *e2, scale = scale, width = .002, label = '_nolegend_')
plt.quiver(*origin2D, *E@R[:,0], scale = scale, width = .002, color = "b")
plt.quiver(*origin2D, *E@R[:,1], scale = scale, width = .002, color = "b", label = '_nolegend_')
plt.quiver(*origin2D, *someVec, scale = scale, width = .002)
plt.quiver(*origin2D, *np.dot(someVec.flatten(),R.T), scale = scale, width = .002, color = "b")
plt.axis([-scale/2,scale/2,-scale/2,scale/2])
plt.legend([r"$e_{1,2}$",r"$\phi_{1,2}$"])

st.pyplot(plt)

st.markdown(r"Rotations in $\mathbb{R}^3$ can be thought of as rotations of the images of the standard bases of $\mathbb{R}^3$, so long as the images are orthonormal to each other, and can be implemented with a general rotation matrix by combining the images of the standard basis.")
st.markdown("The above transformations can be surmised as the following rotation matrix.")

st.latex(r'''\begin{bmatrix}\cos \theta & \sin \theta \\-\sin \theta & \cos \theta\end{bmatrix}''')

st.markdown(r"In $\mathbb{R}^3$, rotation about the x,y,z axes is of the following respective formats.")

st.latex(r'''
    \begin{bmatrix}
1 & 0 & 0 \\
0 & \cos \theta & -\sin \theta \\
0 & \sin \theta & \cos \theta
\end{bmatrix},
\begin{bmatrix}
\cos \theta & 0 & \sin \theta \\
0 & 1 & 0 \\
-\sin \theta & 0 & \cos \theta
\end{bmatrix},
\begin{bmatrix}
\cos \theta & -\sin \theta & 0  \\
\sin \theta & \cos \theta & 0 \\
0 & 0 & 1
\end{bmatrix}
''')

st.markdown("Z rotation:")

code = '''
    zRot = lambda theta: np.vstack([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta),np.cos(theta), 0], 
                                    [0, 0, 1]])
    R = zRot(np.pi/4)
    #   Matrix([[0.707106781186548, -0.707106781186547, 0], 
    #           [0.707106781186547, 0.707106781186548, 0], 
    #           [0, 0, 1.00000000000000]])

    B3 = np.eye(3)
    # Matrix([[1,0,0],
    #         [0,1,0],
    #         [0,0,1]])
'''

st.code(code,line_numbers=True)

zRot = lambda theta: np.vstack([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta),np.cos(theta), 0], 
                                [0, 0, 1]])
R = zRot(np.pi/4)
B3 = np.eye(3)

st.markdown("Rotate matrix B")

code = '''
    rotatedB = np.dot(B3,R.T)
    #   Matrix([[0.707106781186548, -0.707106781186547, 0], 
    #           [0.707106781186547, 0.707106781186548, 0], 
    #           [0, 0, 1.00000000000000]])
'''

st.code(code, line_numbers=True)

st.markdown("Rotate some vector:")

code = '''
    someVec = np.vstack([-2,1,3])

    rotatedVec = np.dot(someVec.T,R).T
    # [−0.707106781186548, 2.12132034355964, 3.0]
'''

rotatedB = np.dot(B3,R.T)
someVec = np.vstack([-2,1,3])
rotatedVec = np.dot(someVec.T,R).T


st.code(code, line_numbers=True)

st.markdown("Projection:")

code = '''
    basis1 = np.vstack([1,0,0])
    basis2 = np.vstack([0,1,0])
    B = np.hstack([basis1,basis2])

    P = B@npl.inv(B.T@B)@B.T # Projection matrix onto basis.
    proj = P@(someVec) # Projection of x onto basis.
    displ = someVec - proj

    newPlane = np.hstack([np.vstack(rotatedB[:,0]),np.vstack(rotatedB[:,1])]) # Span of rotated basis to project onto.
    P1 = newPlane@npl.inv(newPlane.T@newPlane)@newPlane.T # Projection matrix onto basis.
    proj1 = P1@(rotatedVec) # Projection of x onto basis.
    displ1 = rotatedVec - proj1
'''
basis1 = np.vstack([1,0,0])
basis2 = np.vstack([0,1,0])
B = np.hstack([basis1,basis2])

P = B@npl.inv(B.T@B)@B.T # Projection matrix onto basis.
proj = P@(someVec) # Projection of x onto basis.
displ = someVec - proj

newPlane = np.hstack([np.vstack(rotatedB[:,0]),np.vstack(rotatedB[:,1])]) # Span of rotated basis to project onto.
P1 = newPlane@npl.inv(newPlane.T@newPlane)@newPlane.T # Projection matrix onto basis.
proj1 = P1@(rotatedVec) # Projection of x onto basis.
displ1 = rotatedVec - proj1


st.code(code,line_numbers=True)

st.markdown("Figure setup:")

code = '''
    # Figure setup.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    scale3D = 5
    ax.set_xlim3d(-scale3D,scale3D)
    ax.set_ylim3d(scale3D,-scale3D)
    ax.set_zlim3d(-scale3D,scale3D)
    ax.grid(b=None)
    ax.set_title(r"Figure 3.17 Rotation in $\mathbb{R}^3$.", size = 15);
    # Figure setup.

    # Bases.
    quiveropts = dict(color='lightblue', linewidths = .8, arrow_length_ratio = .05, label = '_nolegend_')
    ax.quiver(*origin3D, *B3[:,0], color='lightblue', linewidths = .8, arrow_length_ratio = .05)
    ax.quiver(*origin3D, *B3[:,1], **quiveropts)
    ax.quiver(*origin3D, *B3[:,2], **quiveropts)
    # Bases.

    # Rotations.
    quiveropts = dict(color='gold', linewidths = .8, arrow_length_ratio = .05, label = '_nolegend_')
    ax.quiver(*origin3D, *rotatedB[:,0], color='gold', linewidths = .8, arrow_length_ratio = .05) # Change of basis.
    ax.quiver(*origin3D, *rotatedB[:,1], **quiveropts)
    ax.quiver(*origin3D, *rotatedB[:,2], **quiveropts)
    ax.quiver(*origin3D, *someVec, color='grey', linewidths = 1, arrow_length_ratio = .05) # Some vector.
    ax.quiver(*origin3D, *rotatedVec, color = 'b', linewidths = 1, arrow_length_ratio = .05) # Rotated vector.
    # Rotations.

    # Projections.
    quiveropts = dict(color='grey', linewidths = 1, arrow_length_ratio = .05, label = '_nolegend_')
    ax.quiver(*origin3D, *proj, **quiveropts)
    ax.quiver(*proj, *displ, **quiveropts)
    ax.quiver(*origin3D, *proj1, color='blue', linewidths = .5, arrow_length_ratio = .05)
    ax.quiver(*proj1, *displ1, color='blue', linewidths = .5, arrow_length_ratio = .05)
    # Projections.

    plt.legend([r"Basis $e_{1,2,3}$","Change of Basis","Some Vector","Rotated Vector"]);
'''

st.code(code,line_numbers=True)
# Figure setup.


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scale3D = 5
ax.set_xlim3d(-scale3D,scale3D)
ax.set_ylim3d(scale3D,-scale3D)
ax.set_zlim3d(-scale3D,scale3D)
ax.grid(b=None)
ax.set_title(r"Figure 3.17 Rotation in $\mathbb{R}^3$.", size = 15);
# Figure setup.

# Bases.
quiveropts = dict(color='lightblue', linewidths = .8, arrow_length_ratio = .05, label = '_nolegend_')
ax.quiver(*origin3D, *B3[:,0], color='lightblue', linewidths = .8, arrow_length_ratio = .05)
ax.quiver(*origin3D, *B3[:,1], **quiveropts)
ax.quiver(*origin3D, *B3[:,2], **quiveropts)
# Bases.

# Rotations.
quiveropts = dict(color='gold', linewidths = .8, arrow_length_ratio = .05, label = '_nolegend_')
ax.quiver(*origin3D, *rotatedB[:,0], color='gold', linewidths = .8, arrow_length_ratio = .05) # Change of basis.
ax.quiver(*origin3D, *rotatedB[:,1], **quiveropts)
ax.quiver(*origin3D, *rotatedB[:,2], **quiveropts)
ax.quiver(*origin3D, *someVec, color='grey', linewidths = 1, arrow_length_ratio = .05) # Some vector.
ax.quiver(*origin3D, *rotatedVec, color = 'b', linewidths = 1, arrow_length_ratio = .05) # Rotated vector.
# Rotations.

# Projections.
quiveropts = dict(color='grey', linewidths = 1, arrow_length_ratio = .05, label = '_nolegend_')
ax.quiver(*origin3D, *proj, **quiveropts)
ax.quiver(*proj, *displ, **quiveropts)
ax.quiver(*origin3D, *proj1, color='blue', linewidths = .5, arrow_length_ratio = .05)
ax.quiver(*proj1, *displ1, color='blue', linewidths = .5, arrow_length_ratio = .05)
# Projections.

plt.legend([r"Basis $e_{1,2,3}$","Change of Basis","Some Vector","Rotated Vector"]);

st.pyplot(fig)