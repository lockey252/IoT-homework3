import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D

# Streamlit app title
st.title("3D Scatter Plot with Star-Shaped Data Distribution")

# Generate 2D dataset with a star-like distribution
np.random.seed(42)
num_points = 600
angles = np.linspace(0, 2 * np.pi, num_points)
radius = 5 + 2 * np.sin(5 * angles)  # Creating a star shape by varying the radius
X1 = radius * np.cos(angles) + np.random.normal(0, 0.5, num_points)
X2 = radius * np.sin(angles) + np.random.normal(0, 0.5, num_points)
X3 = np.exp(-(X1**2 + X2**2) / 100)  # Non-linear height feature to create a 3D effect

# Define a threshold for classification (separating inside and outside regions)
threshold_radius = st.slider('Threshold Radius', min_value=3.0, max_value=10.0, value=6.0, step=0.1)
Y = np.where(np.sqrt(X1**2 + X2**2) < threshold_radius, 0, 1)  # Classification rule

X = np.vstack((X1, X2)).T  # Use X1 and X2 as features for SVM training

# Ensure there are at least two classes
if len(np.unique(Y)) > 1:
    # Train SVM model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X, Y)

    # Create grid to plot decision boundary
    xx, yy = np.meshgrid(np.linspace(X1.min() - 1, X1.max() + 1, 50), np.linspace(X2.min() - 1, X2.max() + 1, 50))
    Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 3D Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of data points
    ax.scatter(X1[Y == 0], X2[Y == 0], X3[Y == 0], color='blue', label='Y=0', alpha=0.7)
    ax.scatter(X1[Y == 1], X2[Y == 1], X3[Y == 1], color='red', label='Y=1', alpha=0.7)

    # Plot decision boundary plane (simplified visualization)
    ax.plot_surface(xx, yy, Z * 0, color='gray', alpha=0.3, edgecolor='none')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title('3D Scatter Plot with Star-Shaped Data Distribution')
    ax.legend()

    # Display plot using Streamlit
    st.pyplot(fig)
else:
    st.warning("The data only has one class. Adjust the threshold radius to generate at least two classes.")
