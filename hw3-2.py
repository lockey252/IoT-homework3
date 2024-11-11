import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D

# Streamlit app title
st.title("3D Scatter Plot with Separating Hyperplane")

# Generate 3D dataset with Gaussian distribution
np.random.seed(42)
num_points = 600
X1 = np.random.normal(0, 1, num_points) * 5  # Centered around 0 with spread
X2 = np.random.normal(0, 1, num_points) * 5
X3 = np.exp(-(X1**2 + X2**2) / 50)  # Non-linear height feature to create a peak shape
distance_threshold = st.slider('Distance Threshold', min_value=0.1, max_value=10.0, value=5.0, step=0.1)
Y = np.where(np.sqrt(X1**2 + X2**2) < distance_threshold, 0, 1)  # Circular decision boundary

# Ensure there are at least two classes
if len(np.unique(Y)) > 1:
    X = np.vstack((X1, X2)).T  # Use X1 and X2 as features for SVM training

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
    ax.set_title('3D Scatter Plot with Y Color and Separating Hyperplane')
    ax.legend()

    # Display plot using Streamlit
    st.pyplot(fig)
else:
    st.warning("The data only has one class. Adjust the distance threshold to generate at least two classes.")
