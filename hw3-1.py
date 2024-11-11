import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Step 1: Generate 300 random variables X(i) in the range of 0 to 1000
np.random.seed(42)
X = np.random.uniform(0, 1000, 300).reshape(-1, 1)

# Step 2: Determine Y(i) based on the condition Y(i)=1 if 500 < X(i) < 800, otherwise Y(i)=0
Y = np.where((X > 500) & (X < 800), 1, 0).ravel()

# Step 3: Implement a logistic regression model to predict the outcomes (y1)
log_reg = LogisticRegression()
log_reg.fit(X, Y)
Y1 = log_reg.predict(X)

# Step 4: Implement a support vector machine model to predict the outcomes (y2)
svm = SVC(kernel='linear')
svm.fit(X, Y)
Y2 = svm.predict(X)

# Plotting
plt.figure(figsize=(12, 6))

# Plot for Logistic Regression
plt.subplot(1, 2, 1)
plt.scatter(X, Y, color='blue', alpha=0.5, label='True')
plt.scatter(X, Y1, color='green', alpha=0.5, marker='x', label='Logistic Regression Prediction')
plt.axvline(x=500, color='green', linestyle='--', label='Decision Boundary 1 (500)')
plt.axvline(x=800, color='green', linestyle='--', label='Decision Boundary 2 (800)')
plt.title('Logistic Regression Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid()

# Plot for SVM
plt.subplot(1, 2, 2)
plt.scatter(X, Y, color='blue', alpha=0.5, label='True')
plt.scatter(X, Y2, color='red', alpha=0.5, marker='s', label='SVM Prediction')
plt.axvline(x=500, color='red', linestyle='--', label='Decision Boundary 1 (500)')
plt.axvline(x=800, color='red', linestyle='--', label='Decision Boundary 2 (800)')
plt.title('SVM Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
