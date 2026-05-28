import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0, method="closed_form", lr=0.01, epochs=1000):
        """
        alpha  -> regularization parameter
        method -> "closed_form" or "gradient_descent"
        lr     -> learning rate (for GD)
        epochs -> iterations (for GD)
        """
        self.alpha = alpha
        self.method = method
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    # -------------------------------
    # FIT FUNCTION
    # -------------------------------
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        # Add bias column (for intercept)
        X_bias = np.c_[np.ones((n_samples, 1)), X]

        if self.method == "closed_form":
            self._fit_closed_form(X_bias, y, n_features)

        elif self.method == "gradient_descent":
            self._fit_gradient_descent(X_bias, y, n_features)

    # -------------------------------
    # CLOSED FORM SOLUTION
    # -------------------------------
    def _fit_closed_form(self, X, y, n_features):
        I = np.eye(X.shape[1])
        I[0, 0] = 0  # Don't regularize bias

        # Formula: (X^T X + αI)^(-1) X^T y
        self.weights = np.linalg.inv(X.T @ X + self.alpha * I) @ X.T @ y

    # -------------------------------
    # GRADIENT DESCENT SOLUTION
    # -------------------------------
    def _fit_gradient_descent(self, X, y, n_features):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for epoch in range(self.epochs):
            y_pred = X @ self.weights

            # Gradient
            gradient = (-2 / n_samples) * (X.T @ (y - y_pred)) \
                       + 2 * self.alpha * np.r_[0, self.weights[1:]]

            # Update
            self.weights -= self.lr * gradient

    # -------------------------------
    # PREDICT FUNCTION
    # -------------------------------
    def predict(self, X):
        X = np.array(X)
        n_samples = X.shape[0]

        X_bias = np.c_[np.ones((n_samples, 1)), X]
        return X_bias @ self.weights


# =====================================================
# 🔷 Example Usage
# =====================================================

if __name__ == "__main__":
    # Sample dataset
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 5, 4, 5])

    # Create model
    model = RidgeRegression(alpha=1.0, method="closed_form")

    # Train
    model.fit(X, y)

    # Predict
    predictions = model.predict(X)

    print("Weights (including bias):", model.weights)
    print("Predictions:", predictions)

    # -------------------------------
    # Using Gradient Descent
    # -------------------------------
    model_gd = RidgeRegression(alpha=1.0, method="gradient_descent", lr=0.01, epochs=5000)
    model_gd.fit(X, y)

    predictions_gd = model_gd.predict(X)

    print("\n[Gradient Descent]")
    print("Weights:", model_gd.weights)
    print("Predictions:", predictions_gd)