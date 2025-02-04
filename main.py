import numpy as np
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler

class CanonicalCorrelationAnalysis:
    """
    A class for performing Canonical Correlation Analysis (CCA).

    Canonical Correlation Analysis is a multivariate statistical method used 
    to identify relationships between two sets of variables by finding linear 
    combinations that are maximally correlated.

    Attributes:
        n_components (int): Number of canonical components to compute.
        x_weights_ (np.ndarray or None): Weights for the first set of variables (X), learned during fitting.
        y_weights_ (np.ndarray or None): Weights for the second set of variables (Y), learned during fitting.
        canonical_correlations_ (np.ndarray or None): Canonical correlation coefficients after fitting.

    Methods:
        fit(X, Y): Fits the CCA model to the given datasets.
        transform(X, Y): Transforms input data into canonical components.
    """
    def __init__(self, n_components: int = 2):
        """Initializes the Canonical Correlation Analysis model."""
        self.n_components = n_components
        self.x_weights_ = None
        self.y_weights_ = None
        self.canonical_correlations_ = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Fits the CCA model to the given datasets X and Y."""
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples.")

        X = StandardScaler().fit_transform(X)
        Y = StandardScaler().fit_transform(Y)

        Sxx = np.cov(X, rowvar=False)
        Syy = np.cov(Y, rowvar=False)
        Sxy = np.cov(X.T, Y.T)[:X.shape[1], X.shape[1]:]

        U, s, Vt = svd(np.linalg.inv(Sxx) @ Sxy @ np.linalg.inv(Syy) @ Sxy.T)
        self.canonical_correlations_ = np.sqrt(s[:self.n_components])
        self.x_weights_ = U[:, :self.n_components]
        self.y_weights_ = np.linalg.inv(Syy) @ Sxy.T @ self.x_weights_

        return self

    def transform(self, X: np.ndarray, Y: np.ndarray):
        """Transforms X and Y into their canonical variables."""
        if self.x_weights_ is None or self.y_weights_ is None:
            raise RuntimeError("The model must be fitted before transformation.")

        X = StandardScaler().fit_transform(X) @ self.x_weights_
        Y = StandardScaler().fit_transform(Y) @ self.y_weights_

        return X, Y

    def fit_transform(self, X: np.ndarray, Y: np.ndarray):
        """Fits the model and transforms the datasets into canonical variables."""
        self.fit(X, Y)
        return self.transform(X, Y)

    def summary(self):
        """Prints a summary of the canonical correlations and weight matrices."""
        print("Canonical Correlations:")
        print(self.canonical_correlations_)
        print("\nX Weights:")
        print(self.x_weights_)
        print("\nY Weights:")
        print(self.y_weights_)

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 5)
    Y = np.random.rand(100, 4)

    cca = CanonicalCorrelationAnalysis(n_components=2)
    X_c, Y_c = cca.fit_transform(X, Y)
    cca.summary()
