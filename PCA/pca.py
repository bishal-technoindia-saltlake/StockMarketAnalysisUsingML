import numpy as np


class PCA:

    def __init__(self, n_component):
        self.n_component = n_component
        self.component = None
        self.mean = None

    def fit(self, X):
        # Mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Covariance
        cov = np.cov(X.T)

        # EigenVectors, EigenValues
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort EigenVectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # Store first n_components' EigenVectors
        self.component = eigenvectors[0: self.n_component]

    def transform(self, X):
        # Project data
        X = X-self.mean
        return np.dot(X, self.component.T)
