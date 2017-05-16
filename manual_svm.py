import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split


class SVM:

    def __init__(self, max_iterations=1000, C=1, epsilon=0.001):
        self.max_iterations = max_iterations
        self.C = C
        self.epsilon = epsilon

    def fit(self, X, y):
        # Ensure X and y are numpy arrays
        X = np.array(X).astype('float')
        y = np.array(y).astype('int')

        # Initialize variables
        n, d = X.shape
        alpha = np.zeros(n)
        iterations = 0

        # Calculate starting w & b values
        self.compute_w_b(y, alpha, X)

        for iteration in range(1, self.max_iterations + 1):
            # print('Iteration ', iteration)
            alpha_prev = np.copy(alpha)

            for i in range(0, n):                
                # Get random j so that i != j
                j = random.randint(0, n - 2)
                if j >= i:
                    j += 1

                # Prepare variables
                x_i, y_i = X[i,:], y[i]
                x_j, y_j = X[j,:], y[j]

                # Calculate nu
                nu = np.dot(x_i, x_i) + np.dot(x_j, x_j) - np.dot(x_i, x_j)

                # Calculate lower and upper bounds
                L, H = self.L_H(alpha[i], alpha[j], y_i*y_j, self.C)

                if L == H:
                    continue

                # Compute E values
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)

                if nu == 0:
                    continue
                    """
                    if self.obj_func(nu, alpha[j], L, y_i, E_i, E_j) > self.obj_func(nu, alpha[j], H, y_i, E_i, E_j):
                        new_alpha_j = L
                    else:
                        new_alpha_j = H
                    """
                else:
                    # Compute E values
                    E_i = self.E(x_i, y_i, self.w, self.b)
                    E_j = self.E(x_j, y_j, self.w, self.b)

                    # Compute new alpha j
                    new_alpha_j = alpha[j] + y_j*(E_i - E_j)/nu
                    new_alpha_j = max(min(new_alpha_j, H), L)

                # Compute new alpha i & deltas
                new_alpha_i = alpha[i] + y_i*y_j*(alpha[j] - new_alpha_j)
                delta_i = (new_alpha_i - alpha[i])
                delta_j = (new_alpha_j - alpha[j])

                # Update w
                self.w += delta_i*y_i*x_i + delta_j*y_j*x_j

                # Update b
                b_i = self.b - E_i - y_i*delta_i*np.dot(x_i, x_i) - y_j*delta_j*np.dot(x_i, x_j)
                b_j = self.b - E_i - y_i*delta_i*np.dot(x_i, x_i) - y_j*delta_j*np.dot(x_i, x_j)

                if 0 < new_alpha_i < self.C:
                    self.b = b_i
                elif 0 < new_alpha_j < self.C:
                    self.b = b_j
                else:
                    self.b = (b_i + b_j)/2

                # Update alphas
                alpha[i] = new_alpha_i
                alpha[j] = new_alpha_j

                """
                print('i: ', i, 'j: ', j)
                print('f_i: ', self.f(x_i, self.w, self.b), 'y_i: ', y_i,
                      'f_j: ', self.f(x_j, self.w, self.b), 'y_j: ', y_j)
                print('L: ', L, 'H: ', H, 'E_i: ', E_i, 'E_j: ', E_j)
                print('New unbounded alpha j: ', new_alpha_j)
                print('New alpha j: ', alpha[j])
                """

            # End loop if convergence param is attained
            if np.linalg.norm(alpha - alpha_prev) < self.epsilon:
                break

        # Save support vectors
        self.train_iterations = iterations
        self.support_vectors = X[np.where(alpha > 0)[0], :]

    def score(self, X, y):
        if not self.b:
            print('SVM has not been trained yet')
        else:
            predictions = self.predict(X)
            return np.sum(y == predictions)/y.shape[0]

    def compute_w_b(self, y, alpha, X):
        self.w = np.matmul(y*alpha, X).T
        self.b = np.mean(y - np.matmul(self.w.T, X.T))
                
    def f(self, x, w, b):
        return np.sign(np.matmul(x.astype('float'), w) + b).astype(int)

    def E(self, x, y, w, b):
        return self.f(x, w, b) - y

    def obj_func(self, nu, alpha_j, new_alpha_j, y_j, E_i, E_j):
        return 0.5*nu*new_alpha_j**2 + (y_j*(E_i - E_j) - nu*alpha_j)*new_alpha_j

    def L_H(self, alpha_i, alpha_j, s, C):
        if s == -1:
            return max(0, alpha_j - alpha_i), min(C, C + alpha_j - alpha_i)
        else:
            return max(0, alpha_i + alpha_j - C), min(C, alpha_i + alpha_j)

    def predict(self, features):
        return self.f(features, self.w, self.b)


def test_svm():

    # Read data from text file
    df = pd.read_csv('data/breast-cancer-wisconsin.data')
    df = df.replace('?', -99999999).drop(['id'], axis=1)

    # Prepare X and y inputs
    X = df.drop(['class'], axis=1)
    y = df['class'].replace(2, -1).replace(4, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train SVM
    classifier = SVM()
    classifier.fit(X_train, y_train)

    # Print Model parameters
    print('SVM parameters')
    print('w: ', classifier.w, 'b: ', classifier.b)
    print('Support vector count: ', len(classifier.support_vectors))

    # Test SVM accuracy
    accuracy = classifier.score(X_test, y_test)
    print('SVM Accuracy: ', accuracy)

    # Predict random examples
    example_measures = np.array([[8,10,10,8,7,10,9,7,1], [6,1,1,1,2,1,3,1,1], [3,1,1,1,2,1,2,1,1]])
    predictions = classifier.predict(example_measures)
    print('SVM Predictions: ', predictions, '; Actual: ', [1, -1, -1])


test_svm()

