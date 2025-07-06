import numpy as np

class SVM_kernel():

    def __init__(self, C = 1.0, learning_rate = 0.001, epochs = 1000, gamma = 0.5):
        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.gamma = gamma
        self.X_train = None
        self.b = None
        self.alpha = None
        
       
    def fit(self, X, y):
        self.y = np.where(y == 0, -1, 1).flatten()
        self.X_train = X
        self.alpha = self.sga(self.X_train, self.y)
        self.b = self.bias()

    def rbf_kernel(self, x1, x2):
        
        x1 = np.asarray(x1).flatten()  # Ensure 1D
        x2 = np.asarray(x2).flatten()  # Ensure 1D
        diff = x1 - x2
        norm_squared = np.dot(diff, diff)
        return np.exp(-self.gamma * norm_squared)
    
    def kernel_matrix_cross(self, X_test):
       
        X_train = np.asarray(self.X_train)
        X_test = np.asarray(X_test)

        sq_norms_train = np.sum(X_train ** 2, axis=1).reshape(-1, 1) 
        sq_norms_test = np.sum(X_test ** 2, axis=1).reshape(1, -1)    
        dists = sq_norms_train + sq_norms_test - 2 * X_train.dot(X_test.T)  
        K_cross = np.exp(-self.gamma * dists)

        return K_cross

    def rbf_kernel_matrix(self, X):
        X = np.asarray(X)
        sq_norms = np.sum(X ** 2, axis=1).reshape(-1, 1)  
        dists = sq_norms + sq_norms.T - 2 * np.dot(X, X.T)  
        K = np.exp(-self.gamma * dists)
        
        return K

    def sga(self, X, y):

        n_samples = X.shape[0]
        alpha = np.zeros(n_samples)
        self.K = self.rbf_kernel_matrix(X)
       
        for epoch in range(self.epochs):
    
            for i in range(n_samples):
                s_i = np.sum(alpha * y * self.K[:, i])

                grad = float(1 - y[i] * s_i)
                
                alpha[i] = np.clip(alpha[i] + self.learning_rate * grad, 0, self.C)
                
            
        return alpha

    def bias(self):

        support_idx = (self.alpha > 1e-5) & (self.alpha < self.C)
        b_values = []
        for i in np.where(support_idx)[0]:
            s_i = np.sum(self.alpha * self.y * self.K[:, i])
            b_i = self.y[i] - s_i
            b_values.append(b_i)

        return np.mean(b_values) if b_values else 0.0
        
    
    def predict(self, X):
    
        K_test = self.kernel_matrix_cross(X)
        decision = np.sign(np.dot(self.alpha * self.y, K_test) + self.b)
        print("alpha * y shape:", (self.alpha * self.y).shape)
        print("K_test shape:", K_test.shape)
        return np.where(decision == -1, 0, 1)
