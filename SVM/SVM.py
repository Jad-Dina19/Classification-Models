import numpy as np

class SVM:

    def __init__(self, epochs = 1000, c = 1, learning_rate = 0.001, batch_size = 1):
        self.epochs = epochs
        self.c = c 
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.w = None
    
    def fit(self, X, y):
        #comvert 0 to -1 so model works
        for i in range(len(y)):
            if y[i] == 0:
                y[i] = -1
        
        
        self.sgd(X, y, self.learning_rate, self.epochs, self.batch_size)

    def sgd(self, X, y, learning_rate, epochs, batch_size):
        #size of the dataset
        m, n = X.shape
        #write theta 
        self.w = np.random.randn(n + 1, 1)   #randomly initialize theta for random guess
        #write the X bias term
        X_bias = np.c_[np.ones([m, 1]), X]

        #shuffle the dataset
        for epoch in range(epochs):
            
            indicies = np.random.permutation(m)
            X_shuffle = X_bias[indicies]
            y_shuffle = y[indicies]

            #write for loop for batch size
            for i in range(0, m, batch_size): #for smaller datasents range(m) is fine
                #find xbatch and ybatch
                X_batch = X_shuffle[i: i+batch_size]
                y_batch = y_shuffle[i: i+batch_size]
                
                #make empty gradient matrix
                grad = np.zeros_like(self.w)

                #compute the gradient
                for x_i, y_i in zip(X_batch, y_batch):

                    #reshape x_i for matrix multiplication
                    x_i = x_i.reshape(-1, 1)
                    #calculate condition
                    condition = y_i *(self.w.T.dot(x_i))

                    if condition >= 1: 
                        grad += self.w
                    else:
                        grad += self.w - self.c * y_i * x_i
            
                #update the weights
                grad /= batch_size
                
                self.w -= learning_rate * grad
        
    def predict(self, X):

        X_bias = np.c_[np.ones([X.shape[0], 1]), X]
        y_pred = np.sign(X_bias.dot(self.w))
        
        #convert -1 back to 0 for final prediction
        for i in range(y_pred.shape[0]):
            if y_pred[i][0] == -1:
                y_pred[i][0] = 0
        
        return y_pred
        
        