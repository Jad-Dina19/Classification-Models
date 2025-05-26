import numpy as np
class LogisticRegression:
    def __init__(self, learning_rate = 0.01, epochs = 1000, batch_size = 1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regressor = None
        
    def fit(self, X, y):
        self.regressor = self.sgd(X, y, self.learning_rate, self.epochs, self.batch_size)

    def sigmoid(self, z):
        #calculate sigmoid function
        return 1/(1 + np.exp(-z))

    def sgd(self, X, y, learning_rate, epochs, batch_size):
        #size of the dataset
        m, n = X.shape
        #write theta 
        theta = np.random.randn(n + 1, 1)   #randomly initialize theta for random guess
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
                X_batch = X_shuffle[i: i+1]
                y_batch = y_shuffle[i: i+1]

                #calculate z
                z = X_batch.dot(theta)
                h = self.sigmoid(z)

                #compute the gradient
                gradient =X_batch.T.dot(h - y_batch)/batch_size
            
                #update the weights
                theta -= learning_rate * gradient
        
        #return theta
        return theta
    
    def predict(self, X):
        #calculate prbability
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        probability = self.sigmoid(X_bias.dot(self.regressor))
        
        #convert values to 1 or 0
        for index in range(probability.shape[0]):

            if(probability[index, 0] < 0.5):

                probability[index, 0] = 0
            else:

                probability[index, 0] = 1
        
        return probability


