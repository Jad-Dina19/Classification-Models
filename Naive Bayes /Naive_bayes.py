import numpy as np

class Naive_Bayes:
    
    def __init__(self, X_radius = 0.4):
        self.X_radius = X_radius
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    
    def euclidean_dist(self, point1, point2):
        
        return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))


    def prior_probability(self):
        y = self.y_train.flatten()
        y_0 = 0
        y_1 = 0
        for i in range(len(y)):
            if y[i] == 0:
                y_0 += 1
            else:
                y_1 += 1
        return y_0/len(y), y_1/len(y)

    def likelihood(self, row):
        train_data = np.hstack((self.X_train, self.y_train))
        
        count1 = 0
        count0 = 0
        total0 = 0
        total1 = 0

        for i in range(train_data.shape[0]):

            if train_data[i][-1] == 1:
                total1 += 1
            elif train_data[i][-1] == 0:
                total0 += 1

            point1 = self.X_train[i]
            point2 = row       
            
            
            dist = self.euclidean_dist(point1, point2) 
            
            if dist < self.X_radius and train_data[i][-1] == 1:
                count1 += 1

            elif dist < self.X_radius and train_data[i][-1] == 0:
                count0 += 1
        
        return count0/total0, count1/total1

       
    def predict(self, X):
        prior0, prior1 = self.prior_probability()
        y_pred = []
        
        for row in X:
            
            likelihood0, likelihood1 = self.likelihood(row)

            p_0 = likelihood0 * prior0
            p_1 = likelihood1 *prior1

            if p_1 > p_0:
                y_pred.append(1)
            else:
                y_pred.append(0)
        
        return y_pred
        

