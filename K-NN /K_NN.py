import numpy as np

class K_NN:
    def __init__(self, n_neighbours = 5, p = 2):
        self.p = p
        self.n_neighbours = n_neighbours
        self.y_train= None
        self.classifier = None
    
    def fit(self, X, y):
        self.y_train = y
        self.X_train = X

    def euclidean_distance(self, point1, point2):
        #calculate euclidean distance
        return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))
    
    def manhattan_distance(self, point1, point2):
        #calculate manhattan_distance
        return np.sum(np.abs(np.array(point1) - np.array(point2)))
    
    def predict(self, X):
        predictions = []
    
        for i in range(X.shape[0]):
            distances = []

            for j in range(self.X_train.shape[0]):
                
                if(self.p == 2):
                    dist = self.euclidean_distance(self.X_train[j], X[i])
                    distances.append(dist)
        
                elif(self.p == 1):
                    dist = self.euclidean_distance(self.X_train[j], X[i])
                    distances.append(dist)
        
            indices = np.argsort(distances)[:self.n_neighbours]
            min_labels = self.y_train[indices]
           
            count_ones = np.sum(min_labels == 1)
            count_zeros = np.sum(min_labels == 0)
            

            if count_ones > count_zeros:
                predictions.append(1)
            else:
                predictions.append(0)

        return np.array(predictions)
            
            
        
