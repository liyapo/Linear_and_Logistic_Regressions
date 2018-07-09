''' Logistic Regression with possible modes of 'GD' (gradient descent) and 'SGD' (stochastic GD)
 '''

import numpy as np


class LogisticRegression():
        
        #********************************************************************
        # INITIALIZING  
        #********************************************************************
        def __init__(self, mode = "SGD", learn_rate = 0.01, nb_epochs = 1000, batch_size = 32, beta = 0.9, tolerance = 1e-4):
            if mode not in ["GD", "SGD"]:
                raise ValueError(mode + " is not a valid choice.")
            self.mode = mode
            # parameters
            self.Theta = None
            # learning rate  
            self.learn_rate = learn_rate
            # number of max iterations 
            self.nb_epochs = nb_epochs
            # optimizing with momentum 
            self.beta = beta
            self.mo = None
            # the minimum value of norm(grad) for momentum optimization
            self.tolerance = tolerance
            # size of batch in SGD
            self.batch_size = batch_size
            
        
        
        #********************************************************************    
        # SIGMOID FUNCTION
        #********************************************************************
        
        def __sigmoid(self, z):
            return 1 / (1 + np.exp(-z))
        
        
        
        #********************************************************************    
        # LOSS FUNCTION
        #********************************************************************
        
        def __loss(self, h, y):
            return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        
        
        #********************************************************************
        # FIT 
        #********************************************************************
        
        def fit(self, X, Y):
            
            X_1 = np.c_[np.ones(X.shape[0]), X]

                        
            #----------------------------------------------------------------
            # Gradient Descent
            #----------------------------------------------------------------
            
            if self.mode == "GD":
                if self.Theta is None:
                    self.Theta = np.zeros((X_1.shape[1], 1))
                z = X_1.dot(self.Theta)
                h = self.__sigmoid(z)
                grad =  X_1.T.dot((h - Y)) / X_1.shape[0]
                for i in range(self.nb_epochs):
                    # check that gradient is not too small value
                    if np.linalg.norm(grad) > self.tolerance:
                        self.Theta -= self.learn_rate*grad
                        z = X_1.dot(self.Theta)
                        h = self.__sigmoid(z)
                        grad =  X_1.T.dot((h - Y)) / X_1.shape[0]
                        #print(self.__loss(h, Y))
                return True
            
            
         
            #----------------------------------------------------------------
            # Stochastic Gradient Descent
            #----------------------------------------------------------------
            
            elif self.mode == "SGD":                
                if self.Theta is None:
                    self.Theta = np.zeros((X_1.shape[1], 1))
                    
                # Calculating number of batches
                nb_X = X_1.shape[0]
                nb_batch = int(nb_X / self.batch_size)
            
                # If size of last batch is less that batch_size
                # then consider it as another batch anyway
                if nb_X % nb_batch != 0:
                    nb_batch += 1
                 
                for i in range(self.nb_epochs):
                    # returns permuted range 
                    permuted_nb_batch = np.random.permutation(nb_batch)
                
                    for batch in range(nb_batch):
                        # we run through the list of permuted batches
                        # len(permuted_nb_batch) = nb_batch
                        j = permuted_nb_batch[batch]                    
                        x_batch = X_1[j*self.batch_size:(j+1)*self.batch_size,:]
                        y_batch = Y[j*self.batch_size:(j+1)*self.batch_size,:]
                                        
                        z = x_batch.dot(self.Theta)
                        h = self.__sigmoid(z)
                        grad =  x_batch.T.dot((h - y_batch)) / x_batch.shape[0]
                        self.Theta -= self.learn_rate*grad
                return True
            
                     
        
        #********************************************************************
        # PREDICT 
        #********************************************************************                
            
        def predict(self, X):
            if self.Theta is None:
                return False
            else:
                X_1 = np.c_[np.ones(X.shape[0]), X]
                return self.__sigmoid(X_1.dot(self.Theta))
        
        
        #********************************************************************
        # Accuracy 
        #******************************************************************** 
        def accuracy(self, y_pred, y):
            return np.mean(np.abs(y_pred - y))   