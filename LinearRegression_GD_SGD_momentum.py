''' Linear Regression with possible modes of 'NE' (normal equation, exact solution), 'GD' (gradient descent) ,
'GDM' (GD with momentum), 'SGD' (stochastic GD), 'SGDM' (stochastic GD with momentum) '''

import numpy as np

class LinearRegression():
        
        #********************************************************************
        # INITIALIZING  
        #********************************************************************
        def __init__(self, mode = "NE", learn_rate = 0.01, nb_epochs = 1000, batch_size = 32, beta = 0.9, tolerance = 1e-4):
            if mode not in ["NE", "GD", "GDM", "SGD", "SGDM"]:
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
        # FIT 
        #********************************************************************
        
        def fit(self, X, Y):
            X_1 = np.c_[np.ones(X.shape[0]), X]
            
            #----------------------------------------------------------------
            # The exact solution
            #----------------------------------------------------------------
            
            if self.mode == "NE":
                self.Theta = np.linalg.inv(X_1.T.dot(X_1)).dot(X_1.T.dot(Y))
                return True
            
            
            #----------------------------------------------------------------
            # Gradient Descent
            #----------------------------------------------------------------
            
            elif self.mode == "GD":
                if self.Theta is None:
                    self.Theta = np.zeros((X_1.shape[1], 1))
                grad = X_1.T.dot(X_1.dot(self.Theta) - Y)/X_1.shape[0]
                for i in range(self.nb_epochs):
                    # check that gradient is not too small value
                    if np.linalg.norm(grad) > self.tolerance:
                        self.Theta -= self.learn_rate*grad
                        grad = X_1.T.dot(X_1.dot(self.Theta) - Y)/X_1.shape[0]
                return True
            
            
            #----------------------------------------------------------------
            # Gradient Descent with Momentum
            #----------------------------------------------------------------
            
            elif self.mode == "GDM":
                if self.Theta is None:
                    self.Theta = np.zeros((X_1.shape[1], 1))
                    self.mo = np.zeros_like(self.Theta)
                grad = X_1.T.dot(X_1.dot(self.Theta) - Y)/X_1.shape[0]
                for i in range(self.nb_epochs):
                    # check that gradient is not too small value
                    if np.linalg.norm(grad) > self.tolerance:
                        # momentum optimization
                        self.mo = self.beta*self.mo + grad
                        self.Theta -= self.learn_rate*self.mo
                        grad = X_1.T.dot(X_1.dot(self.Theta) - Y)/X_1.shape[0]
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
                                        
                        grad = (x_batch.T.dot(x_batch.dot(self.Theta) - y_batch)) / x_batch.shape[0]                        
                        self.Theta -= self.learn_rate*grad
                return True
            
            
            #----------------------------------------------------------------
            # Stochastic Gradient Descent with momentum
            #----------------------------------------------------------------
            
            elif self.mode == "SGDM":                
                if self.Theta is None:
                    self.Theta = np.zeros((X_1.shape[1], 1))
                    self.mo = np.zeros_like(self.Theta)
                    
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
                                        
                        grad = (x_batch.T.dot(x_batch.dot(self.Theta) - y_batch)) / x_batch.shape[0]
                        # momentum optimization
                        self.mo = self.beta*self.mo + grad
                        self.Theta -= self.learn_rate*self.mo
                return True
               
        
        #********************************************************************
        # PREDICT 
        #********************************************************************                
            
        def predict(self, X):
            if self.Theta is None:
                return False
            else:
                X_1 = np.c_[np.ones(X.shape[0]), X]
                return X_1.dot(self.Theta)