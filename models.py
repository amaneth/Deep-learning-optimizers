from optimizers import *
import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self,input_dim,hidden_dim,output_dim,optim='adam',batch_size=64,epochs=10, lr=0.001, gamma1=0.9, gamma2=0.9,epsillon=1e-8, alpha=0.2):
        np.random.seed(0)
        self.optim=optim
        self.weights1 = np.random.rand(input_dim,hidden_dim)*1e-1
        self.weights2 = np.random.rand(hidden_dim,output_dim)*1e-1  
        self.vd_weights1=np.zeros((input_dim,hidden_dim))   
        self.vd_weights2=np.zeros((hidden_dim,output_dim))   
        self.sd_weights1=np.zeros((input_dim,hidden_dim))   
        self.sd_weights2=np.zeros((hidden_dim,output_dim))    
        self.output = None
        self.b1= np.random.rand(1,hidden_dim)
        self.b2= np.random.rand(1,output_dim)
        self.vd_bias1=np.zeros((1,hidden_dim))
        self.vd_bias2=np.zeros((1,output_dim))
        self.sd_bias1=np.zeros((1,hidden_dim))
        self.sd_bias2=np.zeros((1,output_dim))
        self.num_epochs = epochs
        self.lr=lr
        self.batch_size=batch_size
        self.gamma1=gamma1
        self.gamma2=gamma2
        self.epsillon=epsillon
        self.alpha=alpha
    def sigmoid(self,x):
      return 1.0/(1+ np.exp(-x))

    def sigmoid_derivative(self,x):
      return self.sigmoid(x) * (1.0 - self.sigmoid(x))


    def cross_entropy(self,y,y_pred):
      loss= -np.sum((y*np.log(y_pred))+((1-y)*np.log(1-y_pred)))/len(y)
      return loss

    def feedforward(self, x):
      self.layer1 = self.sigmoid(np.dot(x, self.weights1)+ self.b1)
      self.output = self.sigmoid(np.dot(self.layer1, self.weights2)+self.b2)
      return self.output
    def backprop(self, x,y,epoch,iteration):
    # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
      d_weights2 = -np.dot(self.layer1.T, (2*(y-self.output ) * self.sigmoid_derivative(self.output)))
      d_weights1 = -np.dot(x.T,  (np.dot(2*(y-self.output) * self.sigmoid_derivative(self.output), self.weights2.T) * self.sigmoid_derivative(self.layer1)))
      
      d_bias2 = -(2*(y-self.output ) * self.sigmoid_derivative(self.output))
      d_bias1 = -(np.dot(2*(y-self.output) * self.sigmoid_derivative(self.output), self.weights2.T) * self.sigmoid_derivative(self.layer1))
      d_bias1= np.sum(d_bias1, axis=0).reshape(1,-1)
      d_bias2= np.sum(d_bias2, axis=0).reshape(1,-1)
      if self.optim=='adam':
          optim_adam(self, d_weights1,d_weights2,d_bias1,d_bias2,epoch,iteration)
      elif self.optim=='sgd':
          optim_sgd(self, d_weights1,d_weights2,d_bias1,d_bias2)
      elif self.optim=='momentum':
          optim_momentum(self, d_weights1,d_weights2,d_bias1,d_bias2)
      elif self.optim=='rmsprop':
          optim_rmsprop(self, d_weights1,d_weights2,d_bias1,d_bias2)
      elif self.optim=='adagrad':
          optim_adagrad(self, d_weights1,d_weights2,d_bias1,d_bias2)
      else:
          print("No optim found!")