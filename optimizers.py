import numpy as np
import pandas as pd

def optim_sgd(model,d_weights1, d_weights2, d_bias1, d_bias2):
  model.weights1 -= model.lr*d_weights1
  model.weights2 -= model.lr*d_weights2
  model.b1 -= model.lr*d_bias1
  model.b2 -= model.lr*d_bias2
def optim_momentum(model,d_weights1, d_weights2, d_bias1, d_bias2):
  model.vd_weights2= model.gamma1*model.vd_weights2 + (1-model.gamma1)*d_weights2
  model.vd_weights1= model.gamma1*model.vd_weights1 + (1-model.gamma1)*d_weights1

  model.vd_bias2= model.gamma1*model.vd_bias2 + (1-model.gamma1)*d_bias2
  model.vd_bias1= model.gamma1*model.vd_bias1 + (1-model.gamma1)*d_bias1
  
  # update the weights with the derivative (slope) of the loss function
  model.weights1 -= model.lr*model.vd_weights1
  model.weights2 -= model.lr*model.vd_weights2

  model.b1 -= model.lr*model.vd_bias1
  model.b2 -= model.lr*model.vd_bias2
def optim_rmsprop(model,d_weights1, d_weights2, d_bias1, d_bias2):
  model.vd_weights2= model.gamma1*model.vd_weights2 + (1-model.gamma1)*np.square(d_weights2)
  model.vd_weights1= model.gamma1*model.vd_weights1 + (1-model.gamma1)*np.square(d_weights1)

  model.vd_bias2= model.gamma1*model.vd_bias2 + (1-model.gamma1)*np.square(d_bias2)
  model.vd_bias1= model.gamma1*model.vd_bias1 + (1-model.gamma1)*np.square(d_bias1)
  
  # update the weights with the derivative (slope) of the loss function
  model.weights1 -= model.lr*(d_weights1/np.sqrt(model.vd_weights1+model.epsillon))
  model.weights2 -= model.lr*(d_weights2/np.sqrt(model.vd_weights2+model.epsillon))

  model.b1 -= model.lr*(d_bias1/np.sqrt(model.vd_bias1+model.epsillon))
  model.b2 -= model.lr*(d_bias2/np.sqrt(model.vd_bias2+model.epsillon))

def optim_adagrad(model,d_weights1, d_weights2, d_bias1, d_bias2):
    model.vd_weights2 += np.square(d_weights2)
    model.vd_weights1 += np.square(d_weights1)

    model.vd_bias2 += np.square(d_bias2)
    model.vd_bias1 +=np.square(d_bias1)
    
    # update the weights with the derivative (slope) of the loss function
    model.weights1 -= model.lr*(d_weights1/np.sqrt(model.vd_weights1+model.epsillon))
    model.weights2 -= model.lr*(d_weights2/np.sqrt(model.vd_weights2+model.epsillon))

    model.b1 -= model.lr*(d_bias1/np.sqrt(model.vd_bias1+model.epsillon))
    model.b2 -= model.lr*(d_bias2/np.sqrt(model.vd_bias2+model.epsillon)) 

def optim_adam(model,d_weights1, d_weights2, d_bias1, d_bias2,epoch,iteration):
      decay_rate=1.0
      t= (10*epoch)+iteration+1
      a= (1/(1+decay_rate*(epoch+1)))*model.alpha
      # if t%100==0.0:
      #   print("alpha here is:", a, model.alpha,1/(1+decay_rate*(epoch+1)) )

      if model.alpha==0.0:
        a=model.lr


      model.vd_weights2= model.gamma1*model.vd_weights2 + (1-model.gamma1)*d_weights2
      model.vd_weights1= model.gamma1*model.vd_weights1 + (1-model.gamma1)*d_weights1
      model.vd_bias2= model.gamma1*model.vd_bias2 + (1-model.gamma1)*d_bias2
      model.vd_bias1= model.gamma1*model.vd_bias1 + (1-model.gamma1)*d_bias1

      #correction
      vd_weights2_corrected= model.vd_weights2/(1-np.power(model.gamma1,t))
      vd_weights1_corrected= model.vd_weights1/(1-np.power(model.gamma1,t))
      vd_bias2_corrected= model.vd_bias2/(1-np.power(model.gamma1,t))
      vd_bias1_corrected= model.vd_bias1/(1-np.power(model.gamma1,t))

      model.sd_weights2= model.gamma2*model.sd_weights2 + (1-model.gamma2)*np.square(d_weights2)
      model.sd_weights1= model.gamma2*model.sd_weights1 + (1-model.gamma2)*np.square(d_weights1)
      model.sd_bias2= model.gamma2*model.sd_bias2 + (1-model.gamma2)*np.square(d_bias2)
      model.sd_bias1= model.gamma2*model.sd_bias1 + (1-model.gamma2)*np.square(d_bias1)

      #correction
      sd_weights2_corrected= model.sd_weights2/(1-np.power(model.gamma2,t))
      sd_weights1_corrected= model.sd_weights1/(1-np.power(model.gamma2,t))
      sd_bias2_corrected= model.sd_bias2/(1-np.power(model.gamma2,t))
      sd_bias1_corrected= model.sd_bias1/(1-np.power(model.gamma2,t))





      # update the weights with the derivative (slope) of the loss function
      model.weights1 -= a*(vd_weights1_corrected/(np.sqrt(sd_weights1_corrected)+model.epsillon))
      model.weights2 -= a*(vd_weights2_corrected/(np.sqrt(sd_weights2_corrected)+model.epsillon))

      model.b1 -= a*(vd_bias1_corrected/(np.sqrt(sd_bias1_corrected)+model.epsillon))
      model.b2 -= a*(vd_bias2_corrected/(np.sqrt(sd_bias2_corrected)+model.epsillon))
