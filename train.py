import numpy as np
import pandas as pd
def train(model, x, y):
  epoch_losses=[]
  great_loss=[]
  m=x.shape[0]
  if((m%model.batch_size)==0):
    num_batches=(x.shape[0]//model.batch_size)
  else:
    num_batches=(x.shape[0]//model.batch_size)+1
  print("Number of batches:", num_batches)
  for epoch in range(model.num_epochs):
    losses=[]
    for i in range(num_batches):
      # t=epoch
      # print(i*self.batch_size, (i+1)*self.batch_size)

      xnew= x[i*model.batch_size:(i+1)*model.batch_size]
      ynew= y[i*model.batch_size:(i+1)*model.batch_size]
      output=model.feedforward(xnew)
      model.backprop(xnew,ynew,epoch,i)
      loss= model.cross_entropy(ynew, output)
      losses.append(loss)
      great_loss.append(loss)
    #   losses.append(loss)
    epoch_losses.append(np.mean(losses))
  return epoch_losses, great_loss