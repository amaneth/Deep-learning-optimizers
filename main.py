from train import train 
from models import NeuralNetwork
import pandas as pd
import argparse
from config import args
from utils import plot

import numpy as np
import pandas as pd

metadata = args.metadata

data= pd.read_csv(metadata)

x=data.iloc[:,1:].values
ycrude=data.iloc[:,0].values

#One hot encoding
y = np.zeros((ycrude.size, ycrude.max()+1))
y[np.arange(ycrude.size),ycrude] = 1
print(x.shape)
print(y.shape)

input_dim = x.shape[1]
hidden_dim = 16
output_dim=y.shape[1]

optim=args.optim
batch_size=args.batch_size
epoch=args.epochs
lr=args.lr
gamma1=args.gamma1
gamma2=args.gamma2
epsillon= args.epsillon
alpha=args.alpha




# Model


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--optim', help='This is name of the optimizer', required=True, type=str)
parser.add_argument('-n', '--num_epochs', help='This is the number of epochs', required=False, type=int, default=epoch)
parser.add_argument('-l', '--lr', help='This is the learning rate', required=False, type=float, default=lr)
parser.add_argument('-g', '--gamma1', help='This is the momentum term', required=False, type=float, default=gamma1)
parser.add_argument('-k', '--gamma2', help='This is the gammma ', required=False, type=float, default=gamma2)
parser.add_argument('-e', '--epsillon', help='This is the epsillon', required=False, type=float, default=epsillon)
parser.add_argument('-a', '--alpha', help='This is the weight decay', required=False, type=float, default=alpha)

main_args = vars(parser.parse_args())
num_epochs = main_args['num_epochs']
lr = main_args['lr']
gamma1 = main_args['gamma1']
gamma2 = main_args['gamma2']
epsillon = main_args['epsillon']
alpha = main_args['alpha']


neuralnet= NeuralNetwork(input_dim,hidden_dim,output_dim, epochs=num_epochs,optim=optim, lr=lr, gamma1=gamma1, gamma2=gamma2, epsillon=epsillon,alpha=alpha)
epoch_loss, _=train(neuralnet,x,y)
plot(epoch_loss, optim)