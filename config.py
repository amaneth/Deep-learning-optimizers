import argparse
args = argparse.Namespace(
    optim='adam',
    batch_size=64,
    epochs=10, 
    lr=0.001,
    gamma1=0.9,
    gamma2=0.9,
    epsillon=1e-8,
    alpha=0.2,
    metadata='./data/mnist_train.csv'
)
