import matplotlib.pyplot as plt

def plot(epoch_loss, optimizer):
    plt.title("Training results: Loss")
    plt.plot(epoch_loss, label=optimizer)
    plt.legend()
    plt.savefig("./figures/train_"+optimizer+".png")
    plt.show()