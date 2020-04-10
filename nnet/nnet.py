import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
from utils import get_MNIST_data

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions

class NeuralNetwork(nn.Module):

    def __init__(self, input_dimension):
        super(NeuralNetwork, self).__init__()

        self.flatten = Flatten()

        # or declare these separately

        # self.features1 = nn.Sequential(
        #     nn.Conv2d(1, 32, (3, 3)),  # Convolutional layer, 32 filters, size 3×3
        #     nn.ReLU(),  # ReLU nonlinearity
        #     nn.MaxPool2d((2, 2)),  # Max pooling layer with size  2×2
        #     nn.Conv2d(32, 64, (3, 3)),  # Convolutional layer, 64 filters, size 3×3
        #     nn.ReLU(),  # ReLU nonlinearity
        #     nn.MaxPool2d((2, 2)),  # Max pooling layer with size  2×2
        #     Flatten(),  # Flatten layer
        #
        #     # print(input.shape) in the forward section of Flatten in train_utils
        #     nn.Linear(64 * 9 * 5, 128),  # A fully connected layer with 128 neurons
        #
        #     nn.Dropout(p=0.5),  # A dropout layer with drop probability 0.5
        #     nn.Linear(128, 10)  # A fully-connected layer with 10 neurons
        # )

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d((3, 3))
        )

        self.flat_fts = self.get_flat_fts(input_dimension, self.features)

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_fts, 100),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.LogSoftmax()
        )

    def get_flat_fts(self, in_size, fts):
        f = fts(Variable(torch.ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        fts = self.features(x)
        flat_fts = fts.view(-1, self.flat_fts)
        out = self.classifier(flat_fts)
        return out

def main(batch_size=32, lr=0.1, momentum=0, leaky=False, hidden_size=10):
    # Load the dataset
    num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = NeuralNetwork(input_dimension)

    # train_model(train_data, dev_data, model, lr=0.01, momentum=0.9, nesterov=False, n_epochs=10
    train_model(train_batches, dev_batches, model, lr=lr, momentum=momentum)

    ## Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), None)
    print("Loss on test set:" + str(loss) + " Accuracy on test set: " + str(accuracy))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()