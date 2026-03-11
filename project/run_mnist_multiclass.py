import minitorch
from minitorch import tensor, Tensor
from minitorch.nn import maxpool2d, logsoftmax, dropout
from minitorch import Conv2dFun
from minitorch.fast_ops import FastOps
from minitorch.tensor_ops import TensorBackend

FastBackend = TensorBackend(FastOps)

class CNN(minitorch.Module):
    def __init__(self):
        super().__init__()

        # Conv layer 1: 1 -> 4 channels, 3x3 kernel
        self.conv1 = minitorch.Conv2d(1, 4, (3, 3))

        # Conv layer 2: 4 -> 8 channels, 3x3 kernel
        self.conv2 = minitorch.Conv2d(4, 8, (3, 3))

        # Fully connected layers
        # After two 2x2 pools: 28 -> 14 -> 7
        # Flattened: 8 * 7 * 7 = 392
        self.fc1 = minitorch.Linear(392, 64)
        self.fc2 = minitorch.Linear(64, 10)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, 1, 28, 28)

        # Conv block 1
        x = self.conv1(x).relu()
        x = maxpool2d(x, (2, 2))

        # Conv block 2
        x = self.conv2(x).relu()
        x = maxpool2d(x, (2, 2))

        # Flatten
        batch = x.shape[0]
        x = x.view(batch, 392)

        # Fully connected
        x = self.fc1(x).relu()
        x = dropout(x, 0.5, ignore=not self.training)
        x = self.fc2(x)

        # Log probabilities
        return logsoftmax(x, dim=1)


def train_mnist():
    # Load MNIST data via sklearn
    from sklearn.datasets import fetch_openml
    import numpy as np

    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    images, labels = mnist.data, mnist.target.astype(int)

    # Shuffle and split
    n_train, n_test = 5000, 500
    idx = np.random.permutation(len(images))
    train_idx, test_idx = idx[:n_train], idx[n_train:n_train + n_test]

    def img_to_list(flat_img):
        """Convert flat 784-pixel image to [1, 28, 28] nested list, normalized."""
        return [[(float(flat_img[i * 28 + j]) / 255.0) for j in range(28)] for i in range(28)]

    # Keep training data as Python lists so we can slice per batch
    X_train_list = [[img_to_list(images[i])] for i in train_idx]
    y_train_list = [float(labels[i]) for i in train_idx]

    # Test data as tensors (evaluated in one shot)
    X_test = minitorch.tensor([
        [img_to_list(images[i])] for i in test_idx
    ], backend=FastBackend)
    y_test = minitorch.tensor([float(labels[i]) for i in test_idx], backend=FastBackend)

    # Create model
    model = CNN()
    optimizer = minitorch.SGD(model.parameters(), lr=0.5)

    import random

    BATCH_SIZE = 32
    n_train = len(X_train_list)

    # Training loop
    for epoch in range(20):
        model.train()

        indices = list(range(n_train))
        random.shuffle(indices)

        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_train)
            batch_idx = indices[start:end]
            bs = end - start

            X_batch = minitorch.tensor(
                [X_train_list[i] for i in batch_idx],
                backend=FastBackend
            )
            y_batch = minitorch.tensor(
                [y_train_list[i] for i in batch_idx],
                backend=FastBackend
            )

            optimizer.zero_grad()
            log_probs = model.forward(X_batch)
            loss = -(log_probs * minitorch.one_hot(y_batch, 10)).sum() / bs
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Evaluate
        model.eval()
        test_probs = model.forward(X_test)
        predictions = minitorch.argmax(test_probs, dim=1)
        targets = minitorch.one_hot(y_test, 10)
        accuracy = (predictions * targets).sum() / y_test.size

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy.item():.2%}")


if __name__ == "__main__":
    train_mnist()