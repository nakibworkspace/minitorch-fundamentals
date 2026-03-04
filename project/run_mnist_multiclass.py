import minitorch
from minitorch import tensor, Tensor
from minitorch.nn import maxpool2d, logsoftmax, dropout
from minitorch import Conv2dFun

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
    # Load MNIST data
    from mnist import MNIST
    mndata = MNIST('data/mnist')
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    # Preprocess
    def preprocess(images, labels, limit=1000):
        X = tensor([
            [[img[i*28:(i+1)*28] for i in range(28)]]
            for img in images[:limit]
        ]) / 255.0

        y = tensor(labels[:limit])
        return X, y

    X_train, y_train = preprocess(train_images, train_labels, 1000)
    X_test, y_test = preprocess(test_images, test_labels, 200)

    # Create model
    model = CNN()
    optimizer = minitorch.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(20):
        model.train()

        # Forward
        log_probs = model.forward(X_train)

        # NLL Loss
        loss = -(log_probs * minitorch.one_hot(y_train, 10)).sum() / len(y_train)

        # Backward
        loss.backward()

        # Update
        optimizer.step()
        optimizer.zero_grad()

        # Evaluate
        model.eval()
        with minitorch.no_grad():
            test_probs = model.forward(X_test)
            predictions = test_probs.argmax(dim=1)
            accuracy = (predictions == y_test).sum() / len(y_test)

        print(f"Epoch {epoch}: Loss={loss.item():.4f}, Accuracy={accuracy.item():.2%}")


if __name__ == "__main__":
    train_mnist()