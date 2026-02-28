import minitorch
from minitorch import tensor, Tensor

# Network architecture: 2 -> hidden -> hidden -> 1

class TensorNetwork(minitorch.Module):
    def __init__(self, hidden_size: int = 10):
        super().__init__()
        # Layer 1: input (2) -> hidden
        self.layer1_weights = self.add_parameter(
            "layer1_weights",
            minitorch.rand((2, hidden_size), requires_grad=True)
        )
        self.layer1_bias = self.add_parameter(
            "layer1_bias",
            minitorch.rand((hidden_size,), requires_grad=True)
        )

        # Layer 2: hidden -> hidden
        self.layer2_weights = self.add_parameter(
            "layer2_weights",
            minitorch.rand((hidden_size, hidden_size), requires_grad=True)
        )
        self.layer2_bias = self.add_parameter(
            "layer2_bias",
            minitorch.rand((hidden_size,), requires_grad=True)
        )

        # Layer 3: hidden -> output (1)
        self.layer3_weights = self.add_parameter(
            "layer3_weights",
            minitorch.rand((hidden_size, 1), requires_grad=True)
        )
        self.layer3_bias = self.add_parameter(
            "layer3_bias",
            minitorch.rand((1,), requires_grad=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, 2)

        # Layer 1: linear + relu
        # TODO: Implement forward pass
        h1 = ___  # Q1: x @ weights + bias, then relu

        # Layer 2: linear + relu
        h2 = ___  # Q2: h1 @ weights + bias, then relu

        # Layer 3: linear + sigmoid
        out = ___  # Q3: h2 @ weights + bias, then sigmoid

        return out


def train(dataset, epochs=500, learning_rate=0.05, hidden_size=10):
    model = TensorNetwork(hidden_size)

    # Convert dataset to tensors
    X = tensor([[p[0], p[1]] for p in dataset.X])
    y = tensor([[p] for p in dataset.y])

    for epoch in range(epochs):
        # Forward pass
        model.train()
        out = model.forward(X)

        # Loss: mean squared error
        loss = ((out - y) ** 2).sum() / len(dataset.X)

        # Backward pass
        loss.backward()

        # Update weights
        for p in model.parameters():
            if p.value.grad is not None:
                p.update(p.value - learning_rate * p.value.grad)

        # Zero gradients
        model.zero_grad_()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model