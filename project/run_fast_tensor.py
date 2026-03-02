import time
import minitorch
from minitorch import tensor, Tensor

def train(dataset, hidden_size=100, epochs=500, learning_rate=0.05, backend="cpu"):
    """Train model with specified backend."""

    if backend == "cpu":
        from minitorch.fast_ops import FastOps
        Backend = minitorch.TensorBackend(FastOps)
    elif backend == "gpu":
        from minitorch.cuda_ops import CudaOps
        Backend = minitorch.TensorBackend(CudaOps)
    else:
        from minitorch.tensor_ops import SimpleOps
        Backend = minitorch.TensorBackend(SimpleOps)

    # Create model (same as Module 2)
    # ...

    for epoch in range(epochs):
        start = time.time()

        # Forward pass
        out = model.forward(X)
        loss = ((out - y) ** 2).sum() / len(dataset.X)

        # Backward pass
        loss.backward()

        # Update weights
        for p in model.parameters():
            if p.value.grad is not None:
                p.update(p.value - learning_rate * p.value.grad)
        model.zero_grad_()

        epoch_time = time.time() - start

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Time: {epoch_time:.3f}s")

    return model