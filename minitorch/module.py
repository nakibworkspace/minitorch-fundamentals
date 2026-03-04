"""
Module system for organizing neural network components and parameters.

This provides the foundation for building complex neural networks
with proper parameter management and hierarchy.
"""

from typing import Dict, List, Tuple, Any, Sequence


class Parameter:
    """A trainable parameter in a neural network."""
    
    def __init__(self, value: Any):
        self.value = value

    def update(self, new_value: Any) -> None:
        self.value = new_value
        
    @property
    def shape(self):
        """Return parameter shape."""
        if hasattr(self.value, 'shape'):
            return self.value.shape
        return ()
    
    def __repr__(self):
        return f"Parameter(value={self.value})"


class Module:
    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence["Module"]:
        results = []
        for module in self._modules.values():
            results.append(module)
            results.extend(module.modules())
        return results

    def train(self):
        self.training = True
        for module in self.modules():
            module.training = True

    def eval(self):
        self.training = False
        for module in self.modules():
            module.training = False

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        results = []
        for name, param in self._parameters.items():
            results.append((name, param))
        for module_name, module in self._modules.items():
            for param_name, param in module.named_parameters():
                full_name = f"{module_name}.{param_name}"
                results.append((full_name, param))
        return results

    def parameters(self) -> Sequence[Parameter]:
        return [param for _, param in self.named_parameters()]

    def add_parameter(self, name: str, value: Any) -> Parameter:
        if isinstance(value, Parameter):
            param = value
        else:
            param = Parameter(value)
        self._parameters[name] = param
        return param

    def __setattr__(self, key: str, value: Any):
        if isinstance(value, Parameter):
            self._parameters[key] = value
        elif isinstance(value, Module):
            self._modules[key] = value

        super().__setattr__(key, value)
