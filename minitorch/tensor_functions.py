class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out

class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        return out

class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        return a.f.lt_zip(a, b)

class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        return a.f.eq_zip(a, b)

class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        return a.f.is_close_zip(a, b)

class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        ctx.save_for_backward(order)
        order_list = [int(order[i]) for i in range(order.size)]
        return a._new(a._tensor.permute(*order_list))

class Mul(Function):
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = ctx.saved_values
        return (
            grad_output.f.mul_zip(grad_output, b),
            grad_output.f.mul_zip(grad_output, a)
        )


class Sigmoid(Function):
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (out,) = ctx.saved_values
        one = out.zeros(out.shape)
        one._tensor._storage[:] = 1.0
        one_minus_out = one.f.add_zip(one, out.f.neg_map(out))
        sig_deriv = out.f.mul_zip(out, one_minus_out)
        return grad_output.f.mul_zip(grad_output, sig_deriv)


class ReLU(Function):
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (out,) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, out)


class LT(Function):
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        zeros_a = grad_output.zeros(grad_output.shape)
        zeros_b = grad_output.zeros(grad_output.shape)
        return zeros_a, zeros_b


class EQ(Function):
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        zeros_a = grad_output.zeros(grad_output.shape)
        zeros_b = grad_output.zeros(grad_output.shape)
        return zeros_a, zeros_b


class Permute(Function):
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        (order,) = ctx.saved_values
        order_list = [int(order[i]) for i in range(order.size)]
        inv_order = [0] * len(order_list)
        for i, o in enumerate(order_list):
            inv_order[o] = i
        return grad_output._new(grad_output._tensor.permute(*inv_order)), 0.0