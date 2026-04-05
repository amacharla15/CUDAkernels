class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.clamp(input,min=0)

    @staticmethod
    def backward(ctx, grad_output):
        originalinput,=ctx.saved_tensors
        relugradient=torch.where(originalinput<0,originalinput*0,1)
        return grad_output*relugradient

class CustomLeakyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input)
        ctx.alpha=alpha
        return torch.where(input<0,input*alpha,input)

    @staticmethod
    def backward(ctx, grad_output):
        originalinput,=ctx.saved_tensors
        alpha=ctx.alpha
        relugradient=torch.where(originalinput<0,alpha,1)
        return relugradient*grad_output, None

class CustomSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        sigmoid_output=1/(1+torch.exp(-input))
        ctx.save_for_backward(sigmoid_output)
        return sigmoid_output

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid,=ctx.saved_tensors
        return grad_output*sigmoid*(1-sigmoid)