class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # save what you need for backward
        # ctx.save_for_backward(input)
        # return the result
        ctx.save_for_backward(input)
        return torch.clamp(input,min=0)

    @staticmethod
    def backward(ctx, grad_output):

        originalinput,=ctx.saved_tensors
        relugradient=torch.where(originalinput<0,originalinput*0,1)
        return grad_output*relugradient

