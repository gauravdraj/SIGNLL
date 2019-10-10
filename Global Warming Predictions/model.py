import torch

class TweetClassifier(torch.nn.Module):

    def __init__(self, num_dims):
        super(TweetClassifier, self).__init__()
        self.weight_vec = torch.nn.Parameter(torch.randn(num_dims))
        self.bias = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        PREDICTION = (self.weight_vec)@(x)+self.bias
        return PREDICTION