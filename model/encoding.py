import torch 

class SinusoidalEncoding(torch.nn.Module):
    def __init__(self, d, base=10000):
        assert (d % 2 == 0)
        
        super(SinusoidalEncoding, self).__init__()
        
        self.dim = d
        self.base = base
        calc_w_k = lambda k: 1 / torch.pow(torch.tensor(base), torch.tensor(2*k/d))
        self.w = torch.tensor([calc(k) for k in range(d//2) for calc in [calc_w_k, calc_w_k]])
        
    def pos_embed(self, max_t):
        pos_embed = torch.outer(torch.arange(0, max_t), self.w)
        pos_embed[::2] = torch.sin(pos_embed[::2])
        pos_embed[1::2] = torch.sin(pos_embed[1::2])
        return pos_embed
    
    def forward(self, x):
        """
        Forward
        Input
            x: input tensor (b,t,d)
        Return
            x: output tensor with position embedding  (b,t,d)
        """
        b, t, d = x.shape
        for batch in range(b):
            x[batch] = x[batch] + self.pos_embed(t)
        return x 

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = torch.zeros((1, 50, 128))
    embedding = SinusoidalEncoding(d=128)
    plt.imshow(torch.squeeze(embedding(x)))
    plt.show()