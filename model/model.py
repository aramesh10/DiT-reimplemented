import torch
import "../"

class DiT(torch.nn.Module):
    """
    Diffusion Transformer (DiT)
    https://arxiv.org/pdf/2212.09748
    
    Parameters

    Returns

    """
    def __init__(self, num_channels, p, H, W):
        super(DiT, self).__init__()
        self.T = H*W/(p*p)
        self.embed_dim = num_channels * p * p
        
        self.patchify = torch.nn.Unfold(self.p, dilation=1, padding=0, stride=self.p)
        self.patchify_embed = 
        # self.embed = embed
        # self.DiTBlocks = [arr of DiTBlock]
        # self.layer_norm = layer norm
        # self.linear = linear

    def forward(self, x, y, t):
        """
        Forward
        Parameters
            x: noised latent (WxHxC)
            y: label ()
            t: timestep (1)
        Returns

        """
        x = self.patchify(x).permute(0, 2, 1)

    def patchify(self, x):
        """
        Parameters
            x: noised latent (B x C x H x W)
        Returns
            x: tokens (B x T x d)
                T: HW/p^2
                p: patch length
                d: embedding dimension (C*p^2)
        """
        B, C, H, W = x.shape
        x = unfold(x)
        _, T, d = x.shape
        return x

class DiTBlock(torch.nn.Module):
    def __init__(self):
        super(DiTBlock, self).__init__()

    def forward(x, c):
        """
        Forward
        Parameters
            x: input tokens (Txd)
            c: conditioning
        """