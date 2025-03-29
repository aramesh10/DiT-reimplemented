import torch
from model.encoding import SinusoidalEncoding 
from ldm.models.autoencoder import VQModelInterface
import utils

class DiT(torch.nn.Module):
    """
    Diffusion Transformer (DiT)
    https://arxiv.org/pdf/2212.09748
    
    Parameters

    Returns

    """
    def __init__(self, vae_param, num_channels, p, H, W):
        super(DiT, self).__init__()
        
        self.T = H*W/(p*p)
        self.embed_dim = num_channels * p * p

        self.unfold = torch.nn.Unfold(kernel_size=(p, p), stride=p)
        self.sinusoid_encoding = SinusoidalEncoding(d=self.embed_dim)
        
        self.patchify = torch.nn.Unfold(self.p, dilation=1, padding=0, stride=self.p)
        self.vae = VQModelInterface(**utils.get_VAE_params(),
                                    ckpt_path='./vq-f8-n256/model.ckpt')
        # self.DiTBlocks = [arr of DiTBlock]
        # self.layer_norm = layer norm
        # self.linear = linear

    def forward(self, x, y, t):
        """
        Parameters
            x: noised latent (B x C x H x W)
        Returns
            x: 
        """
        x = self.patchify(x).permute(0, 2, 1)

    def patchify(self, x):
        """
        Parameters
            x: noised latent (B x C x H x W)
                B: batch size
                C: channel
                H: height
                W: width
        Returns
            x: tokens (B x T x d)
                B: batch size
                T: HW/p^2
                p: patch length
                d: embedding dimension (C*p^2)
        """
        x = self.unfold(x).permute(0, 2, 1)
        x = self.sinusoid_encoding(x)
        return x

class DiTBlock(torch.nn.Module):
    def __init__(self):
        super(DiTBlock, self).__init__()

    def forward(x, c):
        """
        Parameters
            x: input tokens (Txd)
            c: conditioning
        """
        pass
        