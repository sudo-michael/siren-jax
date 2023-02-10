import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from typing import List

from siren.dense import CustomDense, siren_init, siren_init_first, bias_uniform

class Siren(nn.Module):
    num_channels: List[int]
    output_dim: int
    omega: float = 30.0
    
    @nn.compact
    def __call__(self, x):
        x = CustomDense(self.num_channels[0], kernel_init=siren_init_first(), bias_init=bias_uniform())(x)
        x = jnp.sin(self.omega * x)
        for nc in self.num_channels:
            x = CustomDense(nc, kernel_init=siren_init(w0=self.omega), bias_init=bias_uniform())(x)
            x = jnp.sin(self.omega * x)
        x = CustomDense(self.output_dim, kernel_init=siren_init(w0=self.omega), bias_init=bias_uniform())(x)
        return x