import jax
import jax.numpy as jnp

from jax import config
from jaxtyping import (
    PyTree,
    Array,
    Float,
    Int,
    install_import_hook,
)

config.update("jax_enable_x64", True)

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
from gpjax.kernels.computations import DenseKernelComputation


class StationaryDerivativeKernel(gpx.kernels.AbstractKernel):

    '''
    Given a kernel compute the covariance matrix for the given kernel and for the derivative of the kernel.\
    Here the second column of the input data must be 0 or 1, with 0 being the normal and 1 being the derivative
    '''
    def __init__(
        self,
        kernel: gpx.kernels.AbstractKernel,
        ): 
        self.kernel = kernel
        
        super().__init__(compute_engine=DenseKernelComputation())

    def __call__(
        self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]) -> Float[Array, "1"]:
        z = jnp.array(X[1], dtype=int)
        zp = jnp.array(Xp[1], dtype=int)

        k0 = (1 - z)* (1 - zp) * self.kernel(X, Xp)
        k1 = (z * zp) * jnp.array(jax.hessian(self.kernel, argnums=[0,1])(X, Xp), dtype=jnp.float64)[1][0][0][0]
        k2 = (z - zp) * (z - zp) * jnp.array(jax.grad(self.kernel, argnums=[0,1])(X, Xp), dtype=jnp.float64)[zp][0]

        return k0 + k1 + k2 