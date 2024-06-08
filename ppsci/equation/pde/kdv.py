from __future__ import annotations
from typing import Optional
from typing import Tuple

from ppsci.autodiff import jacobian
from ppsci.equation.pde import base


class KDV(base.PDE):
    def __init__(
        self,
        eta: float = 1.0,
        mu: float = 0.022,
        detach_keys: Optional[Tuple[str, ...]] = None,
    ):
        super().__init__()
        self.detach_keys = detach_keys
        self.eta = eta
        self.mu = mu

        # kdv = u.diff(t) + eta * u * u.diff(x) + mu * mu * u.diff(x, 3)
        def kdv(out):
            t, x = out["t"], out["x"]
            u = out["u"]
            u__t, u__x = jacobian(u, [t, x])
            u__x__x = jacobian(u__x, x)
            u__x__x__x = jacobian(u__x__x, x, create_graph=False)

            return u__t + self.eta * u * u__x + self.mu * self.mu * u__x__x__x

        self.add_equation("KDV", kdv)
