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
            u__t, u__x = jacobian(u, [t, x], create_graph=True)
            u__x__x = jacobian(u__x, x, create_graph=True)
            u__x__x__x = jacobian(u__x__x, x, create_graph=False)

            return u__t + self.eta * u * u__x + self.mu**2 * u__x__x__x

        self.add_equation("kdv", kdv)


# class KDV(base.PDE):
#     def __init__(
#         self,
#         eta: float = 1.0,
#         mu: float = 0.022,
#         detach_keys: Optional[Tuple[str, ...]] = None,
#     ):
#         super().__init__()
#         self.detach_keys = detach_keys

#         t, x = self.create_symbols("t x")
#         u = self.create_function("u", (t, x))

#         # u_t + eta * u * u_x + mu**2 * u_xxx
#         kdv = u.diff(t) + eta * u * u.diff(x) + mu**2 * u.diff(x, 3)

#         self.add_equation("kdv", kdv)

#         self._apply_detach()
