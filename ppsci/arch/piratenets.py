import paddle
import paddle.nn as nn
import numpy as np

from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

from ppsci.arch import activation as act_mod
from ppsci.arch import base
from ppsci.arch.mlp import RandomWeightFactorization
from ppsci.arch.mlp import FourierEmbedding
from ppsci.arch.mlp import PeriodEmbedding


class CustomLayer(nn.Layer):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = self.create_parameter(
            [1], default_initializer=nn.initializer.Constant(alpha)
        )

    def forward(self, h, phi_x):
        return self.alpha * h + (1 - self.alpha) * phi_x


class ResidualBlock(nn.Layer):
    def __init__(
        self,
        hidden_size,
        alpha: float = 0,
        activation: str = "tanh",
        mu: float = 1.0,
        sigma: float = 0.1,
    ):
        super().__init__()
        self.dense_1 = RandomWeightFactorization(
            hidden_size, hidden_size, True, mu, sigma
        )
        self.act_1 = act_mod.get_activation(activation)
        self.dense_2 = RandomWeightFactorization(
            hidden_size, hidden_size, True, mu, sigma
        )
        self.act_2 = act_mod.get_activation(activation)
        self.dense_3 = RandomWeightFactorization(
            hidden_size, hidden_size, True, mu, sigma
        )
        self.act_3 = act_mod.get_activation(activation)

        self.addaptive_layer = CustomLayer(alpha)

    def forward(self, phi_x, U, V):
        f = self.act_1(self.dense_1(phi_x))
        z_1 = paddle.multiply(f, U) + paddle.multiply(1 - f, V)
        g = self.act_2(self.dense_2(z_1))
        z_2 = paddle.multiply(g, U) + paddle.multiply(1 - g, V)
        h = self.act_3(self.dense_3(z_2))
        return self.addaptive_layer(h, phi_x)


class PirateNets(base.Arch):
    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        num_layers: int,
        hidden_size: int,
        alpha: float = 0.0,
        mu: float = 1.0,
        sigma: float = 0.1,
        activation: str = "tanh",
        periods: Optional[Dict[int, Tuple[float, bool]]] = None,
        fourier: Optional[Dict[str, Union[float, int]]] = None,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys

        self.block_size = num_layers // 3
        self.periods = periods
        self.fourier = fourier

        if periods:
            self.period_emb = PeriodEmbedding(periods)

        cur_size = len(self.input_keys)
        if periods:
            cur_size += len(periods)

        if fourier:
            self.fourier_emb = FourierEmbedding(
                cur_size, fourier["dim"], fourier["scale"]
            )
            cur_size = fourier["dim"]

        self.linear_U = RandomWeightFactorization(
            cur_size, hidden_size, True, mu, sigma
        )
        self.act_u = act_mod.get_activation(activation)
        self.linear_V = RandomWeightFactorization(
            cur_size, hidden_size, True, mu, sigma
        )
        self.act_v = act_mod.get_activation(activation)

        self.residual_blocks = nn.LayerList(
            [
                ResidualBlock(
                    hidden_size,
                    alpha,
                    activation,
                    mu,
                    sigma,
                )
                for _ in range(self.block_size)
            ]
        )

        # self.linear_out = RandomWeightFactorization(
        #     hidden_size, len(self.output_keys), False, mu, sigma
        # )

        self.linear_out = nn.Linear(hidden_size, len(self.output_keys), bias_attr=False)

    def pi_initialization(
        self, x: Dict[str, paddle.Tensor], y: Dict[str, paddle.Tensor]
    ):
        if self.periods:
            x = self.period_emb(x)
        phi_x = self.concat_to_tensor(x, self.input_keys, axis=-1)
        if self.fourier:
            phi_x = self.fourier_emb(phi_x)

        coeffs, residuals, rank, s = np.linalg.lstsq(
            phi_x.numpy(), y[self.output_keys[0]].numpy(), rcond=None
        )

        print("pi_initialization Residuals:", residuals)
        print("pi_initialization Rank:", rank)
        # print("pi_initialization Singular values:", s)
        
        coeffs = paddle.to_tensor(coeffs, dtype=paddle.get_default_dtype())

        self.linear_out.weight.set_value(coeffs)

    def forward(self, x: Dict[str, paddle.Tensor]):
        if self.periods:
            x = self.period_emb(x)

        phi_x = self.concat_to_tensor(x, self.input_keys, axis=-1)

        if self.fourier:
            phi_x = self.fourier_emb(phi_x)

        U = self.act_u(self.linear_U(phi_x))
        V = self.act_v(self.linear_V(phi_x))
        for block in self.residual_blocks:
            phi_x = block(phi_x, U, V)
        y = self.linear_out(phi_x)

        y = self.split_to_dict(y, self.output_keys, axis=-1)

        return y
