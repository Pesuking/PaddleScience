import paddle
import math
import paddle.nn as nn
import numpy as np
from typing import Tuple, Dict


from ppsci.arch import activation as act_mod
from ppsci.arch import base
from ppsci.utils import initializer
from ppsci.utils import logger


# class WeightNormLinear(nn.Layer):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight_v = self.create_parameter((in_features, out_features))
#         self.weight_g = self.create_parameter((out_features,))
#         if bias:
#             self.bias = self.create_parameter((out_features,))
#         else:
#             self.bias = None
#         self._init_weights()

#     def _init_weights(self) -> None:
#         initializer.xavier_uniform_(self.weight_v)
#         initializer.constant_(self.weight_g, 1.0)
#         if self.bias is not None:
#             initializer.constant_(self.bias, 0.0)

#     def forward(self, input):
#         norm = self.weight_v.norm(p=2, axis=0, keepdim=True)
#         weight = self.weight_g * self.weight_v / norm
#         return nn.functional.linear(input, weight, self.bias)


class RandomWeightFactorization(nn.Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mean: float = 0.5,
        std: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_v = self.create_parameter((in_features, out_features))
        self.weight_g = self.create_parameter((out_features,))
        if bias:
            self.bias = self.create_parameter((out_features,))
        else:
            self.bias = None

        self._init_weights(mean, std)

    def _init_weights(self, mean, std):
        with paddle.no_grad():
            # glorot normal
            fin, fout = self.weight_v.shape
            var = 2.0 / (fin + fout)
            stddev = math.sqrt(var) * 0.87962566103423978
            initializer.trunc_normal_(self.weight_v)
            paddle.assign(self.weight_v * stddev, self.weight_v)

            nn.initializer.Normal(mean, std)(self.weight_g)
            paddle.assign(paddle.exp(self.weight_g), self.weight_g)
            paddle.assign(self.weight_v / self.weight_g, self.weight_v)
            if self.bias is not None:
                initializer.constant_(self.bias, 0.0)

        self.weight_g.stop_gradient = False
        self.weight_v.stop_gradient = False
        self.bias.stop_gradient = False

    def forward(self, input):
        return nn.functional.linear(input, self.weight_g * self.weight_v, self.bias)


class PeriodEmbedding(nn.Layer):
    def __init__(self, periods: Dict[str, Tuple[float, bool]]):
        super().__init__()
        self.freqs_dict = {
            k: self.create_parameter(
                [],
                attr=paddle.ParamAttr(trainable=trainable),
                default_initializer=nn.initializer.Constant(2 * np.pi / eval(p)),
            )  # mu = 2*pi / period for sin/cos function
            for k, (p, trainable) in periods.items()
        }
        self.freqs = paddle.nn.ParameterList(list(self.freqs_dict.values()))

    def forward(self, x: Dict[str, paddle.Tensor]):
        y = {k: v for k, v in x.items()}  # shallow copy to avoid modifying input dict

        for k, w in self.freqs_dict.items():
            y[k] = paddle.concat([paddle.cos(w * x[k]), paddle.sin(w * x[k])], axis=-1)

        return y


class FourierEmbedding(nn.Layer):
    def __init__(self, in_features, out_features, scale, trainable=False):
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError(f"out_features must be even, but got {out_features}.")

        self.kernel = self.create_parameter(
            [in_features, out_features // 2],
            attr=paddle.ParamAttr(trainable=trainable),
            default_initializer=nn.initializer.Normal(std=scale),
        )

        self.kernel.trainable = (
            trainable  # This line is added to make the kernel non-trainable
        )

    def forward(self, x: paddle.Tensor):
        y = paddle.concat(
            [
                paddle.cos(x @ self.kernel),
                paddle.sin(x @ self.kernel),
            ],
            axis=-1,
        )
        return y


class CustomLayer(nn.Layer):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = self.create_parameter(
            [1], default_initializer=nn.initializer.Constant(alpha), dtype="float32"
        )

    def forward(self, h, phi_x):
        return self.alpha * h + (1 - self.alpha) * phi_x


class ResidualBlock(nn.Layer):
    def __init__(self, num_hiddens, alpha, activation, mu, sigma):
        super().__init__()
        self.activation = act_mod.get_activation(activation)
        self.dense_1 = RandomWeightFactorization(
            num_hiddens, num_hiddens, True, mu, sigma
        )
        self.dense_2 = RandomWeightFactorization(
            num_hiddens, num_hiddens, True, mu, sigma
        )
        self.dense_3 = RandomWeightFactorization(
            num_hiddens, num_hiddens, True, mu, sigma
        )
        self.addaptive_layer = CustomLayer(alpha)

    def forward(self, phi_x, U, V):
        f = self.activation(self.dense_1(phi_x))
        z_1 = paddle.multiply(f, U) + paddle.multiply(1 - f, V)
        g = self.activation(self.dense_2(z_1))
        z_2 = paddle.multiply(g, U) + paddle.multiply(1 - g, V)
        h = self.dense_3(z_2)
        return self.addaptive_layer(h, phi_x)


class PirateNets(base.Arch):
    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        num_layers: int,
        num_hiddens: int,
        activation: str,
        alpha: float,
        fourier_in_features: int,
        fourier_out_features: int,
        fourier_scale: float,
        fourier_trainable: bool,
        periods: Dict[str, bool],
        mu: float,
        sigma: float,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        # Architecture
        self.num_blocks = num_layers // 3
        self.num_hiddens = num_hiddens
        self.activation = act_mod.get_activation(activation)
        self.fourier_in_features = fourier_in_features
        self.fourier_out_features = fourier_out_features
        self.fourier_scale = fourier_scale
        self.fourier_trainable = fourier_trainable
        self.periods = periods
        # self.mu = mu
        # self.sigma = sigma

        self.period_emb = PeriodEmbedding(self.periods)

        # Network
        self.fourier_emb = FourierEmbedding(
            self.fourier_in_features + len(self.periods),
            self.fourier_out_features,
            self.fourier_scale,
            self.fourier_trainable,
        )
        self.linear_U = RandomWeightFactorization(
            num_hiddens, num_hiddens, True, mu, sigma
        )
        self.linear_V = RandomWeightFactorization(
            num_hiddens, num_hiddens, True, mu, sigma
        )
        self.residual_blocks = nn.LayerList(
            [
                ResidualBlock(
                    self.num_hiddens,
                    alpha,
                    activation,
                    mu,
                    sigma,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.linear_out = nn.Linear(
            self.num_hiddens, len(self.output_keys), bias_attr=False
        )

    def pi_initialization(
        self, x: Dict[str, paddle.Tensor], y: Dict[str, paddle.Tensor]
    ):
        with paddle.no_grad():
            x = self.period_emb(x)
            x = self.concat_to_tensor(x, self.input_keys, axis=-1)
            phi_x = self.fourier_emb(x)
            w = np.linalg.lstsq(
                phi_x.numpy(), y[self.output_keys[0]].numpy(), rcond=None
            )[0]
            print(f"w.shape: {w.shape}")
            # 将numpy.ndarray转换为Tensor
            w = paddle.to_tensor(w, dtype=paddle.get_default_dtype())
            # 使用w初始化linear_out的权重
            print(self.linear_out.weight.shape)
            self.linear_out.weight.set_value(w)

            print(self.linear_out(phi_x) - y[self.output_keys[0]])

    def forward(self, x: Dict[str, paddle.Tensor]):
        # for k, v in x.items():
        #     print(f"Before periods: {k} v.shape: {v.shape}")
        x = self.period_emb(x)
        # for k, v in x.items():
        # print(f"After periods: {k} v.shape: {v.shape}")
        x = self.concat_to_tensor(x, self.input_keys, axis=-1)
        # print(f"After concat_to_tensor: x.shape: {x.shape}")
        phi_x = self.fourier_emb(x)

        U = self.activation(self.linear_U(phi_x))
        V = self.activation(self.linear_V(phi_x))
        for block in self.residual_blocks:
            phi_x = block(phi_x, U, V)
        y = self.linear_out(phi_x)

        y = self.split_to_dict(y, self.output_keys, axis=-1)
        return y
