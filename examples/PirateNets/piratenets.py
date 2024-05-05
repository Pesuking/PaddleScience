from omegaconf import DictConfig
import hydra
import numpy as np
import paddle
import paddle.nn as nn
from typing import Tuple

import scipy.io as sio
import ppsci
from matplotlib import pyplot as plt
from os import path as osp
from ppsci.utils import misc

dtype = paddle.get_default_dtype()


def plot(
    t_star: np.ndarray,
    x_star: np.ndarray,
    u_ref: np.ndarray,
    u_pred: np.ndarray,
    output_dir: str,
):
    fig = plt.figure(figsize=(18, 5))
    TT, XX = np.meshgrid(t_star, x_star, indexing="ij")
    u_ref = u_ref.reshape([len(t_star), len(x_star)])

    plt.subplot(1, 3, 1)
    plt.pcolor(TT, XX, u_ref, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Exact")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(TT, XX, u_pred, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Predicted")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(TT, XX, np.abs(u_ref - u_pred), cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Absolute error")
    plt.tight_layout()

    fig_path = osp.join(output_dir, "ac.png")
    print(f"Saving figure to {fig_path}")
    fig.savefig(fig_path, bbox_inches="tight", dpi=400)
    plt.close()


class Train:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # weight settings
        self.num_chunks = cfg.WEIGHT.num_chunks
        self.casual_tolerance = cfg.WEIGHT.casual_tolerance
        self.weighting_scheme = cfg.WEIGHT.weighting_scheme

        self.equation = {"AllenCahn": ppsci.equation.AllenCahn(0.01**2)}

        self.data = sio.loadmat(cfg.DATA_PATH)
        self.u_ref = self.data["usol"].astype(dtype)  # (nt, nx)
        self.t_star = self.data["t"].flatten().astype(dtype)  # [nt, ]
        self.x_star = self.data["x"].flatten().astype(dtype)  # [nx, ]
        self.u0 = self.u_ref[0, :]  # [nx, ]
        self.t0 = self.t_star[0]  # float
        self.t1 = self.t_star[-1]  # float
        self.x0 = self.x_star[0]  # float
        self.x1 = self.x_star[-1]  # float

        # set constraint
        def gen_input_batch():
            tx = np.random.uniform(
                [self.t0, self.x0],
                [self.t1, self.x1],
                (cfg.TRAIN.batch_size, 2),
            ).astype(dtype)
            return {
                "t": tx[:, 0:1],
                "x": tx[:, 1:2],
            }

        def gen_label_batch(input_batch):
            return {"allen_cahn": np.zeros([cfg.TRAIN.batch_size, 1], dtype)}

        print(type(dtype), dtype)

        self.pde_constraint = ppsci.constraint.SupervisedConstraint(
            {
                "dataset": {
                    "name": "ContinuousNamedArrayDataset",
                    "input": gen_input_batch,
                    "label": gen_label_batch,
                },
            },
            output_expr=self.equation["AllenCahn"].equations,
            loss=ppsci.loss.CausalMSELoss(
                self.num_chunks, "mean", tol=self.casual_tolerance
            ),
            name="Sub",
        )

        self.ic_input = {
            "t": np.full([len(self.x_star), 1], self.t0),
            "x": self.x_star.reshape([-1, 1]),
        }

        self.ic_label = {"u": self.u0.reshape([-1, 1])}
        self.ic = ppsci.constraint.SupervisedConstraint(
            {
                "dataset": {
                    "name": "IterableNamedArrayDataset",
                    "input": self.ic_input,
                    "label": self.ic_label,
                },
            },
            output_expr={"u": lambda out: out["u"]},
            loss=ppsci.loss.MSELoss("mean"),
            name="IC",
        )

        # wrap constraints together
        self.constraint = {
            self.pde_constraint.name: self.pde_constraint,
            self.ic.name: self.ic,
        }

        # set model
        self.model = ppsci.arch.PirateNets(**cfg.MODEL)

        # In cases where the data comes from an initial condition u0(x), we initialize the weights of the last layer to fit uθ(t, x) to u0(x)for all t.
        self.model.pi_initialization(
            {
                "t": paddle.to_tensor(
                    np.full([len(self.x_star), 1], self.t0), dtype=dtype
                ),
                "x": paddle.to_tensor(self.x_star.reshape([-1, 1]), dtype=dtype),
            },
            {"u": paddle.to_tensor(self.u0.reshape([-1, 1]), dtype=dtype)},
        )

        print(
            self.model(
                {
                    "t": paddle.to_tensor(
                        np.full([len(self.x_star), 1], self.t0), dtype=dtype
                    ),
                    "x": paddle.to_tensor(self.x_star.reshape([-1, 1]), dtype=dtype),
                }
            )["u"]
            - self.ic_label["u"]
        )

        # set optimizer
        self.lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
            **cfg.TRAIN.lr_scheduler
        )()
        self.optimizer = ppsci.optimizer.Adam(self.lr_scheduler)(self.model)

        # set validator
        self.tx_star = misc.cartesian_product(self.t_star, self.x_star).astype(dtype)
        self.eval_data = {"t": self.tx_star[:, 0:1], "x": self.tx_star[:, 1:2]}
        self.eval_label = {"u": self.u_ref.reshape([-1, 1])}
        u_validator = ppsci.validate.SupervisedValidator(
            {
                "dataset": {
                    "name": "NamedArrayDataset",
                    "input": self.eval_data,
                    "label": self.eval_label,
                },
                "batch_size": cfg.EVAL.batch_size,
            },
            ppsci.loss.MSELoss("mean"),
            {"u": lambda out: out["u"]},
            metric={"L2Rel": ppsci.metric.L2Rel()},
            name="u_validator",
        )
        self.validator = {u_validator.name: u_validator}

        # initialize solver
        self.solver = ppsci.solver.Solver(
            self.model,
            self.constraint,
            self.cfg.output_dir,
            self.optimizer,
            self.cfg.TRAIN.epochs,
            self.cfg.TRAIN.iters_per_epoch,
            save_freq=self.cfg.TRAIN.save_freq,
            log_freq=self.cfg.log_freq,
            eval_during_train=True,
            eval_freq=self.cfg.TRAIN.eval_freq,
            seed=self.cfg.seed,
            equation=self.equation,
            validator=self.validator,
            pretrained_model_path=self.cfg.TRAIN.pretrained_model_path,
            checkpoint_path=self.cfg.TRAIN.checkpoint_path,
            eval_with_no_grad=self.cfg.EVAL.eval_with_no_grad,
            use_tbd=True,
            cfg=self.cfg,
        )

    # def u_net(self, t, x):
    #     return

    # def r_net(self, t, x):
    #     t.stop_gradient = False
    #     x.stop_gradient = False
    #     z = paddle.concat([t, x], axis=-1)
    #     u = self.model(z)
    #     u_t = paddle.grad(outputs=u, inputs=t, create_graph=True)
    #     u_x = paddle.grad(outputs=u, inputs=x, create_graph=True)
    #     u_xx = paddle.grad(outputs=u_x, inputs=x, create_graph=True)
    #     return u_t + 5 * u**3 - 5 * u - 0.0001 * u_xx

    # def losses(self, batch, *args):
    #     t0 = paddle.full_like(self.x_star, self.t0)
    #     z = paddle.concat([t0, self.x_star], axis=-1)
    #     u0_pred = self.model(z)
    #     u_ref = self.u_ref[0, :]
    #     ics_loss = paddle.mean((u0_pred - u_ref) ** 2)

    #     # use causal
    #     indices = paddle.argsort(batch[:, 0])
    #     batch_sorted = paddle.gather(batch, indices, axis=0)
    #     u_pred = self.model(batch_sorted)
    #     # u_pred = paddle.reshape(u_pred, self.num_chunks, -1)
    #     u_pred = u_pred.reshape(self.num_chunks, -1)
    #     l = paddle.mean(u_pred**2, axis=1)
    #     w = paddle.exp(-self.casual_tolerance * (paddle.matmul(self.M, l)))
    #     res_loss = paddle.mean(l * w)
    #     # w = paddle.exp(-self.tol * (paddle.matmul(self.M, l.unsqueeze(-1)))).squeeze(-1)
    #     return ics_loss + res_loss

    # def compute_diag_ntk(params, barch, *args):
    #     raise NotImplementedError()

    # def loss(params, weights, batch, *args):
    #     pass

    # def compute_weights(params, batch, *args):
    #     pass

    # def compute_weights(params, batch, *args):
    #     pass

    # def update_weights(state, batch, *args):
    #     pass

    # def step(state, batch, *args):
    #     pass

    def train(self):
        # train model
        self.solver.train()
        # model = ppsci.arch.PirateNets(**cfg.MODEL) # set model

        # data = sio.loadmat(cfg.DATA_PATH)
        # u_ref = data["usol"].astype(dtype)  # (nt, nx)
        # t_star = data["t"].flatten().astype(dtype)  # [nt, ]
        # x_star = data["x"].flatten().astype(dtype)  # [nx, ]

        # print(f"u_ref.shape: {u_ref.shape}")
        # print(f"t_star.shape: {t_star.shape}")
        # print(f"x_star.shape: {x_star.shape}")

        # u0 = u_ref[0, :]  # [nx, ]

        # t0 = t_star[0]  # float
        # t1 = t_star[-1]  # float

        # x0 = x_star[0]  # float
        # x1 = x_star[-1]  # float

        # print(f"u0.shape: {u0.shape}")
        # print(f"t0: {t0}")
        # print(f"t1: {t1}")
        # print(f"x0: {x0}")
        # print(f"x1: {x1}")

        # batch = gen_input_batch(t0, t1, x0, x1, batch_size=64)
        # print(f"batch: {batch}")

        # lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        #     epochs=1000,
        #     iterations_per_epoch=100,
        #     learning_rate=0.001,
        #     gamma=0.5,
        #     decay_steps=1000,
        #     by_epoch=False,
        # )

    #


@hydra.main(version_base=None, config_path="./conf", config_name="piratenets.yaml")
def main(cfg: DictConfig):
    train = Train(cfg)
    train.train()


if __name__ == "__main__":
    main()
