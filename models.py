import numpy as np
from sklearn.linear_model import LinearRegression

import numpy_ml
from copy import deepcopy

from utility import d_tanh, h_seq_pre_processing, sgn
from figures import show_activity_forward_backward
# import ipdb
import torch
import torch.nn as nn

# np.random.seed(1926)


class CtlRNN(nn.Module):
    """
    EchoState Network
    """
    def __init__(self,
                 n_hidden,
                 n_target,
                 n_input=None,
                 tau=10,
                 enableFA=True):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_target = n_target
        if not n_input:
            self.n_input = n_target
        else:
            self.n_input = n_input

        self.i2h = nn.Linear(n_input, n_hidden)
        for param in self.i2h.parameters():
            param.requires_grad = False
        self.h2h = nn.Linear(n_hidden, n_hidden)
        for param in self.h2h.parameters():
            param.requires_grad = False

        self.h2o = nn.Linear(n_hidden, n_target)
        self.phi = torch.tanh
        self.tau = tau
        self.c = (1 - 1 / self.tau)
        self.d = 1 - self.c

    def forward(self, h, x):
        h = self.c * h + self.d * (self.h2h(self.phi(h)) + self.i2h(x))
        o = self.h2o(self.phi(h))
        return h, o


class R2N2:
    def __init__(
        self,
        n_hidden,
        n_target,
        n_input=None,
        g=1.5,
        tau=10,
        lr=1e-2,
        enableFA=True,
        g_forward=1.5,
        g_backward=1.0,
    ):
        self.n_hidden = n_hidden
        self.n_hA = n_hidden
        self.n_hB = n_hidden
        self.n_target = n_target
        self.g_forward = g_forward
        self.g_backward = g_backward
        if not n_input:
            self.n_input = n_target
        else:
            self.n_input = n_input

        self.tau = tau
        self.c = 1 - 1 / tau
        self.d = 1 / tau

        self.phi = np.tanh
        self.d_phi = d_tanh
        self.enableFA = enableFA

        self.init_ws(g)

        self._reset_forward_samples()

        # self.optimizer = numpy_ml.neural_nets.optimizers.Adam(
        #     lr=lr,
        #     lr_scheduler=numpy_ml.neural_nets.schedulers.ExponentialScheduler(
        #         initial_lr=lr, stage_length=5000))

        self.optimizer = numpy_ml.neural_nets.optimizers.Adam(lr=lr)

    def init_ws(self, g):
        self._init_w_forward(self.g_forward)
        self._init_w_backward(self.g_forward)
        self._init_w_input(self.g_forward)
        self._init_w_output(self.g_forward)
        self._init_w_FA(self.g_backward)

    # def _init_w_FA(self, ):
    #     self.wFA_A2B = np.random.normal(0, 1 / self.n_hA**0.5,
    #                                     (self.n_hA, self.n_hB))

    #     self.wFA_B2A = np.random.normal(0, 1 / self.n_hB**0.5,
    #                                     (self.n_hB, self.n_hA))

    #     self.wFA_B2O = np.random.normal(0, 1 / self.n_hB**0.5,
    #                                     (self.n_hB, self.n_target))

    # def _init_w_FA(self, g):
    #     self.wFA_A2B = g * np.random.normal(
    #         0, 1, (self.n_hA, self.n_hB)) / self.n_hB**0.5

    #     self.wFA_B2A = g * np.random.normal(
    #         0, 1, (self.n_hB, self.n_hA)) / self.n_hA**0.5

    #     self.wFA_B2O = g * np.random.normal(
    #         0, 1, (self.n_hB, self.n_target)) / self.n_hA**0.5

    def _init_w_FA(self, g):
        # test 2021.04.12
        self.wFA_A2B = g * np.random.normal(0, 1 / self.n_hB**0.5,
                                            (self.n_hA, self.n_hB))

        self.wFA_B2A = g * np.random.normal(0, 1 / self.n_hA**0.5,
                                            (self.n_hB, self.n_hA))

        self.wFA_B2O = g * np.random.normal(0, 1 / self.n_target**0.5,
                                            (self.n_hB, self.n_target))

    def hard_reset_w_backward(self):
        self.w_A2B_b = -self.w_A2B_f.copy()
        self.w_B2A_b = -self.w_B2A_f.copy()

        self._reset_forward_samples()

    def _reset_forward_samples(self):
        self.origin_for_As = []
        self.current_to_As = []

        self.origin_for_Bs = []
        self.current_to_Bs = []

    def solve_w_backward(self, ):
        # for group A:
        XA = np.array(self.origin_for_As)
        yA = -np.array(self.current_to_As)
        regA = LinearRegression().fit(XA, yA)
        self.w_B2A_b = regA.coef_.copy()

        # for group B:
        XB = np.array(self.origin_for_Bs)
        yB = -np.array(self.current_to_Bs)
        regB = LinearRegression().fit(XB, yB)
        self.w_A2B_b = regB.coef_.copy()

        self._reset_forward_samples()

    def _add_forward_samples(self, oA, cA, oB, cB):
        self.origin_for_As.append(oA)
        self.current_to_As.append(cA)

        self.origin_for_Bs.append(oB)
        self.current_to_Bs.append(cB)

    def _init_w_forward(self, g):
        self.w_A2B_f = g * np.random.normal(0, 1 / self.n_hB**0.5,
                                            (self.n_hB, self.n_hA))
        self.w_B2A_f = g * np.random.normal(0, 1 / self.n_hA**0.5,
                                            (self.n_hA, self.n_hB))

    def _init_w_input(self, g):
        """Input is sent to group A"""
        self.w_I2A = 0.1 * np.random.normal(0, 1 / self.n_hA**0.5,
                                            (self.n_hA, self.n_input))

    def _init_w_output(self, g):
        """Output is from group B"""
        self.w_B2O = 0.1 * np.random.normal(0, 1 / self.n_target**0.5,
                                            (self.n_target, self.n_hB))

    def _init_w_backward(self, g):
        self.w_A2B_b = g * np.random.normal(0, 1 / self.n_hB**0.5,
                                            (self.n_hB, self.n_hA))
        self.w_B2A_b = g * np.random.normal(0, 1 / self.n_hA**0.5,
                                            (self.n_hA, self.n_hB))

    def zero_grads(self, ):
        self.dw_B2O = np.zeros_like(self.w_B2O)
        self.dw_I2A = np.zeros_like(self.w_I2A)
        self.dw_A2B = np.zeros_like(self.w_A2B_f)
        self.dw_B2A = np.zeros_like(self.w_B2A_f)

    def apply_grads_naive(self, t_max, lr=1e-2):
        self.w_B2O -= lr * self.dw_B2O / t_max
        self.w_I2A -= lr * self.dw_I2A / t_max
        self.w_A2B_f -= lr * self.dw_A2B / t_max
        self.w_B2A_f -= lr * self.dw_B2A / t_max

    def apply_grads_Adam(self, t_max, scaling_out_lr=1.0):
        self.w_B2O = self.optimizer.update(
            self.w_B2O, scaling_out_lr * self.dw_B2O / t_max, "w_B2O")
        self.w_I2A = self.optimizer.update(self.w_I2A, self.dw_I2A / t_max,
                                           "w_I2A")
        self.w_A2B_f = self.optimizer.update(self.w_A2B_f, self.dw_A2B / t_max,
                                             "w_A2B")
        self.w_B2A_f = self.optimizer.update(self.w_B2A_f, self.dw_B2A / t_max,
                                             "w_B2A")

    def forward_update(self, hs, x, collect_samples=True):
        """
        Args:
            hs: (h_A, h_B)  h_A/B is a numpy array with shape (n_hA/B,)
            x: A numpy array with shape (n_target,)
            collect_samples: to turn on the learning of backward connectivity or not.
        Returns:
            (h_A_next, h_B_next): updated hidden states 
            o: output
        """
        h_A, h_B = hs

        origin_for_A = self.phi(h_B)
        current_to_A = self.w_B2A_f.dot(origin_for_A)

        h_A_next = self.c * h_A + self.d * (current_to_A + self.w_I2A.dot(x))

        origin_for_B = self.phi(h_A_next)
        current_to_B = self.w_A2B_f.dot(origin_for_B)

        h_B_next = self.c * h_B + self.d * (current_to_B)
        o = self.w_B2O.dot(self.phi(h_B_next))

        if collect_samples:
            self._add_forward_samples(origin_for_A, current_to_A, origin_for_B,
                                      current_to_B)

        return (h_A_next, h_B_next), o

    def backward_update(self, hs, x):
        """
        Args:
            hs: hidden activity 
            x: input,
        """
        # ipdb.set_trace()
        h_A, h_B = hs

        o = self.w_B2O.dot(self.phi(h_B))
        # we test backward function with substraction opeartion first
        h_B_prev = 1 / self.c * h_B + self.d / self.c * (self.w_A2B_b.dot(
            self.phi(h_A)))
        h_A_prev = 1 / self.c * h_A + self.d / self.c * self.w_B2A_b.dot(
            self.phi(h_B_prev)) - self.d / self.c * self.w_I2A.dot(x)

        return (h_A_prev, h_B_prev), o

    def forward(self, target_seq):
        """
        Args:
            target_seq: a numpy array with shape (t_max, n_target)

        """
        t_max = target_seq.shape[0]
        th = 5
        std = 3
        hs = (
            np.clip(np.random.normal(0, std, self.n_hA), -th, th),
            np.clip(np.random.normal(0, std, self.n_hB), -th, th),
        )
        hs_seq = [hs]
        output_seq = []
        for t in range(t_max):
            hs, o = self.forward_update(hs, target_seq[t])
            # print(f"t = {t}: A:{hs[0]}, B:{hs[1]}")
            hs_seq.append(deepcopy(hs))
            output_seq.append(deepcopy(o))

        return hs_seq, output_seq

    def backward_replay(self, hs_last, xs):
        """
        Simple demo of the capability of backward running.

        Args:
            hs_last: the last state of the RNN (h_A, h_B)  h_A/B is a numpy array with shape (n_hA/B,)
            xs: input history 
        Returns:
            h_backward_seq: a list of activations in the backward running
        """
        # hs_last = (None, None)
        hs = hs_last
        hs_seq = [hs]
        t_max = xs.shape[0]
        for t in range(t_max):
            hs, _ = self.backward_update(hs, xs[-(t + 1)])
            hs_seq.append(deepcopy(hs))

        return hs_seq

    def forward_seq_prediction(self, target_seq, f_loss):
        """
        For a given sequence x, the RNN use x_t combined with h_t to predict x_{t+1}.
        Args:
            target_seq: a numpy array with shape (t_max, n_target)
            f_loss: a function to compute the single step prediction loss. Can be MSE or cross entropy, etc.
        Return:
            hs_seq: hidden unit activations
            output_seq: output generated by RNN
            loss: Loss for the given target sequence (averaged by time)
        """
        t_max = target_seq.shape[0] - 1
        th = 0.2
        std = 3
        hs = (
            np.clip(np.random.normal(0, std, self.n_hA), -th, th),
            np.clip(np.random.normal(0, std, self.n_hB), -th, th),
        )

        # hs = (
        #     th * np.ones(self.n_hA),
        #     th * np.ones(self.n_hB),
        # )
        hs_seq = [hs]
        output_seq = []
        loss_t_seq = []
        for t in range(t_max):
            hs, o = self.forward_update(hs, target_seq[t])
            # print(f"prediction at {t} is: {o} | new h is {hs}")
            loss_t = f_loss(o, target_seq[t + 1])

            hs_seq.append(deepcopy(hs))
            output_seq.append(deepcopy(o))

            loss_t_seq.append(loss_t)

        return hs_seq, output_seq, np.mean(loss_t_seq)

    def forward_seq_decision_making(self, input_seq, target, f_loss):
        """
        For a given sequence x, the RNN generates output after sequentially received all inputs.
        Args:
            input_seq: a numpy array with shape (t_max, n_target)
            target: desired output at last time step.
            f_loss: a function to compute the single step prediction loss. Can be MSE or cross entropy, etc.
        Return:
            hs_seq: hidden unit activations
            output_seq: output generated by RNN
            loss: Loss for the given target sequence (averaged by time)
        """
        t_max = input_seq.shape[0]
        th = 0.2
        # std = 3

        hs = (
            th * np.ones(self.n_hA),
            th * np.ones(self.n_hB),
        )
        hs_seq = [hs]
        output_seq = []
        for t in range(t_max):
            hs, o = self.forward_update(hs, input_seq[t])
            # print(f"prediction at {t} is: {o} | new h is {hs}")
            hs_seq.append(deepcopy(hs))
            output_seq.append(deepcopy(o))

        # only make a decision at the last time step
        loss_t = f_loss(o, target)

        return hs_seq, output_seq, loss_t

    def backward_learning_decision_making(self, hs_last, xs, target, d_f_loss):
        """
        BPTT-like propagation of gradients for sequence predictions.
        The input is a sequence at time t while the target is sequence at time t+1.

        Args:
            hs_last: the last state of the RNN (h_A, h_B)  h_A/B is a numpy array with shape (n_hA/B,)
            xs: input history
            target: desired output.
            d_f_loss: derivative of loss function.
        Returns:
            h_backward_seq: a list of activations in the backward running
        """
        hs = hs_last
        hs_seq = [hs]
        t_max = xs.shape[0]
        # to start from 1 means the last element means it does not particapte in the forward computation.
        grad_L_h_A = np.zeros_like(hs[0])
        grad_L_h_B = np.zeros_like(hs[1])
        grad_L_o = None

        self.zero_grads()

        if self.enableFA:
            w_B2O_back = self.wFA_B2O
            grads_iterate = self._backward_grad_propagation_FA
        else:
            w_B2O_back = self.w_B2O.T
            grads_iterate = self._backward_grad_propagation_BP

        for t in range(t_max):
            inputs = xs[-(t + 1)]
            # ipdb.set_trace()

            hs, output = self.backward_update(hs, inputs)
            hs_seq.append(deepcopy(hs))
            hs_later = hs_seq[-2]
            hs_prev = hs_seq[-1]

            h_A_later, h_B_later = hs_later
            h_A_prev, h_B_prev = hs_prev

            # output to hidden part
            if t == 0:
                # only in the last time step will the RNN generate a classification result.
                grad_L_o = d_f_loss(output, target)
            else:
                grad_L_o = np.zeros_like(output)

            grad_L_h_B_tmp = w_B2O_back.dot(grad_L_o) * self.d_phi(h_B_later)
            grad_L_h_A_prev, grad_L_h_B_prev, grad_L_h_A, grad_L_h_B = grads_iterate(
                hs_prev, hs_later, inputs, grad_L_h_A, grad_L_h_B,
                grad_L_h_B_tmp)

            dw_B2O = grad_L_o.reshape(-1, 1).dot(
                self.phi(h_B_later.reshape(-1, 1).T))
            self.dw_B2O += dw_B2O

            dw_A2B = self.d * grad_L_h_B.reshape(-1, 1).dot(
                self.phi(h_A_later).reshape(-1, 1).T)
            self.dw_A2B += dw_A2B

            dw_B2A = self.d * grad_L_h_A.reshape(-1, 1).dot(
                self.phi(h_B_prev).reshape(-1, 1).T)
            self.dw_B2A += dw_B2A

            dw_I2A = self.d * grad_L_h_A.reshape(-1, 1).dot(
                inputs.reshape(-1, 1).T)
            self.dw_I2A += dw_I2A

            # grad_L_h_A_prev = grad_L_h_A
            # grad_L_h_B_prev = grad_L_h_B
            # ipdb.set_trace()

        return hs_seq

    def backward_learning_seq_prediction(self,
                                         hs_last,
                                         xs,
                                         d_f_loss,
                                         t_max=None):
        """
        BPTT-like propagation of gradients for sequence predictions.
        The input is a sequence at time t while the target is sequence at time t+1.

        Args:
            hs_last: the last state of the RNN (h_A, h_B)  h_A/B is a numpy array with shape (n_hA/B,)
            xs: input history
            d_f_loss: derivative of loss function.
            t_max: time step for temporal propagation
        Returns:
            h_backward_seq: a list of activations in the backward running
        """
        hs = hs_last
        hs_seq = [hs]
        if t_max is None:
            t_max = xs.shape[0]
        # to start from 1 means the last element means it does not particapte in the forward computation.
        grad_L_h_A = np.zeros_like(hs[0])
        grad_L_h_B = np.zeros_like(hs[1])
        grad_L_o = None

        self.zero_grads()

        if self.enableFA:
            w_B2O_back = self.wFA_B2O
            grads_iterate = self._backward_grad_propagation_FA
        else:
            w_B2O_back = self.w_B2O.T
            grads_iterate = self._backward_grad_propagation_BP

        for t in range(1, t_max):
            target = xs[-t]
            inputs = xs[-(t + 1)]
            # ipdb.set_trace()

            hs, output = self.backward_update(hs, inputs)
            hs_seq.append(deepcopy(hs))
            hs_later = hs_seq[-2]
            hs_prev = hs_seq[-1]

            h_A_later, h_B_later = hs_later
            h_A_prev, h_B_prev = hs_prev

            # output to hidden part
            grad_L_o = d_f_loss(output, target)

            grad_L_h_B_tmp = w_B2O_back.dot(grad_L_o) * self.d_phi(h_B_later)
            grad_L_h_A_prev, grad_L_h_B_prev, grad_L_h_A, grad_L_h_B = grads_iterate(
                hs_prev, hs_later, inputs, grad_L_h_A, grad_L_h_B,
                grad_L_h_B_tmp)

            dw_B2O = grad_L_o.reshape(-1, 1).dot(
                self.phi(h_B_later.reshape(-1, 1).T))
            self.dw_B2O += dw_B2O

            dw_A2B = self.d * grad_L_h_B.reshape(-1, 1).dot(
                self.phi(h_A_later).reshape(-1, 1).T)
            self.dw_A2B += dw_A2B

            dw_B2A = self.d * grad_L_h_A.reshape(-1, 1).dot(
                self.phi(h_B_prev).reshape(-1, 1).T)
            self.dw_B2A += dw_B2A

            dw_I2A = self.d * grad_L_h_A.reshape(-1, 1).dot(
                inputs.reshape(-1, 1).T)
            self.dw_I2A += dw_I2A

            grad_L_h_A_prev = grad_L_h_A
            grad_L_h_B_prev = grad_L_h_B
            # ipdb.set_trace()

        return hs_seq

    def _backward_grad_propagation_BP(self, hs_prev, hs_later, x_prev,
                                      grad_hA_later, grad_hB_later, grad_O2hB):
        """
        Complete and precise one step backpropagation.
        """
        h_A_later, h_B_later = hs_later
        h_A_prev, h_B_prev = hs_prev

        grad_hB_later = grad_hB_later + grad_O2hB
        grad_hA_later = grad_hA_later + self.d * self.w_A2B_f.T.dot(
            grad_hB_later) * self.d_phi(h_A_later)

        grad_hA_prev = self.c * grad_hA_later

        grad_hB_prev = self.c * grad_hB_later + self.d * self.w_B2A_f.T.dot(
            grad_hA_later) * self.d_phi(h_B_prev)

        return grad_hA_prev, grad_hB_prev, grad_hA_later, grad_hB_later

    def _backward_grad_propagation_FA(self, hs_prev, hs_later, x_prev,
                                      grad_hA_later, grad_hB_later, grad_O2hB):
        """
        Complete and precise one step backpropagation.
        """
        h_A_later, h_B_later = hs_later
        h_A_prev, h_B_prev = hs_prev

        grad_hB_later = grad_hB_later + grad_O2hB
        grad_hA_later = grad_hA_later + self.d * self.wFA_A2B.dot(
            grad_hB_later) * self.d_phi(h_A_later)

        grad_hA_prev = self.c * grad_hA_later

        grad_hB_prev = self.c * grad_hB_later + self.d * self.wFA_B2A.dot(
            grad_hA_later) * self.d_phi(h_B_prev)

        return grad_hA_prev, grad_hB_prev, grad_hA_later, grad_hB_later


class HFRNN:
    def __init__(self, N, g=1.2, tau=6):
        self.n_h = N
        self.w_1 = np.zeros(
            (N, N)) + g * np.random.normal(0, 1 / N**0.5, (N, N))
        self.w_2 = np.zeros(
            (N, N)) + g * np.random.normal(0, 1 / N**0.5, (N, N))

        self.tau = tau
        self.phi = sgn

    def forward(self, h, lba):
        w = lba * self.w_2 + (1 - lba) * self.w_1
        h_new = (1 - 1 / self.tau) * h + (1 / self.tau) * w.dot(self.phi(h))

        o = self.phi(h_new)
        return h_new, o

    def associate(self, x_pre, x_post, lba):
        if lba is True:
            self.w_2 += np.outer(x_post, x_pre + x_post)
        else:
            self.w_1 += np.outer(x_post, x_pre + x_post)

    def learn_sequence(self, seq):
        """seq:(T, n_neurons)
        """
        seq = seq.copy()
        rho = np.sum(seq) / (len(seq) * self.n_h)

        lba = True
        for t in range(len(seq) - 1):
            p_pre = seq[t] - rho
            p_post = seq[t + 1] - rho

            # self.associate(p_pre, p_post, lba)
            self.associate(p_post, p_pre, lba)
            lba = not lba

        self.w_2 /= self.n_h
        self.w_1 /= self.n_h


if __name__ == "__main__":
    n_x = 1
    n_h = 64
    model = R2N2(n_h, n_x, g=2.5, tau=10)
    T = 100
    for c in range(4):
        xs = np.random.rand(T, n_x) * 0.5
        h_forward_seq, output_seq = model.forward(xs)
        model.solve_w_backward()

    # model.hard_reset_w_backward()
    h_backward_seq = model.backward_replay(h_forward_seq[-1], xs)

    hs_A_f, hs_B_f = h_seq_pre_processing(h_forward_seq)
    hs_A_b, hs_B_b = h_seq_pre_processing(h_backward_seq)
    show_activity_forward_backward(hs_A_f, hs_B_f, hs_A_b, hs_B_b)
