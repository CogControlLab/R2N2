from pylab import *

import numpy as np


def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


def theta(x):
    return 0.5 * (1 + sign(x))


def f(x):
    return np.tanh(x)


def df(x):
    return 1 / np.cosh(10 * np.tanh(x / 10))**2  # the tanh prevents oveflow


class RfloRNN:
    '''
    A recurrent neural network.
    Adopted from James Murray's code for RFLO paper.

    Parameters:
    ----------
    n_in, n_rec, n_out : number of input, recurrent, and hidden units.

    h0 : The initial state vector of the RNN.

    tau_m : The network time constant, in units of timesteps.
    '''
    def __init__(self, n_in, n_rec, n_out, h0, tau_m=10):
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.h0 = h0
        self.tau_m = tau_m

        # Initialize weights:
        self.w_in = 0.1 * (np.random.rand(n_rec, n_in) - 1)
        self.w_rec = 1.5 * np.random.randn(n_rec, n_rec) / n_rec**0.5
        self.w_out = 0.1 * (2 * np.random.rand(n_out, n_rec) - 1) / n_rec**0.5

        # Random error feedback matrix:
        self.b = np.random.randn(n_rec, n_out) / n_out**0.5

    def run_trial(self,
                  x,
                  y_,
                  eta=[0.1, 0.1, 0.1],
                  learning=None,
                  online_learning=False):
        '''
        Run the RNN for a single trial.

        Parameters:
        -----------
        x : The input to the network. x[t,i] is input from unit i at timestep t.

        y_ : The target RNN output, where y_[t,i] is output i at timestep t.

        eta : A list of 3 learning rates, for w_in, w_rec, and w_out,
            respectively.

        learning : Specify the learning algorithm with one of the following
            strings: 'rtrl', 'bptt', or 'rflo'. If None, run the network without
            learning.

        online_learning : If True (and learning is on), update weights at each
            timestep. If False (and learning is on), update weights only at the
            end of each trial. Online learning cannot be used with BPTT.

        Returns:
        --------
        y : The time-dependent network output. y[t,i] is output i at timestep t.

        h : The time-dependent RNN state vector. h[t,i] is unit i at timestep t.

        u : The inputs to RNN units (feedforward plus recurrent) at each
            timestep.
        '''

        # Boolean shorthands to specify learning algorithm:
        rtrl = learning == 'rtrl'
        bptt = learning == 'bptt'
        rflo = learning == 'rflo'

        [eta3, eta2, eta1] = eta  # learning rates for w_in, w_rec, and w_out
        t_max = np.shape(x)[0]  # number of timesteps

        dw_in, dw_rec, dw_out = 0, 0, 0  # changes to weights

        u = np.zeros((t_max, self.n_rec))  # input (feedforward plus recurrent)
        h = np.zeros((t_max, self.n_rec))  # time-dependent RNN activity vector
        h[0] = self.h0  # initial state
        y = np.zeros((t_max, self.n_out))  # RNN output
        err = np.zeros((t_max, self.n_out))  # readout error

        # If rflo, eligibility traces p and q should have rank 2; if rtrl, rank 3:
        if rtrl:
            p = np.zeros((self.n_rec, self.n_rec, self.n_rec))
            q = np.zeros((self.n_rec, self.n_rec, self.n_in))
        elif rflo:
            p = np.zeros((self.n_rec, self.n_rec))
            q = np.zeros((self.n_rec, self.n_in))

        for jj in range(self.n_rec):
            if rtrl:
                q[jj, jj, :] = df(u[0, jj]) * x[0, :] / self.tau_m
            elif rflo:
                q[jj, :] = df(u[0, jj]) * x[0, :] / self.tau_m

        for tt in range(t_max - 1):
            u[tt +
              1] = np.dot(self.w_rec, h[tt]) + np.dot(self.w_in, x[tt + 1])
            h[tt + 1] = h[tt] + (-h[tt] + f(u[tt + 1])) / self.tau_m
            y[tt + 1] = np.dot(self.w_out, h[tt + 1])
            err[tt + 1] = y_[tt + 1] - y[tt + 1]  # readout error

            if rflo:
                p = (1 - 1 / self.tau_m) * p
                q = (1 - 1 / self.tau_m) * q
                p += np.outer(df(u[tt + 1, :]), h[tt, :]) / self.tau_m
                q += np.outer(df(u[tt + 1, :]), x[tt, :]) / self.tau_m
            elif rtrl:
                p = np.tensordot((1 - 1 / self.tau_m) * np.eye(self.n_rec) +
                                 df(u[tt + 1]) * self.w_rec / self.tau_m,
                                 p,
                                 axes=1)
                q = np.tensordot((1 - 1 / self.tau_m) * np.eye(self.n_rec) +
                                 df(u[tt + 1]) * self.w_rec / self.tau_m,
                                 q,
                                 axes=1)
                for jj in range(self.n_rec):
                    p[jj, jj, :] += df(u[tt + 1, jj]) * h[tt] / self.tau_m
                    q[jj, jj, :] += df(u[tt + 1, jj]) * x[tt + 1] / self.tau_m

            if rflo and online_learning:
                dw_out = eta1 / t_max * np.outer(err[tt + 1], h[tt + 1])
                dw_rec = eta2 * np.outer(np.dot(self.b, err[tt + 1]),
                                         np.ones(self.n_rec)) * p / t_max
                dw_in = eta3 * np.outer(np.dot(self.b, err[tt + 1]),
                                        np.ones(self.n_in)) * q / t_max
            elif rflo and not online_learning:
                dw_out += eta1 / t_max * np.outer(err[tt + 1], h[tt + 1])
                dw_rec += eta2 * np.outer(np.dot(self.b, err[tt + 1]),
                                          np.ones(self.n_rec)) * p / t_max
                dw_in += eta3 * np.outer(np.dot(self.b, err[tt + 1]),
                                         np.ones(self.n_in)) * q / t_max
            elif rtrl and online_learning:
                dw_out = eta1 / t_max * np.outer(err[tt + 1], h[tt + 1])
                dw_rec = eta2 / t_max * np.tensordot(
                    np.dot(err[tt + 1], self.w_out), p, axes=1)
                dw_in = eta3 / t_max * np.tensordot(
                    np.dot(err[tt + 1], self.w_out), q, axes=1)
            elif rtrl and not online_learning:
                dw_out += eta1 / t_max * np.outer(err[tt + 1], h[tt + 1])
                dw_rec += eta2 / t_max * np.tensordot(
                    np.dot(err[tt + 1], self.w_out), p, axes=1)
                dw_in += eta3 / t_max * np.tensordot(
                    np.dot(err[tt + 1], self.w_out), q, axes=1)

            if online_learning and not bptt:
                self.w_out = self.w_out + dw_out
                self.w_rec = self.w_rec + dw_rec
                self.w_in = self.w_in + dw_in

        if bptt:  # backward pass for BPTT
            z = np.zeros((t_max, self.n_rec))
            z[-1] = np.dot((self.w_out).T, err[-1])
            for tt in range(t_max - 1, 0, -1):
                z[tt - 1] = z[tt] * (1 - 1 / self.tau_m)
                z[tt - 1] += np.dot((self.w_out).T, err[tt])
                z[tt - 1] += np.dot(z[tt] * df(u[tt]), self.w_rec) / self.tau_m

                # Updates for the weights:
                dw_out += eta1 * np.outer(err[tt], h[tt]) / t_max
                dw_rec += eta2 / (t_max * self.tau_m) * np.outer(
                    z[tt] * df(u[tt]), h[tt - 1])
                dw_in += eta3 / (t_max * self.tau_m) * np.outer(
                    z[tt] * df(u[tt]), x[tt])

        if not online_learning:  # wait until end of trial to update weights
            self.w_out = self.w_out + dw_out
            self.w_rec = self.w_rec + dw_rec
            self.w_in = self.w_in + dw_in

        return y, h, u

    def run_trial_seq_classification(self,
                                     x,
                                     target,
                                     eta=[0.1, 0.1, 0.1],
                                     learning=None,
                                     online_learning=False):
        '''
        Run the RNN for a single trial for sequential classification.

        Parameters:
        -----------
        x : The input to the network. x[t,i] is input from unit i at timestep t.

        target (int) : The target RNN output class label

        eta : A list of 3 learning rates, for w_in, w_rec, and w_out,
            respectively.

        learning : Specify the learning algorithm with one of the following
            strings: 'rtrl', 'bptt', or 'rflo'. If None, run the network without
            learning.

        online_learning : If True (and learning is on), update weights at each
            timestep. If False (and learning is on), update weights only at the
            end of each trial. Online learning cannot be used with BPTT.

        '''

        # Boolean shorthands to specify learning algorithm:
        rtrl = learning == 'rtrl'
        bptt = learning == 'bptt'
        rflo = learning == 'rflo'

        [eta3, eta2, eta1] = eta  # learning rates for w_in, w_rec, and w_out
        t_max = np.shape(x)[0]  # number of timesteps

        dw_in, dw_rec, dw_out = 0, 0, 0  # changes to weights

        u = np.zeros((t_max, self.n_rec))  # input (feedforward plus recurrent)
        h = np.zeros((t_max, self.n_rec))  # time-dependent RNN activity vector
        h[0] = self.h0  # initial state
        y = np.zeros((t_max, self.n_out))  # RNN output
        err = np.zeros((t_max, self.n_out))  # readout error

        # If rflo, eligibility traces p and q should have rank 2; if rtrl, rank 3:
        if rtrl:
            p = np.zeros((self.n_rec, self.n_rec, self.n_rec))
            q = np.zeros((self.n_rec, self.n_rec, self.n_in))
        elif rflo:
            p = np.zeros((self.n_rec, self.n_rec))
            q = np.zeros((self.n_rec, self.n_in))

        for jj in range(self.n_rec):
            if rtrl:
                q[jj, jj, :] = df(u[0, jj]) * x[0, :] / self.tau_m
            elif rflo:
                q[jj, :] = df(u[0, jj]) * x[0, :] / self.tau_m

        loss = np.inf

        for tt in range(t_max - 1):
            u[tt +
              1] = np.dot(self.w_rec, h[tt]) + np.dot(self.w_in, x[tt + 1])
            h[tt + 1] = h[tt] + (-h[tt] + f(u[tt + 1])) / self.tau_m
            y[tt + 1] = np.dot(self.w_out, h[tt + 1])

            # err is always zero except for the last time step.
            if tt == t_max - 2:
                # import ipdb as pdb
                # pdb.set_trace()
                prob = stable_softmax(y[tt + 1])
                prediction = np.argmax(prob)
                # cross entropy
                log_likelihood = -np.log(prob[target])
                loss = log_likelihood

                # gradient for output nodes
                prob[target] -= 1

                err[tt + 1] = prob

            if rflo:
                p = (1 - 1 / self.tau_m) * p
                q = (1 - 1 / self.tau_m) * q
                p += np.outer(df(u[tt + 1, :]), h[tt, :]) / self.tau_m
                q += np.outer(df(u[tt + 1, :]), x[tt, :]) / self.tau_m

            if rflo and online_learning:
                dw_out = eta1 / t_max * np.outer(err[tt + 1], h[tt + 1])
                dw_rec = eta2 * np.outer(np.dot(self.b, err[tt + 1]),
                                         np.ones(self.n_rec)) * p / t_max
                dw_in = eta3 * np.outer(np.dot(self.b, err[tt + 1]),
                                        np.ones(self.n_in)) * q / t_max
            elif rflo and not online_learning:
                dw_out += eta1 / t_max * np.outer(err[tt + 1], h[tt + 1])
                dw_rec += eta2 * np.outer(np.dot(self.b, err[tt + 1]),
                                          np.ones(self.n_rec)) * p / t_max
                dw_in += eta3 * np.outer(np.dot(self.b, err[tt + 1]),
                                         np.ones(self.n_in)) * q / t_max

            if online_learning and not bptt:
                self.w_out = self.w_out - dw_out
                self.w_rec = self.w_rec - dw_rec
                self.w_in = self.w_in - dw_in

        if not online_learning:  # wait until end of trial to update weights
            self.w_out = self.w_out - dw_out
            self.w_rec = self.w_rec - dw_rec
            self.w_in = self.w_in - dw_in

        return prediction, loss


class RfloRnnZeroSynTau:
    '''
    A recurrent neural network.
    Adopted from James Murray's code for RFLO paper.
    But with no synaptic eligibity traces

    Parameters:
    ----------
    n_in, n_rec, n_out : number of input, recurrent, and hidden units.

    h0 : The initial state vector of the RNN.

    tau_m : The network time constant, in units of timesteps.
    '''
    def __init__(self, n_in, n_rec, n_out, h0, tau_m=10):
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.h0 = h0
        self.tau_m = tau_m

        # Initialize weights:
        self.w_in = 0.1 * (np.random.rand(n_rec, n_in) - 1)
        self.w_rec = 1.5 * np.random.randn(n_rec, n_rec) / n_rec**0.5
        self.w_out = 0.1 * (2 * np.random.rand(n_out, n_rec) - 1) / n_rec**0.5

        # Random error feedback matrix:
        self.b = np.random.randn(n_rec, n_out) / n_out**0.5

    def run_trial(self,
                  x,
                  y_,
                  eta=[0.1, 0.1, 0.1],
                  learning=None,
                  online_learning=False):
        '''
        Run the RNN for a single trial.

        Parameters:
        -----------
        x : The input to the network. x[t,i] is input from unit i at timestep t.

        y_ : The target RNN output, where y_[t,i] is output i at timestep t.

        eta : A list of 3 learning rates, for w_in, w_rec, and w_out,
            respectively.

        learning : Specify the learning algorithm with one of the following
            strings: 'rtrl', 'bptt', or 'rflo'. If None, run the network without
            learning.

        online_learning : If True (and learning is on), update weights at each
            timestep. If False (and learning is on), update weights only at the
            end of each trial. Online learning cannot be used with BPTT.

        Returns:
        --------
        y : The time-dependent network output. y[t,i] is output i at timestep t.

        h : The time-dependent RNN state vector. h[t,i] is unit i at timestep t.

        u : The inputs to RNN units (feedforward plus recurrent) at each
            timestep.
        '''

        # Boolean shorthands to specify learning algorithm:
        rtrl = learning == 'rtrl'
        bptt = learning == 'bptt'
        rflo = learning == 'rflo'

        [eta3, eta2, eta1] = eta  # learning rates for w_in, w_rec, and w_out
        t_max = np.shape(x)[0]  # number of timesteps

        dw_in, dw_rec, dw_out = 0, 0, 0  # changes to weights

        u = np.zeros((t_max, self.n_rec))  # input (feedforward plus recurrent)
        h = np.zeros((t_max, self.n_rec))  # time-dependent RNN activity vector
        h[0] = self.h0  # initial state
        y = np.zeros((t_max, self.n_out))  # RNN output
        err = np.zeros((t_max, self.n_out))  # readout error

        # If rflo, eligibility traces p and q should have rank 2; if rtrl, rank 3:
        if rtrl:
            p = np.zeros((self.n_rec, self.n_rec, self.n_rec))
            q = np.zeros((self.n_rec, self.n_rec, self.n_in))
        elif rflo:
            p = np.zeros((self.n_rec, self.n_rec))
            q = np.zeros((self.n_rec, self.n_in))

        for jj in range(self.n_rec):
            if rtrl:
                q[jj, jj, :] = df(u[0, jj]) * x[0, :] / self.tau_m
            elif rflo:
                q[jj, :] = df(u[0, jj]) * x[0, :] / self.tau_m

        for tt in range(t_max - 1):
            u[tt +
              1] = np.dot(self.w_rec, h[tt]) + np.dot(self.w_in, x[tt + 1])
            h[tt + 1] = h[tt] + (-h[tt] + f(u[tt + 1])) / self.tau_m
            y[tt + 1] = np.dot(self.w_out, h[tt + 1])
            err[tt + 1] = y_[tt + 1] - y[tt + 1]  # readout error

            if rflo:
                # p = (1 - 1 / self.tau_m) * p
                # q = (1 - 1 / self.tau_m) * q
                p = np.outer(df(u[tt + 1, :]), h[tt, :])
                q = np.outer(df(u[tt + 1, :]), x[tt, :])
            elif rtrl:
                p = np.tensordot((1 - 1 / self.tau_m) * np.eye(self.n_rec) +
                                 df(u[tt + 1]) * self.w_rec / self.tau_m,
                                 p,
                                 axes=1)
                q = np.tensordot((1 - 1 / self.tau_m) * np.eye(self.n_rec) +
                                 df(u[tt + 1]) * self.w_rec / self.tau_m,
                                 q,
                                 axes=1)
                for jj in range(self.n_rec):
                    p[jj, jj, :] += df(u[tt + 1, jj]) * h[tt] / self.tau_m
                    q[jj, jj, :] += df(u[tt + 1, jj]) * x[tt + 1] / self.tau_m

            if rflo and online_learning:
                dw_out = eta1 / t_max * np.outer(err[tt + 1], h[tt + 1])
                dw_rec = eta2 * np.outer(np.dot(self.b, err[tt + 1]),
                                         np.ones(self.n_rec)) * p / t_max
                dw_in = eta3 * np.outer(np.dot(self.b, err[tt + 1]),
                                        np.ones(self.n_in)) * q / t_max
            elif rflo and not online_learning:
                dw_out += eta1 / t_max * np.outer(err[tt + 1], h[tt + 1])
                dw_rec += eta2 * np.outer(np.dot(self.b, err[tt + 1]),
                                          np.ones(self.n_rec)) * p / t_max
                dw_in += eta3 * np.outer(np.dot(self.b, err[tt + 1]),
                                         np.ones(self.n_in)) * q / t_max
            elif rtrl and online_learning:
                dw_out = eta1 / t_max * np.outer(err[tt + 1], h[tt + 1])
                dw_rec = eta2 / t_max * np.tensordot(
                    np.dot(err[tt + 1], self.w_out), p, axes=1)
                dw_in = eta3 / t_max * np.tensordot(
                    np.dot(err[tt + 1], self.w_out), q, axes=1)
            elif rtrl and not online_learning:
                dw_out += eta1 / t_max * np.outer(err[tt + 1], h[tt + 1])
                dw_rec += eta2 / t_max * np.tensordot(
                    np.dot(err[tt + 1], self.w_out), p, axes=1)
                dw_in += eta3 / t_max * np.tensordot(
                    np.dot(err[tt + 1], self.w_out), q, axes=1)

            if online_learning and not bptt:
                self.w_out = self.w_out + dw_out
                self.w_rec = self.w_rec + dw_rec
                self.w_in = self.w_in + dw_in

        if bptt:  # backward pass for BPTT
            z = np.zeros((t_max, self.n_rec))
            z[-1] = np.dot((self.w_out).T, err[-1])
            for tt in range(t_max - 1, 0, -1):
                z[tt - 1] = z[tt] * (1 - 1 / self.tau_m)
                z[tt - 1] += np.dot((self.w_out).T, err[tt])
                z[tt - 1] += np.dot(z[tt] * df(u[tt]), self.w_rec) / self.tau_m

                # Updates for the weights:
                dw_out += eta1 * np.outer(err[tt], h[tt]) / t_max
                dw_rec += eta2 / (t_max * self.tau_m) * np.outer(
                    z[tt] * df(u[tt]), h[tt - 1])
                dw_in += eta3 / (t_max * self.tau_m) * np.outer(
                    z[tt] * df(u[tt]), x[tt])

        if not online_learning:  # wait until end of trial to update weights
            self.w_out = self.w_out + dw_out
            self.w_rec = self.w_rec + dw_rec
            self.w_in = self.w_in + dw_in

        return y, h, u

    def run_trial_seq_classification(self,
                                     x,
                                     target,
                                     eta=[0.1, 0.1, 0.1],
                                     learning=None,
                                     online_learning=False):
        '''
        Run the RNN for a single trial for sequential classification.

        Parameters:
        -----------
        x : The input to the network. x[t,i] is input from unit i at timestep t.

        target (int) : The target RNN output class label

        eta : A list of 3 learning rates, for w_in, w_rec, and w_out,
            respectively.

        learning : Specify the learning algorithm with one of the following
            strings: 'rtrl', 'bptt', or 'rflo'. If None, run the network without
            learning.

        online_learning : If True (and learning is on), update weights at each
            timestep. If False (and learning is on), update weights only at the
            end of each trial. Online learning cannot be used with BPTT.

        '''

        # Boolean shorthands to specify learning algorithm:
        rtrl = learning == 'rtrl'
        bptt = learning == 'bptt'
        rflo = learning == 'rflo'

        [eta3, eta2, eta1] = eta  # learning rates for w_in, w_rec, and w_out
        t_max = np.shape(x)[0]  # number of timesteps

        dw_in, dw_rec, dw_out = 0, 0, 0  # changes to weights

        u = np.zeros((t_max, self.n_rec))  # input (feedforward plus recurrent)
        h = np.zeros((t_max, self.n_rec))  # time-dependent RNN activity vector
        h[0] = self.h0  # initial state
        y = np.zeros((t_max, self.n_out))  # RNN output
        err = np.zeros((t_max, self.n_out))  # readout error

        # If rflo, eligibility traces p and q should have rank 2; if rtrl, rank 3:
        if rtrl:
            p = np.zeros((self.n_rec, self.n_rec, self.n_rec))
            q = np.zeros((self.n_rec, self.n_rec, self.n_in))
        elif rflo:
            p = np.zeros((self.n_rec, self.n_rec))
            q = np.zeros((self.n_rec, self.n_in))

        for jj in range(self.n_rec):
            if rtrl:
                q[jj, jj, :] = df(u[0, jj]) * x[0, :] / self.tau_m
            elif rflo:
                q[jj, :] = df(u[0, jj]) * x[0, :] / self.tau_m

        loss = np.inf

        for tt in range(t_max - 1):
            u[tt +
              1] = np.dot(self.w_rec, h[tt]) + np.dot(self.w_in, x[tt + 1])
            h[tt + 1] = h[tt] + (-h[tt] + f(u[tt + 1])) / self.tau_m
            y[tt + 1] = np.dot(self.w_out, h[tt + 1])

            # err is always zero except for the last time step.
            if tt == t_max - 2:
                # import ipdb as pdb
                # pdb.set_trace()
                prob = stable_softmax(y[tt + 1])
                prediction = np.argmax(prob)
                # cross entropy
                log_likelihood = -np.log(prob[target])
                loss = log_likelihood

                # gradient for output nodes
                prob[target] -= 1

                err[tt + 1] = prob

            if rflo:
                p = (1 - 1 / self.tau_m) * p
                q = (1 - 1 / self.tau_m) * q
                p += np.outer(df(u[tt + 1, :]), h[tt, :]) / self.tau_m
                q += np.outer(df(u[tt + 1, :]), x[tt, :]) / self.tau_m

            if rflo and online_learning:
                dw_out = eta1 / t_max * np.outer(err[tt + 1], h[tt + 1])
                dw_rec = eta2 * np.outer(np.dot(self.b, err[tt + 1]),
                                         np.ones(self.n_rec)) * p / t_max
                dw_in = eta3 * np.outer(np.dot(self.b, err[tt + 1]),
                                        np.ones(self.n_in)) * q / t_max
            elif rflo and not online_learning:
                dw_out += eta1 / t_max * np.outer(err[tt + 1], h[tt + 1])
                dw_rec += eta2 * np.outer(np.dot(self.b, err[tt + 1]),
                                          np.ones(self.n_rec)) * p / t_max
                dw_in += eta3 * np.outer(np.dot(self.b, err[tt + 1]),
                                         np.ones(self.n_in)) * q / t_max

            if online_learning and not bptt:
                self.w_out = self.w_out - dw_out
                self.w_rec = self.w_rec - dw_rec
                self.w_in = self.w_in - dw_in

        if not online_learning:  # wait until end of trial to update weights
            self.w_out = self.w_out - dw_out
            self.w_rec = self.w_rec - dw_rec
            self.w_in = self.w_in - dw_in

        return prediction, loss
