"""
Test and compare peformance of R2N2 and RFLO RNN on anbn task (R2N2)
"""
import numpy as np

from models import R2N2
from figures import show_loss_and_accuracy

from tqdm import trange

import pickle
# import ipdb as pdb


def mse_loss(output, target):
    return np.sum(0.5 * (output - target)**2)


def d_mse_loss(output, target):
    return output - target


n_repeat = 10
for ii in range(n_repeat):
    for min_length in [1 + i * 5 for i in range(6)]:
        max_length = min_length + 3
        data_path = f"data/anbn_{min_length}-{max_length}.pkl"

        with open(data_path, 'rb') as f:
            seqs = pickle.load(f)

        n_samples = len(seqs)

        exp_name = f"exp_9_R2N2_FA_{min_length}-{max_length}_{ii}"

        n_x = 3
        T = n_x

        n_h = 32
        lr = 1e-3  # 5e-4

        model = R2N2(n_h,
                     n_x,
                     n_input=n_x,
                     g=2.5,
                     tau=10,
                     lr=lr,
                     enableFA=True,
                     g_backward=0.9)
        loss_list = []
        results_list = []

        n_epoch = 10000

        kk = trange(n_epoch,
                    desc="Average Classification Accuracy: ",
                    leave=True)

        sample_ids = np.arange(n_samples)

        for e in kk:
            np.random.shuffle(sample_ids)

            for i in sample_ids:
                model.zero_grads()
                sensory_seq = seqs[i]
                sensory_seq_next = seqs[i][1:]
                T = len(sensory_seq)
                h_forward_seq, output_seq, loss = model.forward_seq_prediction(
                    sensory_seq, mse_loss)

                model.hard_reset_w_backward()
                h_backward_seq = model.backward_learning_seq_prediction(
                    h_forward_seq[-1], sensory_seq, d_mse_loss)

                model.apply_grads_Adam(T)
                loss_list.append(loss)
                output_seq = np.array(output_seq)
                pred_ids = np.argmax(output_seq, axis=1)

                # if the prediction is perfect, there should only one error.
                pred_pos_correct = sensory_seq_next[np.arange(T - 1),
                                                    pred_ids].sum()
                if pred_pos_correct >= T - 2:
                    res = True
                else:
                    res = False

                results_list.append(res)
                # pdb.set_trace()

            if (e + 1) % 50 == 0:
                kk.set_description(
                    f"Correct rate: {np.mean(results_list[-n_samples:])} | Loss {np.around(np.mean(loss_list[-n_samples:]), 4)}"
                )

        show_loss_and_accuracy(loss_list,
                               results_list,
                               exp_name=exp_name,
                               show=True)

        pickle.dump({
            "loss": loss_list,
            "result": results_list
        }, open(f"results/exp9/{exp_name}.pkl", "wb"))
