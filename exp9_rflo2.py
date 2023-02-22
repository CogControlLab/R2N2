"""
Test and compare peformance of R2N2 and RFLO RNN on anbn task (RFLO)
Here the rflo is with no eligibility traces.
"""
import numpy as np

from figures import show_loss_and_accuracy

from rflo_model import RfloRnnZeroSynTau

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

        exp_name = f"exp_9_RFLO_ZEROSYNTAU_{min_length}-{max_length}_{ii}"

        n_x = 3
        T = n_x

        n_h = 32
        n_h_t = n_h * 2
        lr = 0.01

        h_init = 0.1 * np.ones(n_h_t)

        model = RfloRnnZeroSynTau(n_x, n_h_t, n_x, h_init)

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
                sensory_seq = seqs[i][:-1]
                sensory_seq_next = seqs[i][1:]
                T = len(sensory_seq)

                output_seq, h, u = model.run_trial(sensory_seq,
                                                   sensory_seq_next,
                                                   eta=[lr, lr, lr],
                                                   learning='rflo')
                # pdb.set_trace()
                loss_list.append(mse_loss(output_seq, sensory_seq_next) / T)

                pred_ids = np.argmax(output_seq, axis=1)
                # if the prediction is perfect, there should only one error.
                pred_pos_correct = sensory_seq_next[np.arange(T),
                                                    pred_ids].sum()
                if pred_pos_correct >= T - 1:
                    res = True
                else:
                    res = False

                results_list.append(res)
                # pdb.set_trace()

            if (e + 1) % 20 == 0:
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