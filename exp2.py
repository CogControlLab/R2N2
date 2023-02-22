"""
Experiment 2: Show its capability in navigation tasks, specifically a T-maze navigation task.
"""
import numpy as np
import matplotlib.pyplot as plt
from models import R2N2
from utility import h_seq_pre_processing
from figures import show_activity_forward_backward, show_loss_curve, output_target_comparison, show_loss_and_accuracy
import ipdb
from tqdm import tqdm, trange
import numpy as np
from tasks import TMazeNavTask
import pickle
from copy import deepcopy
from pathlib import Path

da = {0: "L", 1: "U", 2: "R", 3: "D"}


def sensory_inputs(a, reward, observation):
    # order: action(4), reward(2), observation(5)
    s = np.zeros((1, 11))
    noise_amp = 0.01
    s[0, a] = 1
    s[0, 4 + int(reward)] = 1
    s[0, 6 + int(observation)] = 1
    return s + np.random.normal(0, 1, s.shape) * noise_amp


def generate_training_data(dataset_path="dataset.pkl",
                           n_repeat=20,
                           cue_length_min=13,
                           cue_length_max=16,
                           delay_length_min=20,
                           delay_length_max=30,
                           branch_length_min=20,
                           branch_length_max=30,
                           step=2):

    sensory_seqs = []
    sensory_seq_nexts = []
    ki = 0

    for cue_length in range(cue_length_min, cue_length_max):
        for delay_length in range(delay_length_min, delay_length_max, step):
            for branch_length in range(branch_length_min, branch_length_max,
                                       step):
                cue_type_1lens = []
                cue_type_2lens = []
                for cue_type in [1, 2]:

                    for rep in range(n_repeat):
                        ki += 1

                        # cue_type = np.random.choice([1,2])
                        # print(cue_type)
                        task = TMazeNavTask(cue_length, delay_length,
                                            branch_length, cue_type)

                        # plt.imshow(task.maze, vmin=-1, vmax=2)
                        # r_coord = 0 if cue_type == 1 else task.l_branch*2
                        # plt.scatter(r_coord, task.l_cue+task.l_delay1, c='r')

                        loss = 0
                        if cue_type == 1:
                            actions = [
                                3 for i in range(cue_length + delay_length)
                            ] + [0 for i in range(branch_length)]
                        elif cue_type == 2:
                            actions = [
                                3 for i in range(cue_length + delay_length)
                            ] + [2 for i in range(branch_length)]

                        sensory_seq = []
                        sensory_seq_next = []

                        # init sensory condition
                        observation = task.reset()
                        reward = False
                        # for ii in range(10):
                        # a = np.random.choice(4)
                        a = 1
                        for t in range(len(actions)):
                            # a = np.random.choice(4) # ,p=[0.2, 0.2, 0.3, 0.3])

                            # previous sensory input
                            sensory_seq.append(
                                sensory_inputs(a, reward, observation))

                            a = actions[t]  # desired action
                            prev_coords = tuple(task.current_coords)
                            observation, reward, done = task.step(a)

                            current_coords = tuple(task.current_coords)
                            sensory_seq_next.append((a, reward, observation))
                            # print(f"T = {t+1}| action: {da[a]}| prev: {prev_coords}| new: {current_coords}| observation: {observation}| reward: {reward} | done: {done}")
                        sensory_seq.append(
                            sensory_inputs(a, reward, observation))
                        if reward == True:
                            sensory_seqs.append(
                                np.concatenate(sensory_seq, axis=0))
                            sensory_seq_nexts.append(
                                deepcopy(sensory_seq_next))
                        assert len(sensory_seq) == len(sensory_seq_next) + 1
                        if cue_type == 1:
                            cue_type_1lens.append(len(sensory_seq))
                        elif cue_type == 2:
                            cue_type_2lens.append(len(sensory_seq))

                assert cue_type_2lens == cue_type_1lens
    # plt.show()
    print(f"Finished. {len(sensory_seqs)} trials are generated.")
    pickle.dump((sensory_seqs, sensory_seq_nexts), open(dataset_path, "wb"))
    return (sensory_seqs, sensory_seq_nexts)


def load_training_data(dataset_path="dataset.pkl"):
    with open(dataset_path, "rb") as f:
        (sensory_seqs, sensory_seq_nexts) = pickle.load(f)

    return sensory_seqs, sensory_seq_nexts


fn = "data/T-maze-long.pkl"
datf = Path(fn)
if datf.is_file():
    print("loading ")
    sensory_seqs, sensory_seq_nexts = load_training_data(fn)
else:
    sensory_seqs, sensory_seq_nexts = generate_training_data(
        dataset_path=fn,
        n_repeat=10,
        cue_length_min=20,
        cue_length_max=24,
        delay_length_min=50,
        delay_length_max=56,
        branch_length_min=20,
        branch_length_max=26,
        step=1)


def mse_loss(output, target):
    return np.sum(0.5 * (output - target)**2)


def d_mse_loss(output, target):
    return output - target


def validate(model,
             cue_length=3,
             delay_length=18,
             branch_length=18,
             n_trials=20):
    cue_types = []
    trial_results = []
    t_max = cue_length + delay_length + branch_length  # + 5

    kk = trange(n_trials, desc="Validation Average Reward Rate ", leave=True)

    records = []
    for k in kk:
        if k % 2 == 0:
            cue_type = 1
        else:
            cue_type = 2
        # cue_type = np.random.choice([1, 2])
        cue_types.append(cue_type)
        task = TMazeNavTask(cue_length, delay_length, branch_length, cue_type)
        observation = task.reset()

        th = 0.2
        hs = (
            th * np.ones(model.n_hA),
            th * np.ones(model.n_hB),
        )
        a = 1
        reward = False
        x = sensory_inputs(a, reward, observation).reshape(-1)

        predictions = []

        hs_seq = []
        distance = []
        state_ns = []
        for t in range(t_max):
            hs, x_prediction = model.forward_update(hs, x)
            hs_seq.append(hs)

            a = np.argmax(x_prediction[:4])
            prev_coords = tuple(task.current_coords)

            observation, reward, done = task.step(a)
            current_coords = tuple(task.current_coords)

            # actual next step sensory inputs
            x_next = sensory_inputs(a, reward, observation).reshape(-1)

            predictions.append((
                np.argmax(x_prediction[:4]),
                np.argmax(x_prediction[4:6]),
                np.argmax(x_prediction[6:]),
            ))
            distance.append(task.get_manhattan_distance(current_coords))
            state_ns.append(task.get_state_n(current_coords))

            x = x_next

            if done:
                # print(f"T = {t+1}| action: {da[a]}| prev: {prev_coords}| \
                #       new: {current_coords}| observation: {observation}|\
                #       reward: {reward} | done: {done}")
                break

        trial_results.append(reward)

        records.append({
            'cue_type': cue_type,
            'reward': reward,
            'task': deepcopy(task),
            'actions': deepcopy(predictions),
            'hs_seq': deepcopy(hs_seq),
            'distance': deepcopy(distance),
            'state_ns': state_ns
        })
        kk.set_description(f"Average Reward: {np.mean(trial_results)*100} %")

    # import ipdb as pdb
    # pdb.set_trace()
    return records


exp_name = "exp_2_FA_LONG"
stop_threshold = 0.85
n_x = 11
n_h = 64
lr = 1e-3
model = R2N2(n_h, n_x, g=2.5, tau=10, lr=lr, enableFA=True)
# T = n_x
loss_list = []
results_list = []
nt = len(sensory_seqs)

n_trials = 100
kk = trange(n_trials, desc="Average Training Reward Rate ", leave=True)
nseq = len(sensory_seqs)
len_min = nseq * 3

for k in kk:
    indices = np.random.choice(nseq, nseq, replace=False)
    for ind in tqdm(indices):

        model.zero_grads()
        sensory_seq = sensory_seqs[ind]
        sensory_seq_next = np.array(sensory_seq_nexts[ind])

        T = sensory_seq.shape[0]

        h_forward_seq, output_seq, loss = model.forward_seq_prediction(
            sensory_seq, mse_loss)

        model.hard_reset_w_backward()
        h_backward_seq = model.backward_learning_seq_prediction(
            h_forward_seq[-1], sensory_seq, d_mse_loss)

        model.apply_grads_Adam(T)
        loss_list.append(loss / T)

        output_seq = np.array(output_seq)
        action_prediction = np.argmax(output_seq[:, :4], axis=1)
        reward_prediction = np.argmax(output_seq[:, 4:6], axis=1)
        observation_prediction = np.argmax(output_seq[:, 6:], axis=1)
        results_list.append(np.all(action_prediction == sensory_seq_next[:,
                                                                         0]))

        kk.set_description(
            f"Reward: {np.mean(results_list[-10:])} | Loss {np.around(np.mean(loss_list[-10:]), 2)}"
        )

    if len(results_list) >= len_min and np.mean(
            results_list[-len_min:]) > stop_threshold:
        break

records = validate(model,
                   cue_length=22,
                   delay_length=52,
                   branch_length=24,
                   n_trials=40)

show_loss_and_accuracy(loss_list, results_list, exp_name=exp_name, show=True)

output_seq = np.array(output_seq)
output_target_comparison(
    output_seq,
    sensory_seq[1:],
    exp_name=exp_name,
)

hs_A_f, hs_B_f = h_seq_pre_processing(h_forward_seq)
hs_A_b, hs_B_b = h_seq_pre_processing(h_backward_seq)
show_activity_forward_backward(hs_A_f,
                               hs_B_f,
                               hs_A_b,
                               hs_B_b,
                               exp_name=exp_name)

pickle.dump(
    {
        "hs": {
            "hs_A_f": hs_A_f,
            "hs_B_f": hs_B_f,
            "hs_A_b": hs_A_b,
            "hs_B_b": hs_B_b,
        },
        "comparison": {
            "output": output_seq,
            "sensory_seq": sensory_seq[1:]
        },
        "loss": {
            "loss_list": loss_list,
            "results_list": results_list
        },
        "records": records
    }, open(f"results/{exp_name}.pkl", "wb"))
