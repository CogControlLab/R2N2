import numpy as np
import matplotlib.pyplot as plt
from models import HFRNN
from utility import sgn
from scipy.spatial import distance
# import ipdb
import pickle

exp_name = "exp_6"
# define patterns

N = 500  # 300
n_patterns = 20


def generate_patterns(N, n_patterns, fill_ratio=0.4):
    k = int(N * 0.4)
    patterns = np.zeros((n_patterns, N)) - 1
    for t in range(n_patterns):
        patterns[t, np.random.choice(N, k, replace=False)] += 2
    return patterns


patterns = generate_patterns(N, n_patterns, fill_ratio=0.3)

####################################################################################
# External control signals #########################################################
####################################################################################
t_transition = 30
n_period = n_patterns - 1
t_max = t_transition * n_period
ccs = 0.5 * (np.cos(np.linspace(0, n_period * np.pi, t_max)) + 1)

# learn the sequence
rnn2 = HFRNN(N, g=3.5, tau=6)
rnn2.learn_sequence(patterns)
# recall
h = patterns[-1]
h_list = np.zeros((t_max + 1, N))
h_list[0] = h.copy()

# overlaps = np.zeros((t_max, n_patterns))

hammings = np.zeros((t_max, n_patterns))

for t in range(t_max):
    h_list[t + 1], o = rnn2.forward(h_list[t], ccs[t])
    for i_p in range(n_patterns):
        hammings[t, i_p] = distance.hamming(o, patterns[i_p])

# learn patterns
t_transition = 30
n_period = n_patterns - 1
t_max = t_transition * n_period
ccs_s = np.zeros(t_max)

rnn2 = HFRNN(N, g=2.5, tau=6)
rnn2.learn_sequence(patterns)

h = patterns[-1]  # patterns[j]
h_list_self = np.zeros((t_max + 1, N))
h_list_self[0] = h.copy()

overlaps = np.zeros((t_max, n_patterns))

ccs_self = 1
T = 0
outputs = [h.copy()]

hammings_self = np.zeros((t_max, n_patterns))

for t in range(t_max):

    h_list_self[t + 1], o = rnn2.forward(h_list_self[t], 0.5 * (ccs_self + 1))

    stable = distance.hamming(o, sgn(h_list_self[t])) == 0
    ccs_self = (-1)**(stable).astype(np.int) * ccs_self
    if stable:
        outputs.append(o.copy())

    overlaps[t] = (np.tile(o.reshape(1, -1),
                           (n_patterns, 1)) == patterns).sum(axis=1) / N

    for i_p in range(n_patterns):
        hammings_self[t, i_p] = distance.hamming(o, patterns[i_p])

    ccs_s[t] = 0.5 * (ccs_self + 1)
    T += 1
    # first elements
    if np.argmax(overlaps[t]) == 0:
        break

outputs = np.array(outputs)

fig, axes = plt.subplots(figsize=[12, 8],
                         nrows=3,
                         ncols=1,
                         sharex='col',
                         gridspec_kw={'height_ratios': [1, 1, 8]})

axes[0].plot(ccs_s[:T])
pos_list = np.argmax(overlaps, axis=1)

cls = plt.cm.jet(np.linspace(0, 1, n_patterns))

axes[1].scatter(x=np.arange(T), y=pos_list[:T], s=0.5, c=cls[pos_list[:T]])
axes[1].set_ylabel("state #")

for ii in range(n_patterns):
    axes[2].plot(overlaps[:T, ii], color=cls[ii], lw=2)
axes[2].set_ylabel("Overlap")
axes[2].set_xlabel("Time (ms)")

plt.show()

pickle.dump(
    {
        "h_list": h_list,
        "hammings": hammings,
        "ccs": ccs,
        "h_list_self": h_list_self[:T],
        "hammings_self": hammings_self[:T],
        "ccs_self": ccs_s[:T],
    }, open(f"results/{exp_name}.pkl", "wb"))
