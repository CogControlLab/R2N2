"""
Experiment 3:
Show it's capability in sequential hand written didigt classification task.

Data is from https://github.com/aiddun/binary-mnist.
"""
import numpy as np

from models import R2N2
from utility import cross_entropy_loss, d_cross_entropy_loss
from figures import show_loss_and_accuracy

from tqdm import tqdm, trange

import pickle

data_path = "data/mnist_binary.pkl"
with open(data_path, 'rb') as f:
    mnist = pickle.load(f)

x_train = mnist["training_images"]
y_train = mnist["training_labels"]
x_test = mnist["test_images"]
y_test = mnist["test_labels"]

# print(y_train[:10])
# ns = int(np.sqrt(x_train[0].shape[0]))

# ipdb.set_trace()

exp_name = "exp_3_FA"

n_x = 28
T = n_x

n_h = 64
lr = 1e-3

x_test[0].reshape(T, T)

model = R2N2(n_h, 10, n_input=28, g=2.5, tau=10, lr=lr, enableFA=True)

loss_list = []
results_list = []




def validate(x_v, y_v):
    results_list = []
    loss_list = []
    for i in range(x_v.shape[0]):
        xs = x_v[i].reshape(T, T)
        y = y_v[i]
        h_forward_seq, output_seq, loss = model.forward_seq_decision_making(
            xs, y, cross_entropy_loss)

        loss_list.append(loss)

        prediction = np.argmax(output_seq[-1])
        results_list.append(prediction == y)

    # print(
    #     f"Validate Accuracy: {np.around(np.mean(results_list), 2)} | Loss (Cross Entropy) {np.around(np.mean(loss_list), 2)}"
    # )
    return np.mean(results_list), np.mean(loss_list)


n_samples = x_train.shape[0]
n_epoch = 5

kk = trange(n_epoch, desc="Average Classification Accuracy: ", leave=True)
acc_v, loss_v = 0, 0
acc_v_list = []

for e in kk:
    acc_v, loss_v = validate(x_test, y_test)
    acc_v_list.append(acc_v)
    for i in tqdm(range(n_samples)):
        model.zero_grads()
        xs = x_train[i].reshape(T, T)
        y = y_train[i]
        h_forward_seq, output_seq, loss = model.forward_seq_decision_making(
            xs, y, cross_entropy_loss)

        model.hard_reset_w_backward()
        h_backward_seq = model.backward_learning_decision_making(
            h_forward_seq[-1], xs, y, d_cross_entropy_loss)

        model.apply_grads_Adam(T)
        loss_list.append(loss)

        prediction = np.argmax(output_seq[-1])
        results_list.append(prediction == y)
        # print(prediction, y)

        kk.set_description(
            f"Epoch {e+1} | Train: Acc: {np.around(np.mean(results_list[-1000:]), 2)} | L: {np.around(np.mean(loss_list[-1000:]), 2)}; Validate in epoch {e}: Acc {np.around(acc_v, 2)}"
        )

show_loss_and_accuracy(loss_list,
                       results_list,
                       exp_name=exp_name,
                       show=True,
                       yL_label="Crosss entropy",
                       yR_label="Accuracy")

pickle.dump({
    "loss": loss_list,
    "result": results_list
}, open(f"results/{exp_name}.pkl", "wb"))
