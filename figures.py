import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
from utility import moving_average  # errorfill
plt.rcParams.update({'font.size': 20})


def show_loss_curve(loss_list,
                    time_normalization=True,
                    t_max=None,
                    ylabel="Loss Type",
                    exp_name="exp",
                    show=False):
    loss_list = np.array(loss_list)
    if time_normalization:
        assert t_max > 1
        loss_list = loss_list / t_max

    plt.semilogy(loss_list)
    plt.ylabel(ylabel)
    plt.savefig(f"figs/{exp_name}_loss.png")
    if show:
        plt.show()


def show_loss_and_accuracy(loss_list,
                           results_list,
                           exp_name="exp",
                           fig_path=None,
                           yL_label="MSE",
                           yR_label="Reward Rate",
                           show=False,
                           rasterized=True):

    # fig, ax1 = plt.subplots(figsize=[10, 8])
    fig, ax1 = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(left=0.2)
    fig.subplots_adjust(right=0.8)
    ax2 = ax1.twinx()
    ax1.semilogy(moving_average(loss_list, 100),
                 lw=3,
                 color="b",
                 rasterized=rasterized)
    ax2.plot(moving_average(results_list, 100),
             "-",
             lw=3,
             color="g",
             rasterized=rasterized)
    # ax1.set_ylim([0, 3])
    ax2.set_ylim([0, 1.1])

    ax1.set_xlabel('Trials')
    ax1.set_ylabel(yL_label, color='b')
    ax2.set_ylabel(yR_label, color='g')

    if fig_path is None:
        plt.savefig(f"figs/{exp_name}_loss_and_accuracy.png")
    else:
        plt.savefig(fig_path)


def output_target_comparison(outputs, targets, exp_name="exp", show=False):
    vmax = max(np.max(outputs), np.max(targets))
    vmin = max(np.min(outputs), np.min(targets))
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
    )  # facecolor="w", sharey='row')
    axes[0].imshow(outputs, vmin=vmin, vmax=vmax)
    axes[1].imshow(targets, vmin=vmin, vmax=vmax)
    plt.savefig(f"figs/{exp_name}_comparison.png")
    if show:
        plt.show()


def show_activity_forward_backward(hsAf,
                                   hsBf,
                                   hsAb,
                                   hsBb,
                                   exp_name="exp",
                                   show=False):
    fig, axes = plt.subplots(nrows=2, ncols=3, facecolor="w", sharey='row')
    axes[0, 0].plot(hsAf)
    axes[0, 1].plot(np.flip(hsAb, axis=0))
    axes[0, 2].plot(hsAf - np.flip(hsAb, axis=0))

    axes[1, 0].plot(hsBf)
    axes[1, 1].plot(np.flip(hsBb, axis=0))
    axes[1, 2].plot(hsBf - np.flip(hsBb, axis=0))

    axes[0, 0].set_ylabel("Group A")
    axes[1, 0].set_ylabel("Group B")
    plt.savefig(f"figs/{exp_name}_forward_backward.png")
    if show:
        plt.show()
