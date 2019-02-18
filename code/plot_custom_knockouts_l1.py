import pickle
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.colors import ListedColormap

import scipy.stats as spst

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from train_test_net import Net


def plot_weights(weights, scale, unit_struct, pixel_metrics, pixel_metrics_untrained, title, name):
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05, wspace=0.2, hspace=0.5)
    for i in range(20):
        ax = fig.add_subplot(4, 5, i+1)
        i_weights = weights[i].reshape(28, 28)
        ax.matshow(i_weights, cmap='seismic', vmin=scale[0]-np.mean(scale), vmax=scale[1]-np.mean(scale))
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title("U: {0:.2e}, p: {1:.2e}".format(int(unit_struct[i, 2]), unit_struct[i, 3]))
        ax.set_xlabel("mean: {0:.2f}, std: {1:.4f}".format(int(unit_struct[i, 0]), unit_struct[i, 1]))
        ax.set_ylabel("rel. pixel metric: {0:.2f}".format(np.sum(pixel_metrics[i])/np.sum(pixel_metrics_untrained[i])))
    plt.suptitle("{0}".format(title))
    plt.savefig("../plots/weights_" + name)
    plt.close()
    # plt.show()


def calc_unit_struct_metric(weights_trained, weights_untrained):

    def pixel_diff_metric(weights):
        weights = weights.reshape(28, 28)
        pixel_metrics = np.zeros((28, 28))
        for i_row in range(len(weights)):
            for i_col in range(len(weights)):
                pixel_value = weights[i_row, i_col]
                if i_row in [0, 27] and i_col in [0, 27]:  # corners
                    if i_row == 0 and i_col == 0:
                        neighbors = weights[:2, :2]
                    elif i_row == 0 and i_col == 27:
                        neighbors = weights[:2, 26:]
                    elif i_row == 27 and i_col == 0:
                        neighbors = weights[26:, :2]
                    elif i_row == 27 and i_col == 27:
                        neighbors = weights[26:, 26:]
                elif i_row in [0, 27] or i_col in [0, 27]:  # edges
                    if i_row == 0:
                        neighbors = weights[:2, i_col-1:i_col+2]
                    elif i_row == 27:
                        neighbors = weights[26:, i_col-1:i_col+2]
                    elif i_col == 0:
                        neighbors = weights[i_row-1:i_row+2, :2]
                    elif i_col == 27:
                        neighbors = weights[i_row:i_row+2, 26:]
                else:  # neither corners nor edges but in the middle
                    neighbors = weights[i_row-1:i_row+2, i_col-1:i_col+2]
                pixel_metric = np.abs(np.sum((neighbors - pixel_value))/(len(neighbors)))
                pixel_metrics[i_row, i_col] = pixel_metric
        return pixel_metrics

    unit_struct = np.zeros((len(weights_trained), 4))
    pixel_metrics = np.zeros((len(weights_trained), 28, 28))
    for i_unit in range(len(weights_trained)):
        mean, std = np.mean(weights_trained[i_unit]), np.std(weights_trained[i_unit])
        U, p = spst.mannwhitneyu(weights_trained[i_unit], weights_untrained[i_unit])
        pixel_metric = pixel_diff_metric(weights_trained[i_unit])
        unit_struct[i_unit] = mean, std, U, p
        pixel_metrics[i_unit] = pixel_metric
    return unit_struct, pixel_metrics


def plot_acc_metric_corr(metrics, accuracies):
    accuracies = accuracies[np.argsort(metrics)]
    metrics = np.sort(metrics)
    r, p = spst.pearsonr(metrics, accuracies)
    r2, p2 = spst.spearmanr(metrics, accuracies)
    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.96, bottom=0.10, wspace=0.2, hspace=0.2)
    ax = fig.add_subplot(111)
    ax.invert_xaxis()
    ax.semilogx(metrics, accuracies, lw=0, marker='o', label="pearson - r: {0:.2f}, p: {1:.2e} \n"
                                                             "spearman - r: {2:.2f}, p: {3:.2e}".format(r, p, r2, p2))
    ax.set_xlabel("metric")
    ax.set_ylabel("delta accuracy [%]")
    ax.legend(loc=2)
    ax.grid(axis='y')
    plt.savefig("../plots/acc_metric_corr.png")


def plot_tSNE(testloader, labels, num_samples, name=None, title=None):
    X_img = testloader.dataset.test_data.numpy()[:num_samples]

    print("loading fitted tSNE coordinates...")
    X_tsne = pickle.load(open("../data/tSNE/X_tSNE_10000.p".format(num_samples), "rb"))

    print("plotting tSNE...")
    # scaling
    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)
    t0 = time.time()

    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
    ax = fig.add_subplot(111)

    # Define custom color maps
    custom_cmap_black = plt.cm.Greys
    custom_cmap_black_colors = custom_cmap_black(np.arange(custom_cmap_black.N))
    custom_cmap_black_colors[:, -1] = np.linspace(0, 1, custom_cmap_black.N)
    custom_cmap_black = ListedColormap(custom_cmap_black_colors)

    custom_cmap_red = plt.cm.bwr
    custom_cmap_red_colors = custom_cmap_red(np.arange(custom_cmap_red.N))
    custom_cmap_red_colors[:, -1] = np.linspace(0, 1, custom_cmap_red.N)
    custom_cmap_red = ListedColormap(custom_cmap_red_colors)

    custom_cmap_white = plt.cm.Greys
    custom_cmap_white_colors = custom_cmap_white(np.arange(custom_cmap_white.N))
    custom_cmap_white_colors[:, -1] = 0
    custom_cmap_white = ListedColormap(custom_cmap_white_colors)

    custom_cmap_green = plt.cm.brg
    custom_cmap_green_colors = custom_cmap_green(np.arange(custom_cmap_green.N))
    custom_cmap_green_colors[:, -1] = np.linspace(0, 1, custom_cmap_green.N)
    custom_cmap_green = ListedColormap(custom_cmap_green_colors)

    color_maps = [custom_cmap_red, custom_cmap_black, custom_cmap_green, custom_cmap_white]

    if hasattr(offsetbox, 'AnnotationBbox'):
        for i_digit in range(num_samples):
            # correct color for plotting
            X_img[i_digit][X_img[i_digit, :, :] > 10] = 255
            X_img[i_digit][X_img[i_digit, :, :] <= 10] = 0
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(X_img[i_digit],
                                                                      cmap=color_maps[labels[i_digit]],
                                                                      zoom=0.25),
                                                X_tsne[i_digit],
                                                frameon=False,
                                                pad=0)
            ax.add_artist(imagebox)

    ax.set_title(title)
    ax.axis("off")
    # save figure
    plt.savefig("../plots/MNIST_tSNE_{0}_{1}.png".format(num_samples, name), dpi=1200)
    t1 = time.time()
    print("done! {0:.2f} seconds".format(t1 - t0))


def plot_unit_class_acc(acc, acc_class, title, name, color='k', paper=False):
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.05)
    ax = fig.add_subplot(111)

    bar_pos = np.linspace(-1, 9, 11, endpoint=True)
    bar_pos[0] -= 0.8
    bar_widths = np.zeros(11) + 0.8
    bar_widths[0] += 0.4
    bar_heights = np.insert(acc_class*100, 0, acc)
    ax.bar(x=bar_pos, align='center', height=bar_heights, width=bar_widths,
           color=color, edgecolor='k', lw=2)
    for i, val in enumerate(bar_heights):
        ax.text(bar_pos[i]-0.4, 110, "{0:.2f}%".format(val), color='k', fontweight='bold', fontsize=20, rotation=45)
    ax.axvline(x=-0.8, lw=4, ls='--', c='k')
    ax.axhline(y=acc, ls='--', lw=4, c='r', label="class averaged accuracy")
    ax.set_xlabel("class label", fontsize=20, fontweight='bold')
    ax.set_ylabel("accuracy [%]", fontsize=20, fontweight='bold')
    ax.set_xticks(bar_pos)
    labels = ["combined"] + np.arange(0, 10, 1).tolist()
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=20)
    ax.set_ylim(0, 115)
    # ax.set_ylim(-10, 110)
    plt.legend(loc=4, fontsize=20)
    plt.tight_layout()
    plt.savefig("../plots/unit_acc_" + name)
    # plt.show()


def plot_unit_class_acc2(accs, accs_class, title, name):
    fig = plt.figure(figsize=(12, 10))
    fig.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.05)
    ax = fig.add_subplot(111)

    bar_pos = np.linspace(-3, 27, 11, endpoint=True)
    bar_pos[0] -= 2.0
    bar_widths = np.zeros(11) + 0.8
    bar_widths[0] += 0.8
    bar_heights1 = np.insert(accs_class[0]*100, 0, accs[0])
    bar_heights2 = np.insert((accs_class[0]-accs_class[1]) * 100, 0, accs[0]-accs[1])
    bar_heights3 = bar_heights1-bar_heights2

    ax.bar(x=bar_pos - bar_widths, align='center', height=bar_heights1, width=bar_widths,
           color='k', edgecolor='k', lw=2)
    ax.bar(x=bar_pos, align='center', height=bar_heights3, width=bar_widths,
           color='b', edgecolor='k', lw=2)
    ax.bar(x=bar_pos + bar_widths, align='center', height=bar_heights2, width=bar_widths,
           color='r', edgecolor='k', lw=2)
    colors = ['k', 'b', 'r']
    for i_bars, bar_heights in enumerate([bar_heights1, bar_heights3, bar_heights2]):
        for i, val in enumerate(bar_heights):
            if i == 0:
                ax.text(bar_pos[i]-2.0+1.6*i_bars, 112, "{0:.2f}%".format(val), color=colors[i_bars],
                        fontsize=20, fontweight='bold', rotation=80)
            else:
                ax.text(bar_pos[i] - 1.2 + 0.8*i_bars, 112, "{0:.2f}%".format(val), color=colors[i_bars],
                        fontsize=14, fontweight='bold', rotation=80)
    ax.axhline(y=accs[0], ls='--', lw=4, c='k')
    ax.axhline(y=accs[1], ls='--', lw=4, c='b')
    ax.axhline(y=accs[0]-accs[1], ls='--', lw=4, c='r')
    ax.set_xlabel("class label", fontsize=20, fontweight='bold')
    ax.set_ylabel("accuracy [%]", fontsize=20, fontweight='bold')
    ax.set_yticks(np.linspace(-10, 100, 12, endpoint=True))
    ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=20)
    ax.set_xticks(bar_pos)
    labels = ["combined"] + np.arange(0, 10, 1).tolist()
    ax.set_xticklabels(labels, fontsize=20)
    # ax.set_title(title)
    ax.set_ylim(-10, 115)
    plt.tight_layout()
    plt.savefig("../plots/unit_acc_" + name)
    # plt.show()


def plot_unit_class_acc2_paper(accs, accs_class, title, name):
    fig = plt.figure(figsize=(10, 8))
    fig.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.05)
    ax = fig.add_subplot(111)

    bar_pos = np.linspace(-2, 18, 11, endpoint=True)
    bar_pos[0] -= 2.0
    bar_widths = np.zeros(11) + 0.8
    bar_widths[0] += 0.8
    bar_heights1 = np.insert(accs_class[0]*100, 0, accs[0])
    bar_heights2 = np.insert((accs_class[0]-accs_class[1]) * 100, 0, accs[0]-accs[1])
    bar_heights3 = bar_heights1-bar_heights2

    ax.bar(x=bar_pos - 0.5*bar_widths, align='center', height=bar_heights3, width=bar_widths,
           color='k', edgecolor='k', lw=2)
    ax.bar(x=bar_pos + 0.5*bar_widths, align='center', height=bar_heights2, width=bar_widths,
           color='r', edgecolor='k', lw=2)
    colors = ['k', 'r']
    for i_bars, bar_heights in enumerate([bar_heights3, bar_heights2]):
        for i, val in enumerate(bar_heights):
            if i == 0:
                if i_bars == 0:
                    ax.text(bar_pos[i]-0.8+0.8*i_bars, 118, "{0:.1f}%".format(val), color=colors[i_bars],
                            fontsize=20, fontweight='bold', rotation=80)
                else:
                    ax.text(bar_pos[i] - 0.8 + 0.7 * i_bars, 118, "+{0:.1f}%p".format(val), color=colors[i_bars],
                            fontsize=20, fontweight='bold', rotation=80)
            else:
                if val < 0:
                    color = 'lime'
                    ax.text(bar_pos[i] - 0.8 + 0.7 * i_bars, 118, "{0:.1f}%p".format(val), color=color,
                            fontsize=16, fontweight='bold', rotation=80)
                else:
                    color = colors[i_bars]
                    if i_bars == 0:
                        ax.text(bar_pos[i] - 0.8 + 0.8*i_bars, 118, "{0:.1f}%".format(val), color=color,
                                fontsize=16, fontweight='bold', rotation=80)
                    else:
                        ax.text(bar_pos[i] - 0.8 + 0.7 * i_bars, 118, "+{0:.1f}%p".format(val), color=color,
                                fontsize=16, fontweight='bold', rotation=80)
    ax.axhline(y=accs[1], ls='--', lw=4, c='k')
    ax.axhline(y=accs[0]-accs[1], ls='--', lw=4, c='r')
    ax.set_xlabel("Class label", fontsize=20)
    ax.set_ylabel("Accuracy [%]", fontsize=20)
    ax.set_yticks(np.linspace(-10, 100, 12, endpoint=True))
    ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=20)
    ax.set_xticks(bar_pos)
    labels = ["combined"] + np.arange(0, 10, 1).tolist()
    ax.set_xticklabels(labels, fontsize=20)
    # ax.set_title(title)
    ax.set_ylim(-10, 125)
    plt.tight_layout()
    plt.savefig("../plots/unit_acc_" + name)
    # plt.show()


def plot_activation(testloader, weights, weights_ini, accs_class_full, accs_class, plot_corr_acc_act):
    inputs = testloader.dataset.test_data.numpy()
    targets = testloader.dataset.test_labels.numpy()

    # loop through all classes
    total_unit_activations = np.zeros((20, 10))
    total_unit_activations_ini = np.zeros((20, 10))
    for i_class in range(10):
        # creating plotting environment
        fig, ax = plt.subplots(4, 5, figsize=(20, 10))
        fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05, wspace=0.2, hspace=0.5)
        # set plotting scale

        input = inputs[targets == i_class]
        input_avr = np.mean(input, axis=0)
        for i_unit in range(20):
            unit_activation = input_avr * weights[i_unit].reshape(28, 28)
            unit_activation_ini = input_avr * weights_ini[i_unit].reshape(28, 28)
            # calculate statistics on activation
            U, p = spst.mannwhitneyu(unit_activation.flatten(), unit_activation_ini.flatten())
            total_unit_activations[i_unit][i_class] = np.sum(unit_activation)
            total_unit_activations_ini[i_unit][i_class] = np.sum(unit_activation_ini)
            scale = (np.min(unit_activation), np.max(unit_activation))
            # plot on axis
            row = i_unit // 5
            col = i_unit % 5
            # ax[row][col].matshow(input.reshape(28, 28), cmap=plt.cm.Greys,alpha=1)
            ax[row][col].matshow(unit_activation.reshape(28, 28), cmap='bwr', alpha=1.0,
                                 vmin=scale[0]-np.mean(scale), vmax=scale[1]-np.mean(scale))
            ax[row][col].set_title("activation: {0:.2f}, delta_acc: {1:.2f}%".format(
                unit_activation.sum(), 100*(accs_class_full[i_class]-accs_class[i_unit][i_class])))
            # plotting cosmetics
            ax[row][col].set_xticks([])
            ax[row][col].set_yticks([])
            ax[row][col].set_xlabel("U: {0:.2e}, p: {1:.2e}".format(U, p))
        plt.suptitle("class {0}, accuracy: {1:.2f}%".format(i_class, 100*accs_class_full[i_class]))

        # save plot
        plt.savefig("../plots/activation_class_{0}".format(i_class))

    if plot_corr_acc_act:
        # sclaing of activation
        # total_unit_activations = F.softmax(torch.from_numpy(total_unit_activations), dim=0).numpy()
        total_unit_activations /= np.max(np.abs([np.min(total_unit_activations),
                                                 np.max(total_unit_activations)]))

        # plot correlation between activation and accuracy drop
        fig = plt.figure(figsize=(8, 6))
        fig.subplots_adjust(left=0.08, right=0.97, top=0.96, bottom=0.10, wspace=0.2, hspace=0.2)
        ax = fig.add_subplot(111)
        x = total_unit_activations.flatten()
        y = 100*(-accs_class+accs_class_full).flatten()

        r, p = spst.pearsonr(x, y)
        r2, p2 = spst.spearmanr(x, y)
        ax.plot(x, y, lw=0, marker='o',
                label="pearson - r: {0:.2f}, p: {1:.2e} \n spearman - r: {2:.2f}, p: {3:.2e}".format(r, p, r2, p2))
        ax.set_xlabel("activation")
        ax.set_ylabel("delta accuracy [%]")
        ax.set_yticks(np.arange(-10, 101, 10))
        ax.set_ylim(-10, 130)
        ax.set_xlim(-1.1, 1.1)
        ax.legend(loc=2)
        ax.grid(axis='y')
        plt.savefig("../plots/acc_activation_corr_total.png")

        # plot correlation between activation and accuracy drop for each class separately
        fig = plt.figure(figsize=(15, 10))
        fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.05, wspace=0.3, hspace=0.3)
        for i_class in range(10):
            ax = fig.add_subplot(3, 4, i_class+1)
            x = total_unit_activations[:, i_class]
            y = 100*(accs_class_full[i_class]-accs_class[:, i_class])
            r, p = spst.pearsonr(x, y)
            r2, p2 = spst.spearmanr(x, y)
            ax.plot(x, y, lw=0, marker='o',
                    label="pearson - r: {0:.2f}, p: {1:.2e} \n spearman - r: {2:.2f}, p: {3:.2e}".format(r, p, r2, p2))
            ax.set_xlabel("activation")
            ax.set_ylabel("delta accuracy [%]")
            ax.set_yticks(np.arange(-10, 101, 10))
            ax.set_ylim(-10, 130)
            ax.set_xlim(-1.1, 1.1)
            ax.set_title("class {0}".format(i_class))
            ax.legend(loc=2)
            ax.grid(axis='y')
        plt.savefig("../plots/acc_activation_corr.png")


def plot_activation_ko(testloader, unit_label_masks, weights, accs_class_full, accs_class, name):
    inputs = testloader.dataset.test_data.numpy()
    targets = testloader.dataset.test_labels.numpy()

    # loop through all classes
    total_unit_activations = np.zeros((20, 10))
    for i_class in range(10):
        # creating plotting environment
        fig, ax = plt.subplots(4, 5, figsize=(20, 10))
        fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05, wspace=0.2, hspace=0.5)
        # set plotting scale

        for i_unit in range(20):
            input = inputs[unit_label_masks[i_unit] & (targets == i_class)]
            input_avr = np.mean(input, axis=0)
            unit_activation = input_avr * weights[i_unit].reshape(28, 28)
            # calculate statistics on activation
            total_unit_activations[i_unit][i_class] = np.sum(unit_activation)
            scale = (np.min(unit_activation), np.max(unit_activation))
            # plot on axis
            row = i_unit // 5
            col = i_unit % 5
            # ax[row][col].matshow(input.reshape(28, 28), cmap=plt.cm.Greys,alpha=1)
            ax[row][col].matshow(unit_activation.reshape(28, 28), cmap='bwr', alpha=1.0,
                                 vmin=scale[0]-np.mean(scale), vmax=scale[1]-np.mean(scale))
            ax[row][col].set_title("activation: {0:.2f}, delta_acc: {1:.2f}%".format(
                unit_activation.sum(), 100*(accs_class_full[i_class]-accs_class[i_unit][i_class])))
            # plotting cosmetics
            ax[row][col].set_xticks([])
            ax[row][col].set_yticks([])
        plt.suptitle("class {0}, accuracy: {1:.2f}%".format(i_class, 100*accs_class_full[i_class]))

        # save plot
        plt.savefig("../plots/activation_class_{0}_{1}".format(i_class, name))


if __name__ == "__main__":

    # ToDo: FIND OUT WHY PLOTTING TSNE CHANGES THE ACCURACIES! The more samples are considered for tSNE the more the accuracy changes!
    # setting rng seed for reproduceaility
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    """ setting flags """
    plot_w = False
    plot_tSNE_ = False
    plot_unit_acc = True

    # load nets and weights
    net_trained = Net()
    net_untrained = Net()
    # send nets to GPU
    for net in [net_trained, net_untrained]:
        net.to(device)
    criterion = nn.NLLLoss()  # nn.CrossEntropyLoss()
    net_trained.load_state_dict(torch.load('../nets/MNIST_MLP(20, 10)_trained.pt'))
    net_trained.eval()
    net_untrained.load_state_dict(torch.load('../nets/MNIST_MLP(20, 10)_untrained.pt'))
    net_untrained.eval()

    unit_struct_untrained, pixel_metrics_untrained = calc_unit_struct_metric(net_untrained.fc1.weight.data.cpu().numpy()+1e-10,
                                                                             net_untrained.fc1.weight.data.cpu().numpy())
    unit_struct, pixel_metrics = calc_unit_struct_metric(net_trained.fc1.weight.data.cpu().numpy(),
                                                         net_untrained.fc1.weight.data.cpu().numpy())

    # load data and test network
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    # plot fully trained network weights
    # weights = (net.fc1.weight.data.numpy().T + net.fc1.bias.data.numpy()).T  # biases considered
    weights = net_trained.fc1.weight.data.cpu().numpy()
    scale = (np.min(weights), np.max(weights))
    acc_full, labels, acc_class_full, _ = net_trained.test_net(criterion, testloader, device)
    if plot_w:
        plot_weights(weights, scale, unit_struct, pixel_metrics, pixel_metrics_untrained,
                     title="trained accuracy: {0}%".format(acc_full), name="full")
    plot_tSNE(testloader, labels, num_samples=10000, name="", title="accuray: {0}%".format(acc_full))
    labels[labels == 0] = -1
    if plot_tSNE_:
        plot_tSNE(testloader, labels, num_samples=10000, name="clean", title="accuray: {0}%".format(acc_full))
    if plot_unit_acc:
        plot_unit_class_acc(acc_full, acc_class_full, title="accuray: {0}%".format(acc_full), name="full", paper=True)

    # plot untrained network weights
    weights_ini = net_untrained.fc1.weight.data.cpu().numpy()
    acc_untrained, _, _, _ = net_untrained.test_net(criterion, testloader, device)
    if plot_w:
        plot_weights(weights_ini, scale, unit_struct_untrained, pixel_metrics_untrained, pixel_metrics_untrained,
                     title="untrained accuracy: {0}%".format(acc_untrained), name="0full")

    # specify custom knockout
    ko_units = [5, 10, 12, 17]

    # modify net, test accuracy and plot weights
    unit_labels_ko = np.zeros((20, 10000))
    print("knockout unit {0}".format(ko_units))
    net_trained.load_state_dict(torch.load('../nets/MNIST_MLP(20, 10)_trained.pt'))
    net_trained.eval()
    for i_unit in ko_units:
        net_trained.fc1.weight.data[i_unit, :] = torch.zeros(784)
    # weights = (net.fc1.weight.data.numpy().T + net.fc1.bias.data.numpy()).T  # biases considered
    weights = net_trained.fc1.weight.data.cpu().numpy()
    acc, labels_ko, acc_class, _ = net_trained.test_net(criterion, testloader, device)
    unit_labels_ko = labels_ko
    if plot_w:
        plot_weights(weights, scale, unit_struct, pixel_metrics, pixel_metrics_untrained,
                     title="knockout_" + str(ko_units) + ", accuray: {0}%, delta_acc: {1:.2f}%".format(acc, acc_full-acc),
                     name="knockout_" + str(ko_units))
    if plot_tSNE_:
        labels_ko_cp = copy.deepcopy(labels_ko)
        labels_ko[labels == -1] = -1
        labels_ko[labels_ko_cp == 1] = 2
        labels_ko[labels == 1] = 1
        labels_ko[labels_ko_cp == 0] = 0
        plot_tSNE(testloader, labels_ko, num_samples=10000, name="ko_" + str(ko_units),
                  title="knockout_" + str(ko_units) + ", accuray: {0}%, delta_acc: {1:.2f}%".format(acc,
                                                                                                          acc_full - acc))
    if plot_unit_acc:
        # plot_unit_class_acc(acc, acc_class,
        #                     title="accuray: {0}%, delta_acc: {1:.2f}%".format(acc_full, acc_full-acc),
        #                     name="knockout_{0}".format(i_unit + 1))
        # plot_unit_class_acc(acc_full-acc, acc_class_full-acc_class,
        #                     title="accuray: {0}%, delta_acc: {1:.2f}%".format(acc_full, acc_full-acc),
        #                     color='r', name="knockout_{0}_delta".format(i_unit+1))
        # plot_unit_class_acc2((acc_full, acc), (acc_class_full, acc_class),
        #                     title="knockout_{0} - accuray: {1}%, delta_acc: {2:.2f}%".format(ko_units, acc_full, acc_full-acc),
        #                     name="knockout_{0}_combined".format(ko_units))
        plot_unit_class_acc2_paper((acc_full, acc), (acc_class_full, acc_class),
                             title="knockout_{0} - accuray: {1}%, delta_acc: {2:.2f}%".format(ko_units, acc_full,
                                                                                              acc_full - acc),
                             name="knockout_{0}_combined".format(ko_units))