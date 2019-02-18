import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE

import torch
import torchvision
import torchvision.transforms as transforms


def plot_single_digits(trainloader):
    for img in trainloader.dataset.train_data:
        npimg = img.numpy()
        plt.imshow(npimg, cmap='Greys')
        plt.show()


def plot_tSNE(testloader, num_samples, fit=False, colored=True):
    tsne = TSNE(n_components=2, perplexity=40, n_iter=200000, n_iter_without_progress=250,
                init='random', random_state=1337, verbose=4, n_jobs=12)
    X_img = testloader.dataset.test_data.numpy()[:num_samples]
    Y = testloader.dataset.test_labels.numpy()[:num_samples]

    if fit:
        X = X_img.reshape(-1, X_img.shape[1] * X_img.shape[2])  # flattening out squared images for tSNE

        print("fitting PCA...")
        t0 = time.time()
        pca = PCA(n_components=30)
        X = pca.fit_transform(X)
        t1 = time.time()
        print("done! {0:.2f} seconds".format(t1 - t0))

        print("fitting tSNE...")
        t0 = time.time()
        X_tsne = tsne.fit_transform(X)
        t1 = time.time()
        print("done! {0:.2f} seconds".format(t1 - t0))

        # scaling
        x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
        X_tsne = (X_tsne - x_min) / (x_max - x_min)

        pickle.dump(X_tsne, open("../data/tSNE/X_tSNE_{0}.p".format(num_samples), "wb"))

    print("loading fitted tSNE coordinates...")
    X_tsne = pickle.load(open("../data/tSNE/X_tSNE_{0}.p".format(num_samples), "rb"))

    print("plotting tSNE...")
    t0 = time.time()

    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
    ax = fig.add_subplot(111)

    # define class colors
    cmaps = [plt.cm.bwr,
             plt.cm.bwr,
             plt.cm.Wistia,
             plt.cm.Greys,
             plt.cm.cool,
             plt.cm.Purples,
             plt.cm.coolwarm,
             plt.cm.bwr,
             plt.cm.PiYG,
             plt.cm.cool]

    if hasattr(offsetbox, 'AnnotationBbox'):
        for i_digit in range(num_samples):
            # create colormap
            custom_cmap = cmaps[Y[i_digit]]
            custom_cmap_colors = custom_cmap(np.arange(custom_cmap.N))
            if Y[i_digit] in [7, 6, 9]:
                custom_cmap_colors = custom_cmap_colors[::-1]
            custom_cmap_colors[:, -1] = np.linspace(0, 1, custom_cmap.N)
            custom_cmap = ListedColormap(custom_cmap_colors)

            if not colored:
                custom_cmap = plt.cm.Greys
                custom_cmap_colors = custom_cmap(np.arange(custom_cmap.N))
                custom_cmap_colors[:, -1] = np.linspace(0, 1, custom_cmap.N)
                custom_cmap = ListedColormap(custom_cmap_colors)

            # correct color for plotting
            X_img[i_digit][X_img[i_digit, :, :] > 10] = 255
            X_img[i_digit][X_img[i_digit, :, :] <= 10] = 0
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(X_img[i_digit],
                                                                      cmap=custom_cmap,
                                                                      zoom=0.25),
                                                X_tsne[i_digit],
                                                frameon=False,
                                                pad=0)
            ax.add_artist(imagebox)
    ax.axis("off")
    fig_path = "../plots/MNIST_tSNE_{0}_colored.png".format(num_samples)
    if not colored:
        fig_path = "../plots/MNIST_tSNE_{0}.png".format(num_samples)
    plt.savefig(fig_path, dpi=1200)
    # plt.show()
    t1 = time.time()
    print("done! {0:.2f} seconds".format(t1 - t0))


if __name__ == "__main__":

    # load data
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    num_digits = testloader.dataset.test_labels.size()

    # plot_single_digits(trainloader)
    plot_tSNE(testloader, num_samples=10000, fit=False, colored=False)