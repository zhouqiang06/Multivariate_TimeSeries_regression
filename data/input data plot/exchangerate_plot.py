import numpy as np
import pandas as pd

from sys import platform
import os
import matplotlib.pyplot as plt

if platform == "win32":
    home_path = r'C:\Users\qzhou\Downloads\phd\FreqDe'
else:
    home_path = r'/home/qzhou/FreqDe'

def exchangerate_plot():
    #####################temp piecewise model with lag######################
    file_path = os.path.join(home_path, 'exchange_rate.csv')

    nations = ['Australia', 'British', 'Canada', 'Switzerland', 'China', 'Japan', 'New Zealand', 'Singapore']
    # linestyles = ['-', '--', '-.', ':']
    exchangerate_data = np.asarray(pd.read_csv(file_path, delimiter=','))
    print(exchangerate_data.shape)

    for i in range(exchangerate_data.shape[1]):
        plt.plot(exchangerate_data[:, i], linewidth=1, label=nations[i]) #, linestyle=linestyles[i], color='k'

    plt.xlabel('Time stamp')
    plt.ylabel('Exchange Rate')
    plt.legend()
    plt.show()

def exchangerate_plot_stack():
    #####################temp piecewise model with lag######################

    freq = np.loadtxt(os.path.join(home_path, 'exchange_rate_real_models_freq.txt'))
    y = np.loadtxt(os.path.join(home_path, 'exchange_rate_real_models_y.txt'))
    y_hat = np.loadtxt(os.path.join(home_path, 'exchange_rate_real_models_y_hat.txt'))
    mse = np.loadtxt(os.path.join(home_path, 'exchange_rate_real_models_mse_test.txt'))

    print(freq.shape)
    freq_x = freq[:3788][::-1]
    print(freq_x.shape)

    # nations = ['Australia', 'British', 'Canada', 'Switzerland', 'China', 'Japan', 'New Zealand', 'Singapore']
    linestyles = ['-', '--', '-.', ':']
    # exchangerate_data = np.asarray(pd.read_csv(file_path, delimiter=','))
    # print(exchangerate_data.shape)

    plt.subplot(2, 1, 1)

    plt.plot(y, linestyle=linestyles[0], linewidth=1, label='y')  # , linestyle=linestyles[i], color='k'
    plt.plot(y_hat, linestyle=linestyles[0], linewidth=1, label='y_hat')

    plt.ylabel('Exchange Rate')
    plt.xlabel('Time stamp')
    plt.legend()

    # plt.subplot(2, 1, 2)
    # plt.plot(freq_x, mse, 'ko', freq_x, mse, 'k', markersize=2)
    # plt.axvline(freq[-5], linewidth=1, color='k')
    # plt.xlim(freq_x[0], freq_x[-1])
    # plt.ylabel('MSE')
    # plt.xlabel('Frequency')

    mse_y = mse
    # plot the same data on both axes
    ax = plt.subplot(2, 2, 3)
    ax2 = plt.subplot(2, 2, 4)

    ax.plot(freq_x, mse_y, 'ko', freq_x, mse_y, 'k')
    ax2.plot(freq_x, mse_y, 'ko', freq_x, mse_y, 'k')

    ax.set_ylabel('MSE')
    ax.set_xlim(freq_x[0], freq_x[5])
    ax2.set_xlim(freq_x[-10], freq_x[-1])

    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax.tick_params(labelright='off')
    ax2.yaxis.tick_right()

    ax2.axes.axvline(freq_x[-5], linestyle='--', linewidth=1, color='k')

    # This looks pretty good, and was fairly painless, but you can get that
    # cut-out diagonal lines look with just a bit more work. The important
    # thing to know here is that in axes coordinates, which are always
    # between 0-1, spine endpoints are at these locations (0,0), (0,1),
    # (1,0), and (1,1).  Thus, we just need to put the diagonals in the
    # appropriate corners of each of our axes, and so long as we use the
    # right transform and disable clipping.

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)

    # What's cool about this is that now if we vary the distance between
    # ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
    # the diagonal lines will move accordingly, and stay right at the tips
    # of the spines they are 'breaking'

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # pos = plt.gca().transData.transform((1.5, 1.8))
    # print(pos)
    # plt.text(pos[0], pos[1], 'Frequency', ha='center')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    exchangerate_plot()
    # exchangerate_plot_stack()

