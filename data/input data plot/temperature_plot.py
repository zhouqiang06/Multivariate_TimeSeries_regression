import numpy as np
import pandas as pd

from sys import platform
import os
import matplotlib.pyplot as plt

if platform == "win32":
    home_path = r'C:\Users\qzhou\Downloads\phd\FreqDe'
else:
    home_path = r'/home/qzhou/FreqDe'

def temperature_plot():
    #####################temp piecewise model with lag######################
    file_path = os.path.join(home_path, 'Temperature1970001_201804.csv')

    cities = ['Colorado (Springs)', 'Maryland (Baltimore)', 'New Mexico (Roswell)', 'Los Angeles (California)']
    linestyles = ['-', '--', '-.', ':']
    temperature_data = np.asarray(pd.read_csv(file_path, delimiter=','))
    print(temperature_data.shape)

    subplot = 221
    # plt.figure(1)
    for i in range(temperature_data.shape[1]):
        # plt.subplot(subplot + i)
        plt.plot(temperature_data[:, i], linewidth=1, label=cities[i])#, color='k', linestyle=linestyles[i]
        # plt.title(cities[i])


    plt.xlabel('Time stamp')
    plt.ylabel('Temperature (F)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    temperature_plot()

