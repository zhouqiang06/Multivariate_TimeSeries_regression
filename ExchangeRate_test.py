import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from sys import platform
import os

import logging

from FFTregressor.FFTregressor import FFTregressor

if platform == "win32":
    home_path = r'C:\Users\qzhou\Downloads\phd\FreqDe'
else:
    home_path = r'/home/qzhou/FreqDe' 

def exchange_rate_real():
    """This fucntion considers the Australia exchange rate as the dependent 
    variable and explore its relationship with other countries"""
    
    file_path = os.path.join(home_path,'exchange_rate.csv')

    my_data = pd.read_csv(file_path, delimiter=',', parse_dates=True)
    my_array = np.asarray(my_data.iloc[:, :])
    freq = np.fft.rfftfreq(my_array.shape[0], d=1)
    print('Freq: ', freq)
    
    x_fft = np.fft.rfft(my_array[:, 1:], axis=0)

    freq_x = freq[:x_fft.shape[0] - x_fft.shape[1]][::-1]
#    freq_x = freq[:x_fft.shape[0] - x_fft.shape[1]]
    print('freq_x: ', freq_x)
    
    x_axis = np.asarray(range(my_array.shape[0])) + datetime.strptime('1990_01_01','%Y_%m_%d').toordinal()

    fftreg = FFTregressor()
#    logging.debug('build model')
    logging.info('exchange_rate_real')
    fftreg.fit_segwLag(my_array[:, 1:], my_array[:, 0], 2, cpu=20)

    out_path = os.path.join(os.path.dirname(file_path), 'exchange_rate_real_models.txt')
    output = open(out_path, 'w', encoding='utf-8')

    [output.write(str(model) + '\n') for model in fftreg.models]
    output.close()

    mse_L = []
    
    for i in range(x_fft.shape[0] - x_fft.shape[1] - 1, -1, -1):
        y_fft = np.fft.rfft(my_array[:, 0])
        y_fft[:i] = complex(0, 0)

        y_fft_hat = np.dot(x_fft, fftreg.models[0].coefs)
        y_fft_hat[:i] = complex(0, 0)

        y = np.fft.irfft(y_fft)
        y_hat = np.fft.irfft(y_fft_hat)
        mse_L.append(mean_squared_error(y, y_hat))
        
        if i == fftreg.models[0].start:
            np.savetxt(os.path.join(os.path.dirname(file_path), 'exchange_rate_real_models_y.txt'), np.asarray(y))
            np.savetxt(os.path.join(os.path.dirname(file_path), 'exchange_rate_real_models_y_hat.txt'), y_hat)
    np.savetxt(os.path.join(os.path.dirname(file_path), 'exchange_rate_real_models_mse_test.txt'), np.asarray(mse_L))
    np.savetxt(os.path.join(os.path.dirname(file_path), 'exchange_rate_real_models_freq.txt'), freq)
    
    ########################Break // in x axis of matplotlib#########################################

    mse_y = mse_L
    # plot the same data on both axes
    f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')
    
    ax.plot(freq_x, mse_y, 'ko', freq_x, mse_y, 'k')
    ax2.plot(freq_x, mse_y, 'ko', freq_x, mse_y, 'k')
    
    ax.set_xlim(freq_x[0],freq_x[5])
    ax2.set_xlim(freq_x[-10],freq_x[-1])
    
    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax.tick_params(labelright='off')
    ax2.yaxis.tick_right()
    
    ax2.axes.axvline(freq_x[-5], linewidth=1, color='k')
    
   
    # This looks pretty good, and was fairly painless, but you can get that
    # cut-out diagonal lines look with just a bit more work. The important
    # thing to know here is that in axes coordinates, which are always
    # between 0-1, spine endpoints are at these locations (0,0), (0,1),
    # (1,0), and (1,1).  Thus, we just need to put the diagonals in the
    # appropriate corners of each of our axes, and so long as we use the
    # right transform and disable clipping.
    
    d = .015 # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1-d,1+d), (-d,+d), **kwargs)
    ax.plot((1-d,1+d),(1-d,1+d), **kwargs)
    
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    ax2.plot((-d,+d), (-d,+d), **kwargs)
    
    # What's cool about this is that now if we vary the distance between
    # ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
    # the diagonal lines will move accordingly, and stay right at the tips
    # of the spines they are 'breaking'
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.show()
    ########################################################################################################
    
    x_fit = np.zeros_like(my_array[:, 1:])
    x_sub = np.zeros_like(my_array[:, 1:])
    for model in fftreg.models:
        y_fft = np.fft.fft(my_array[:, 0])
        y_fft[0] = 0
        # y_fft = y_fft[1:]

        fft_idx = np.zeros(y_fft.shape, dtype=np.bool)
        start = model.start
        end = model.end
        logging.info('start, end: {} {}; {}, {}'.format(start, end, (len(y_fft) - end), (len(y_fft) - start)))
        fft_idx[start:end] = True
        fft_idx[(len(y_fft) - end):(len(y_fft) - start)] = True

        x_fft = np.fft.fft(my_array[:, 1:],axis=0)
        x_fft[0, :] = 0
        # x_fft = x_fft[1:, :]
        y_fft_plt = np.dot(x_fft, model.coefs)

        y_fft_plt[~fft_idx] = complex(0, 0)
        y_fft[~fft_idx] = complex(0, 0)
        # np.savetxt(r'C:\Users\qzhou\Downloads\phd\FreqDe\y_fft_sub.txt', abs(y_fft))
        # np.savetxt(r'C:\Users\qzhou\Downloads\phd\FreqDe\freq.txt', freq)

        logging.info('model start: {},model end: {},step start: {},step start: {}'.format(model.start, model.end, 1/freq[model.start], 1/freq[model.end]))
        logging.info('model coefs: {}'.format(model.coefs))
        logging.info('model lags: {}'.format(model.lags))
        # print('model coefs amplitude: ', np.abs(model.coefs))
        # print('model coefs phase: ', np.angle(model.coefs))
        logging.info('MSE: {}'.format(model.mse))
        logging.info('r_square: {}'.format(model.r_square))
        # print('R2: ', r2_score(np.fft.ifft(y_fft_plt).real, np.fft.ifft(y_fft).real))
        haty = np.fft.ifft(y_fft_plt).real
        y = np.fft.ifft(y_fft).real
        # print('std.err: ', np.std(y - haty))
        # print('MRE: ', np.mean(abs((y - haty)/y)))
        plt.plot(x_axis, y, 'k-', label='True')
        plt.plot(x_axis, haty,'k--', label='predicted')
        plt.ylabel('%')
        plt.xlabel('Year')
        plt.legend()
        plt.show()
        
        for i in range(len(model.coefs)):
            temp_fit = np.zeros_like(y_fft, dtype=np.complex)
            temp_fit[fft_idx] = x_fft[fft_idx, i] * model.coefs[i]
            x_fit[:, i] += np.fft.ifft(temp_fit).real
            temp_fit = np.zeros_like(y_fft, dtype=np.complex)
            temp_fit[fft_idx] = x_fft[fft_idx, i]
            x_sub[:, i] += np.fft.ifft(temp_fit).real

        x_fill = np.fft.fft(x_sub, axis=0)
        x_fill[0, :] = np.fft.fft(my_array[:, 1:], axis=0)[0, :]
        x_sub = np.fft.ifft(x_fill, axis=0).real



if __name__ == '__main__':
    logging.basicConfig(filename='export_{}.log'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')),
                        level=logging.DEBUG)

    exchange_rate_real()
