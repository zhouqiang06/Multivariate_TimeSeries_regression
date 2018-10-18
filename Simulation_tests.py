import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt

from sys import platform
import os

import logging

from FFTregressor.FFTregressor import FFTregressor

if platform == "win32":
    home_path = r'C:\Users\qzhou\Downloads\phd\FreqDe'
else:
    home_path = r'/home/qzhou/FreqDe' 

def temperature_pieceLagmodel_sim():
    """This function uses all temperature data to investigate the model capability 
    in rejecting redundant variables when all required variables are provided.
    """
    
    noise_lev = 0.05

    for noise_fac in range(0, 3):
        coefs = []
        r_square = []
        mse = []
        frq_range = []
        lags = []
        
        noise_lev = float(noise_fac) * 0.5
        x = np.array(range(0, 400)) / 50 * 2 * np.pi
        fftreg = FFTregressor()
        logging.info('temperature_pieceLagmodel_sim_noise_lev: {}'.format(noise_lev))
        
        file_path = os.path.join(home_path, 'Temperature1970001_201804.csv')
        
        temperature_data = np.asarray(pd.read_csv(file_path, delimiter=','))
        print(temperature_data.shape)
        y1 = temperature_data[:, 0]
        y2 = temperature_data[:, 1]
        y3 = temperature_data[:, 2]
        y4 = temperature_data[:, 3]

        y1FFT = np.fft.rfft(np.roll(y1, -1))
        y2FFT = np.fft.rfft(np.roll(y2, 1))
        y3FFT = np.fft.rfft(np.roll(y3, 2))

        yFFT = np.zeros_like(y1FFT, dtype=np.complex)
        
        yFFT[:100] = 2.3*y1FFT[:100] + 1.2*y2FFT[:100] + 0.8*y3FFT[:100]
        
        y1FFT = np.fft.rfft(np.roll(y1, 2))
        y2FFT = np.fft.rfft(np.roll(y2, -2))
        y4FFT = np.fft.rfft(np.roll(y4, 0))
        yFFT[100:] = -5.3*y1FFT[100:] + 2.6*y2FFT[100:] + 4.8*y4FFT[100:]
        y = np.fft.irfft(yFFT)
        np.savetxt(os.path.join(home_path,'temperature_pieceLagmodel_sim.txt'), np.asarray(y))

        if noise_lev > 0:
            y1 = y1 + np.random.normal(0, noise_lev * float(np.std(y1, dtype=np.float64)))
            y2 = y2 + np.random.normal(0, noise_lev * float(np.std(y2, dtype=np.float64)))
            y3 = y3 + np.random.normal(0, noise_lev * float(np.std(y3, dtype=np.float64)))
            y4 = y4 + np.random.normal(0, noise_lev * float(np.std(y4, dtype=np.float64)))
            y = y + np.random.normal(0, noise_lev * float(np.std(y, dtype=np.float64)))
        fftreg.fit_segwLag(np.vstack((y1, y2, y3, y4)).T, y, 2)
        # fftreg.fit_segwLag(np.vstack((y1, y2, y3)).T, y, 3, -2)

        [coefs.append(model.coefs) for model in fftreg.models]
        [frq_range.append([model.start, model.end]) for model in fftreg.models]
        [mse.append(model.mse) for model in fftreg.models]
        [lags.append(model.lags) for model in fftreg.models]
    #
        
        np.savetxt(os.path.join(home_path,'temperature_fullInput_sim_coefs(noise_lev_{}).txt'.format(noise_lev)), np.asarray(coefs))
        np.savetxt(os.path.join(home_path,'temperature_fullInput_sim_mse(noise_lev_{}).txt'.format(noise_lev)), np.asarray(mse))
        np.savetxt(os.path.join(home_path,'temperature_fullInput_sim_frq_range(noise_lev_{}).txt'.format(noise_lev)), np.asarray(frq_range))
        np.savetxt(os.path.join(home_path,'temperature_fullInput_sim_lags(noise_lev_{}).txt'.format(noise_lev)), np.asarray(lags))
    # print(len(err))


def temperature_pieceLagmodelIncomplete_sim():
    """This fucntion tests the method performance when both redundant and 
    missing variables occur and redundant variables are highly correlated with 
    the missing variable"""
    
    noise_lev = 0.05

    for noise_fac in range(0, 3):
        coefs = []
        r_square = []
        mse = []
        frq_range = []
        lags = []
        
        noise_lev = float(noise_fac) * 0.5
        x = np.array(range(0, 400)) / 50 * 2 * np.pi
        fftreg = FFTregressor()
        logging.info('temperature_pieceLagmodelIncomplete_sim_noise_lev: {}'.format(noise_lev))
        #####################temp piecewise model with lag######################
        file_path = os.path.join(home_path,'Temperature1970001_201804.csv')
            
        temperature_data = np.asarray(pd.read_csv(file_path, delimiter=','))
        print(temperature_data.shape)
        y1 = temperature_data[:, 0]
        y2 = temperature_data[:, 1]
        y3 = temperature_data[:, 2]
        y4 = temperature_data[:, 3]

        y1FFT = np.fft.rfft(np.roll(y1, -1))
        y2FFT = np.fft.rfft(np.roll(y2, 1))
        y3FFT = np.fft.rfft(np.roll(y3, 2))

        yFFT = np.zeros_like(y1FFT, dtype=np.complex)
        
        yFFT[:100] = 2.3*y1FFT[:100] + 1.2*y2FFT[:100] + 0.8*y3FFT[:100]
        
        y1FFT = np.fft.rfft(np.roll(y1, 2))
        y2FFT = np.fft.rfft(np.roll(y2, -2))
        y4FFT = np.fft.rfft(np.roll(y4, 0))
        yFFT[100:] = -5.3*y1FFT[100:] + 2.6*y2FFT[100:] + 4.8*y4FFT[100:]
        y = np.fft.irfft(yFFT)

        if noise_lev > 0:
            y1 = y1 + np.random.normal(0, noise_lev * float(np.std(y1, dtype=np.float64)))
            y2 = y2 + np.random.normal(0, noise_lev * float(np.std(y2, dtype=np.float64)))
            y3 = y3 + np.random.normal(0, noise_lev * float(np.std(y3, dtype=np.float64)))
            y4 = y4 + np.random.normal(0, noise_lev * float(np.std(y4, dtype=np.float64)))
            y = y + np.random.normal(0, noise_lev * float(np.std(y, dtype=np.float64)))
        fftreg.fit_segwLag(np.vstack((y1, y3, y4)).T, y, 2)
        out_path = os.path.join(os.path.dirname(file_path), 'models_noise_lev{}.txt'.format(noise_lev))
        output = open(out_path, 'w', encoding='utf-8')
        [output.write(str(model) + '\n') for model in fftreg.models]
        output.close()

        [coefs.append(model.coefs) for model in fftreg.models]
        [frq_range.append([model.start, model.end]) for model in fftreg.models]
        [mse.append(model.mse) for model in fftreg.models]
        [lags.append(model.lags) for model in fftreg.models]
    #
        print('frq_range: ', frq_range)
        np.savetxt(os.path.join(home_path,'temperature_incompleteInput_sim_coefs(noise_lev_{}).txt'.format(noise_lev)), np.asarray(coefs))
        np.savetxt(os.path.join(home_path,'temperature_incompleteInput_sim_mse(noise_lev_{}).txt'.format(noise_lev)), np.asarray(mse))
        np.savetxt(os.path.join(home_path,'temperature_incompleteInput_sim_frq_range(noise_lev_{}).txt'.format(noise_lev)), np.asarray(frq_range))
        np.savetxt(os.path.join(home_path,'temperature_incompleteInput_sim_lags(noise_lev_{}).txt'.format(noise_lev)), np.asarray(lags))
#    print(len(err))

def random_pieceLagmodelIncomplete_sim():
        """This fucntion tests the method performance when both redundant and 
    missing variables occur and redundant variables are not correlated with 
    the missing variable"""
    
    noise_lev = 0.05

    for noise_fac in range(0, 3):
        coefs = []
        r_square = []
        mse = []
        frq_range = []
        lags = []
        
        noise_lev = float(noise_fac) * 0.5
        fftreg = FFTregressor()
        logging.info('random_pieceLagmodelIncomplete_sim_noise_lev: {}'.format(noise_lev))
        #####################temp piecewise model with lag######################
        file_path = os.path.join(home_path,'Temperature1970001_201804.csv')
            
        temp_data = np.asarray(pd.read_csv(file_path, delimiter=','))
        random_data = np.random.rand(temp_data.shape[0], temp_data.shape[1])
        print(random_data.shape)
        y1 = random_data[:, 0]
        y2 = random_data[:, 1]
        y3 = random_data[:, 2]
        y4 = random_data[:, 3]

        y1FFT = np.fft.rfft(np.roll(y1, -1))
        y2FFT = np.fft.rfft(np.roll(y2, 1))
        y3FFT = np.fft.rfft(np.roll(y3, 2))

        yFFT = np.zeros_like(y1FFT, dtype=np.complex)
        
        yFFT[:100] = 2.3*y1FFT[:100] + 1.2*y2FFT[:100] + 0.8*y3FFT[:100]
        
        y1FFT = np.fft.rfft(np.roll(y1, 2))
        y2FFT = np.fft.rfft(np.roll(y2, -2))
        y4FFT = np.fft.rfft(np.roll(y4, 0))
        yFFT[100:] = -5.3*y1FFT[100:] + 2.6*y2FFT[100:] + 4.8*y4FFT[100:]
        y = np.fft.irfft(yFFT)

        if noise_lev > 0:
            y1 = y1 + np.random.normal(0, noise_lev * float(np.std(y1, dtype=np.float64)))
            y2 = y2 + np.random.normal(0, noise_lev * float(np.std(y2, dtype=np.float64)))
            y3 = y3 + np.random.normal(0, noise_lev * float(np.std(y3, dtype=np.float64)))
            y4 = y4 + np.random.normal(0, noise_lev * float(np.std(y4, dtype=np.float64)))
            y = y + np.random.normal(0, noise_lev * float(np.std(y, dtype=np.float64)))
        fftreg.fit_segwLag(np.vstack((y1, y3, y4)).T, y, 2)
        # print(fftreg.models)
        out_path = os.path.join(os.path.dirname(file_path), 'models_noise_lev{}.txt'.format(noise_lev))
        output = open(out_path, 'w', encoding='utf-8')
        # json.dump(fftreg.models, output)
        [output.write(str(model) + '\n') for model in fftreg.models]
        output.close()

        [coefs.append(model.coefs) for model in fftreg.models]
        [frq_range.append([model.start, model.end]) for model in fftreg.models]
        [mse.append(model.mse) for model in fftreg.models]
        [lags.append(model.lags) for model in fftreg.models]
    #
        print('frq_range: ', frq_range)
        np.savetxt(os.path.join(home_path,'random_incompleteInput_sim_coefs(noise_lev_{}).txt'.format(noise_lev)), np.asarray(coefs))
        np.savetxt(os.path.join(home_path,'random_incompleteInput_sim_mse(noise_lev_{}).txt'.format(noise_lev)), np.asarray(mse))
        np.savetxt(os.path.join(home_path,'random_incompleteInput_sim_frq_range(noise_lev_{}).txt'.format(noise_lev)), np.asarray(frq_range))
        np.savetxt(os.path.join(home_path,'random_incompleteInput_sim_lags(noise_lev_{}).txt'.format(noise_lev)), np.asarray(lags))
#    print(len(err))


if __name__ == '__main__':
    logging.basicConfig(filename='export_{}.log'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')),
                        level=logging.DEBUG)

    temperature_pieceLagmodel_sim()
    temperature_pieceLagmodelIncomplete_sim()
    
    random_pieceLagmodelIncomplete_sim()
