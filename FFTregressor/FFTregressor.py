#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:47:47 2018

@author: qzhou
"""

import numpy as np
from sklearn import linear_model
from collections import namedtuple

from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats
from scipy.optimize import minimize_scalar
import itertools
import matplotlib.pyplot as plt
import logging


# log = logging.getLogger()
# handler = logging.StreamHandler(sys.stdout)
# formatter = logging.Formatter('%(asctime)s %(processName)s: %(message)s')
# handler.setFormatter(formatter)
# log.addHandler(handler)
# log.setLevel(logging.DEBUG)

def outliers_modified_z_score(ys):
    threshold = 3.5

    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    return np.any(np.abs(modified_z_scores) > threshold)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def SSR(residuals): #sum of squared residuals
    return np.sum(np.square(residuals))

def SSR_complex(coef1, coef2): #sum of squared residuals
    par1 = np.hstack((np.real(coef1), np.imag(coef1)))
    par2 = np.hstack((np.real(coef2), np.imag(coef2)))
    return np.sum(np.square(par1 - par2))

def nor_TS(ts):
    dep_nor = np.fft.fft(ts)
    dep_nor[:2] = 0
    dep_nor = np.fft.ifft(dep_nor)

    # dep_nor = (dep_nor - np.mean(dep_nor))/np.std(dep_nor)
    dep_nor = (dep_nor - np.min(dep_nor)) / (np.max(dep_nor) - np.min(dep_nor))

    return dep_nor

def find_lagExp(unit_fft, k, dep_fft, indep_fft):
    res = dep_fft - (unit_fft ** k) * indep_fft
    return(np.sum(np.square(abs(res))))

def find_lagAng(unit_fft, k, dep_fft, indep_fft):
    clf = FFTregressor()
    indep_fft_rot = (unit_fft ** k) * indep_fft
    clf.fit(indep_fft_rot[:, np.newaxis], dep_fft)
    # res = dep_fft / (unit_fft ** k) * indep_fft
    return(abs(np.imag(clf.models[0].coefs)))

def plot_lagCriteria(dep, indep, lagRange, lagStep):
    # dep_fft = np.fft.rfft(nor_TS(dep))
    # indep_fft = np.fft.rfft(nor_TS(indep))
    dep_fft = np.fft.rfft(dep)
    indep_fft = np.fft.rfft(indep)
    # increasement = np.pi * 2 / len(dep)
    # print('calculated increasement: ',increasement)
    phase_dif = np.angle(np.fft.rfft(indep)) - np.angle(np.fft.rfft(np.roll(indep, 1)))

    increasement = phase_dif[1]
    if len(indep) % 2 == 0:
        unit_array = np.asarray(
            [complex(np.cos(a * increasement), np.sin(a * increasement)) for a in range(0, int(len(dep) / 2))])
    else:
        unit_array = np.asarray(
            [complex(np.cos(a * increasement), np.sin(a * increasement)) for a in range(0, 1 + int(len(dep) / 2))])

    criteriaL = [find_lagExp(unit_array, k, dep_fft, indep_fft) for k in np.arange(lagRange[0], lagRange[1], lagStep)]
    plt.plot(np.arange(lagRange[0], lagRange[1], lagStep), criteriaL, 'k')
    plt.ylabel('imaginary part')
    plt.xlabel('lag')
    plt.show()


def find_lagTS(dep, indep):

    dep_fft = np.fft.rfft(dep)
    indep_fft = np.fft.rfft(indep)

    phase_dif = np.angle(np.fft.rfft(indep)) - np.angle(np.fft.rfft(np.roll(indep, 1)))

    increasement = phase_dif[1]

    unit_array = np.asarray(
        [complex(np.cos(a * increasement), np.sin(a * increasement)) for a in range(0, len(indep_fft))])

    lag = minimize_scalar(lambda k: find_lagAng(unit_array, k, dep_fft, indep_fft))
    logging.info('best lag: ', lag.x, find_lagAng(unit_array, lag.x, dep_fft, indep_fft))
    return np.fft.irfft(unit_array ** lag.x * np.fft.rfft(indep)).real

model = namedtuple('model', ['start', 'end', 'lags', 'coefs', 'mse', 'r_square'])

class FFTregressor:
    
    models = []

    def fit(self, X, y, freq=[], d = 1.0): 

        x_fft = np.zeros((X.shape[0], X.shape[1]), dtype=np.complex)
        for i in range(X.shape[1]):
            x_fft[:, i] = np.fft.rfft(X[:, i])

        y_fft = np.fft.rfft(y, axis=0)

        coef_ = self.fft_fit(x_fft, y_fft)
        fft_pred = np.dot(x_fft, coef_)
        accr_cur = r2_score(np.real(np.fft.irfft(y_fft)), np.real(np.fft.irfft(fft_pred)))
        mse = mean_squared_error(np.real(np.fft.irfft(y_fft)), np.real(np.fft.irfft(fft_pred)))

        self.models = []
        self.models.append(model(start=0,
                                      end=len(y_fft),
                                      lags=[0] * X.shape[1],
                                      coefs=coef_,
                                      r_square=accr_cur,
                                      mse=mse))

        return True

    def fft_fit(self, freqX, freqy):
        x_independent = np.zeros((freqX.shape[0] * 2, freqX.shape[1] * 2))
        # print(freqX.shape)
        for i in range(freqX.shape[1]):
            x_independent[:, i * 2] = np.hstack((np.real(freqX[:, i]), np.imag(freqX[:, i])))
            x_independent[:, i * 2 + 1] = np.hstack((-1 * np.imag(freqX[:, i]), np.real(freqX[:, i])))

        y_dependent = np.hstack((np.real(freqy), np.imag(freqy)))

        clf = linear_model.Lasso(fit_intercept=False, max_iter= 10000)
        # clf = linear_model.LinearRegression()
        clf.fit(x_independent, y_dependent)

        coef_ = []
        for idx in range(0, len(clf.coef_), 2):
            coef_.append(complex(clf.coef_[idx], clf.coef_[idx + 1]))

        return np.asarray(coef_)

    def fft_SSE(self, freqX, coefs, freqy):
        x_independent = np.zeros((freqX.shape[0] * 2, freqX.shape[1] * 2))
        x_coef = np.zeros((freqX.shape[1] * 2))
        # print(freqX.shape)
        for i in range(freqX.shape[1]):
            x_independent[:, i * 2] = np.hstack((np.real(freqX[:, i]), np.imag(freqX[:, i])))
            x_independent[:, i * 2 + 1] = np.hstack((-1 * np.imag(freqX[:, i]), np.real(freqX[:, i])))
            x_coef[i * 2] = np.real(coefs[i])
            x_coef[i * 2 + 1] = np.imag(coefs[i])
        y_pred = np.dot(x_independent, x_coef)

        y_dependent = np.hstack((np.real(freqy), np.imag(freqy)))
        print('x_independent, x_coef, y_pred, y_dependent shape:', x_independent.shape, x_coef.shape, y_pred.shape, y_dependent.shape)

        return np.sum(np.square(y_dependent - y_pred))

    def genLaglist(self, minL, maxL):
        lagList = []
        for minlag, maxlag in zip(minL, maxL):
            lag_var = []
            for lagtest in range(minlag, maxlag+1):
                lag_var.append(int(lagtest))
            lagList.append(lag_var)
        lagList = list(itertools.product(*lagList))
        return lagList

    def getLagedFFT(self, X, lagperm):
        x_lag_fft = np.fft.rfft(X, axis=0)

        for xcol in range(X.shape[1]):
            x_lag_fft[:, xcol] = np.fft.rfft(np.roll(X[:, xcol], lagperm[xcol]), axis=0)
        # x_lag_fft = x_lag_fft[1:, :]
        return x_lag_fft
    
    def getmse_lag(self, X, lags, y_fft, fft_idx, alpha):
        x_lag_fft = self.getLagedFFT(X, lags)
        fft_coef_sub = self.fft_fit(x_lag_fft[fft_idx, :], y_fft[fft_idx])
        fft_pred_sub = np.zeros(y_fft.shape, dtype=np.complex)
        fft_pred_sub[fft_idx] = np.dot(x_lag_fft[fft_idx, :], fft_coef_sub)
        fft_y_sub = np.zeros(y_fft.shape, dtype=np.complex)
        fft_y_sub[fft_idx] = y_fft[fft_idx]
        return alpha * np.mean(np.angle(fft_coef_sub) ** 2) + (1 - alpha) * mean_squared_error(
            np.real(np.fft.irfft(fft_y_sub)), np.real(np.fft.irfft(fft_pred_sub)))

    def get_criteria(self, X, lags, y_fft, fft_idx, alpha):
        x_lag_fft = self.getLagedFFT(X, lags)
        fft_coef_sub = self.fft_fit(x_lag_fft[fft_idx, :], y_fft[fft_idx])
        fft_pred_sub = np.zeros(y_fft.shape, dtype=np.complex)
        fft_pred_sub[fft_idx] = np.dot(x_lag_fft[fft_idx, :], fft_coef_sub)
        fft_y_sub = np.zeros(y_fft.shape, dtype=np.complex)
        fft_y_sub[fft_idx] = y_fft[fft_idx]
        return mean_squared_error(np.real(np.fft.irfft(fft_y_sub)), np.real(np.fft.irfft(fft_pred_sub)))

    def findBestLag(self, X, y, fft_idx, minL, maxL, alpha=1.0, cpu=1):
        # alpha = 0.5
        minL = np.asarray(minL)
        maxL = np.asarray(maxL)
        searthwidth = np.asarray((maxL - minL) / 2).astype(int)
#        pool = Pool(processes=cpu)
        mse_cur = np.max(y)
        mse_loop = mse_cur
        lag_cur = [0] * X.shape[1]
#        fft_coef_cur = 0
        x_lag_fft_cur = np.fft.rfft(X, axis=0)
        y_fft = np.fft.rfft(y, axis=0)

        keepsearchLag = True
        while keepsearchLag == True:
            logging.debug('maxL, minL: {}, {}'.format(maxL, minL))
            print('maxL, minL: {}, {}'.format(maxL, minL))
            lagList = self.genLaglist(minL, maxL)

            for lagperm in lagList:
                x_lag_fft = self.getLagedFFT(X, lagperm)
                fft_coef_sub = self.fft_fit(x_lag_fft[fft_idx, :], y_fft[fft_idx])
                fft_pred_sub = np.zeros(y_fft.shape, dtype=np.complex)
                fft_pred_sub[fft_idx] = np.dot(x_lag_fft[fft_idx, :], fft_coef_sub)
                fft_y_sub = np.zeros(y_fft.shape, dtype=np.complex)
                fft_y_sub[fft_idx] = y_fft[fft_idx]
                # accr_cur = r2_score(np.real(np.fft.ifft(fft_y_sub)), np.real(np.fft.ifft(fft_pred_sub)))
                # mse_temp = mean_squared_error(np.real(np.fft.irfft(fft_y_sub)), np.real(np.fft.irfft(fft_pred_sub)))
                mse_temp = alpha * np.mean(np.imag(fft_coef_sub) ** 2) + (1 - alpha) * mean_squared_error(
                    np.real(np.fft.irfft(fft_y_sub)), np.real(np.fft.irfft(fft_pred_sub)))
                if mse_loop > mse_temp: #found better senario in current lag combination
                    lag_cur = lagperm
                    x_lag_fft_cur = x_lag_fft
                    mse_loop = mse_temp
                    fft_coef_cur = fft_coef_sub

            if mse_loop >= mse_cur: # if the current lag combination doesn't improve the historical mse
                break 
            else:
                mse_cur = mse_loop
            keepsearchLag = False
            for iL in range(len(lag_cur)): # update the lag range
                if maxL[iL] == lag_cur[iL]:
                    maxL[iL] = maxL[iL] + searthwidth[iL]
                    minL[iL] = minL[iL] + searthwidth[iL]
                    keepsearchLag = True
                if minL[iL] == lag_cur[iL]:
                    maxL[iL] = maxL[iL] - searthwidth[iL]
                    minL[iL] = minL[iL] - searthwidth[iL]
                    keepsearchLag = True

            lag_lim = int(len(fft_idx) - np.argwhere(fft_idx)[0]) # set the maximum lag search range as the highest period 
            if np.max(abs(np.asarray(maxL))) >= lag_lim or np.max(abs(np.asarray(minL))) >= lag_lim:
                break
#        logging.debug('MSE, coefs: {} {}'.format(mse_cur, fft_coef_cur))
        logging.debug('MSE: {}'.format(mse_cur))
        return x_lag_fft_cur, lag_cur

    def fit_seg(self, X, y):

        x_fft = np.fft.rfft(X, axis=0)
        y_fft = np.fft.rfft(y, axis=0)

        ################Use local maximum to find segment cut#############################
        shift_idx = 0
        end_idx = int(0.5 * len(y_fft))
        search_idx = shift_idx + X.shape[1] + 1
        logging.debug('search end: {}'.format(end_idx))
        self.models = []
        while shift_idx < int(0.5 * len(y_fft)):
            if (shift_idx + X.shape[1] + 1) >= int(0.5 * len(y_fft)):
                break
            for search_idx in range(shift_idx + X.shape[1] + 1, int(0.5 * len(y_fft))): #search local maximum R2

                ###################post frequency#############################################
                post_idx = search_idx + 1
                fft_idx = np.zeros(y_fft.shape, dtype=np.bool)
                fft_idx[shift_idx:post_idx] = True
                # fft_idx[(len(y_fft) - post_idx):(len(y_fft) - shift_idx)] = True
                fft_coef_sub = self.fft_fit(x_fft[fft_idx, :], y_fft[fft_idx])

                fft_pred_sub = np.zeros(y_fft.shape, dtype=np.complex)
                fft_pred_sub[fft_idx] = np.dot(x_fft[fft_idx, :], fft_coef_sub)
                fft_y_sub = np.zeros(y_fft.shape, dtype=np.complex)
                fft_y_sub[fft_idx] = y_fft[fft_idx]
                accr_post = r2_score(np.real(np.fft.irfft(fft_y_sub)), np.real(np.fft.irfft(fft_pred_sub)))
                mse_post = mean_squared_error(np.real(np.fft.irfft(fft_y_sub)), np.real(np.fft.irfft(fft_pred_sub)))

                #################Current search freqruency#################################
                fft_idx = np.zeros(y_fft.shape, dtype=np.bool)
                fft_idx[shift_idx:search_idx] = True
                # fft_idx[(len(y_fft) - search_idx):(len(y_fft) - shift_idx)] = True
                fft_coef_sub = self.fft_fit(x_fft[fft_idx, :], y_fft[fft_idx])

                fft_pred_sub = np.zeros(y_fft.shape, dtype=np.complex)
                fft_pred_sub[fft_idx] = np.dot(x_fft[fft_idx, :], fft_coef_sub)
                fft_y_sub = np.zeros(y_fft.shape, dtype=np.complex)
                fft_y_sub[fft_idx] = y_fft[fft_idx]
                accr_cur = r2_score(np.real(np.fft.irfft(fft_y_sub)), np.real(np.fft.irfft(fft_pred_sub)))
                mse_cur = mean_squared_error(np.real(np.fft.irfft(fft_y_sub)), np.real(np.fft.irfft(fft_pred_sub)))

                # print('from, to, R2, MSE, fft_coef_sub ', shift_idx, search_idx, accr_cur, mse_cur, fft_coef_sub)

                # if accr_cur > accr_pre and accr_cur > accr_post: #found local maximum
                if np.round(mse_cur, decimals=3) < np.round(mse_post, decimals=3) or search_idx == int(0.5 * len(y_fft))-1:  # found MSE increasing or the end of frequency
                    # print('local maximum found', shift_idx, end_idx, search_idx)
                    end_idx = search_idx
                    break

            self.models.append(model(start=shift_idx + 1,
                                          end=end_idx + 1,
                                          lags=[0] * X.shape[1],
                                          coefs=fft_coef_sub,
                                          mse=mse_cur,
                                          r_square=accr_cur))

            shift_idx = end_idx + 1
            ##################Commission test to merge models#######################
            if len(self.models) >=2: #if there are more than 1 models
                cur_model = self.models[-1]
                pre_model = self.models[-2]

                ###################previous model#############################################
                end = pre_model.end - 1
                start = pre_model.start - 1
                fft_idx = np.zeros(y_fft.shape, dtype=np.bool)
                fft_idx[start:end] = True
                fft_idx[(len(y_fft) - end):(len(y_fft) - start)] = True

                fft_pred_sub = np.zeros(y_fft.shape, dtype=np.complex)
                fft_pred_sub[fft_idx] = np.dot(x_fft[fft_idx, :], pre_model.coefs)
                fft_y_sub = np.zeros(y_fft.shape, dtype=np.complex)
                fft_y_sub[fft_idx] = y_fft[fft_idx]
                # S1 = SSR(np.real(np.fft.ifft(fft_y_sub)) - np.real(np.fft.ifft(fft_pred_sub)))
                S1 = SSR_complex(y_fft[fft_idx], fft_pred_sub[fft_idx])

                ###################current frequency#############################################
                end = cur_model.end - 1
                start = cur_model.start - 1
                fft_idx = np.zeros(y_fft.shape, dtype=np.bool)
                fft_idx[start:end] = True
                fft_idx[(len(y_fft) - end):(len(y_fft) - start)] = True

                fft_pred_sub = np.zeros(y_fft.shape, dtype=np.complex)
                fft_pred_sub[fft_idx] = np.dot(x_fft[fft_idx, :], cur_model.coefs)
                fft_y_sub = np.zeros(y_fft.shape, dtype=np.complex)
                fft_y_sub[fft_idx] = y_fft[fft_idx]
                # S2 = SSR(np.real(np.fft.ifft(fft_y_sub)) - np.real(np.fft.ifft(fft_pred_sub)))
                S2 = SSR_complex(y_fft[fft_idx], fft_pred_sub[fft_idx])

                ###################overall frequency#############################################
                end = cur_model.end - 1
                start = pre_model.start - 1
                fft_idx = np.zeros(y_fft.shape, dtype=np.bool)
                fft_idx[start:end] = True
                fft_idx[(len(y_fft) - end):(len(y_fft) - start)] = True
                fft_coef_sub = self.fft_fit(x_fft[fft_idx, :], y_fft[fft_idx])
                fft_pred_sub = np.zeros(y_fft.shape, dtype=np.complex)
                fft_pred_sub[fft_idx] = np.dot(x_fft[fft_idx, :], fft_coef_sub)
                fft_y_sub = np.zeros(y_fft.shape, dtype=np.complex)
                fft_y_sub[fft_idx] = y_fft[fft_idx]
                # Sc = SSR(np.real(np.fft.ifft(fft_y_sub)) - np.real(np.fft.ifft(fft_pred_sub)))
                Sc = SSR_complex(y_fft[fft_idx], fft_pred_sub[fft_idx])

                # N1 = len(fft_y_sub)
                # N2 = N1
                N1 = (cur_model.end - cur_model.start + 1)
                N2 = (pre_model.end - pre_model.start + 1)
                k = X.shape[1]

                chow_score = (Sc - (S1 + S2))/k / ((S1 + S2)/(N1 + N2 - 2 * k))
                # print('Sc, S1, S2, N1, N2, k', Sc, S1, S2, N1, N2, k)
                # print('chow_test:', chow_score, scipy.stats.f.ppf(q=1-0.001, dfn=k, dfd=(N1 + N2 - 2 * k)))
                if chow_score < 0 or chow_score < scipy.stats.f.ppf(q=1-0.001, dfn=k, dfd=(N1 + N2 - 2 * k)): #merge two models
                    self.models = self.models[:-2]
                    accr_cur = r2_score(np.real(np.fft.irfft(fft_y_sub)), np.real(np.fft.irfft(fft_pred_sub)))
                    mse = mean_squared_error(np.real(np.fft.irfft(fft_y_sub)), np.real(np.fft.irfft(fft_pred_sub)))
                    # fft_coef_sub = self.fft_fit(x_fft[fft_idx, :], y_fft[fft_idx])
                    self.models.append(model(start=start + 1,
                                                  end=end + 1,
                                                  lags=[0] * X.shape[1],
                                                  coefs=fft_coef_sub,
                                                  mse=mse,
                                                  r_square=accr_cur))

        return True

    def fit_segwLag(self, X, y, searchWidth, window_size=None, alpha=1.0, chow_q=1-0.001, cpu=1):
        # alpha = 0.5
        # alpha = 0
        if window_size is None:
            window_size = X.shape[1] + 1
        print('window_size', window_size)
        # x_fft = np.fft.rfft(X, axis=0)
        y_fft = np.fft.rfft(y, axis=0)

        ################Use local maximum to find segment cut#############################
        shift_idx = len(y_fft)-1
        end_idx = 0
        # search_idx = shift_idx - X.shape[1] - 1
        logging.debug('search length: {}'.format(shift_idx))
        self.models = []
        while shift_idx > 0:
            if (shift_idx - window_size) <= 0:
                break
            ###############find best lag for the initial model##########################
            search_idx = shift_idx - window_size
            fft_idx = np.zeros(y_fft.shape, dtype=np.bool)
            fft_idx[search_idx:shift_idx] = True

            fft_idx = np.zeros(y_fft.shape, dtype=np.bool)
            fft_idx[search_idx:shift_idx] = True
            x_lag_fft_cur, lag_cur = self.findBestLag(X, y, fft_idx, [-1 * searchWidth] * X.shape[1],
                                                      [searchWidth] * X.shape[1], cpu=cpu)
            logging.debug('initial lags: {}'.format(lag_cur))

            for search_idx in range(shift_idx - window_size, -1, -1): #search local maximum R2

                ###################post frequency#############################################
                if search_idx > 0:
                    post_idx = search_idx - 1
                    fft_idx = np.zeros(y_fft.shape, dtype=np.bool)
                    fft_idx[post_idx:shift_idx] = True

                    criteria_post = self.get_criteria(X, lag_cur, y_fft, fft_idx, alpha)
                    # print('mse compare: ', self.getmse(X, lag_cur, y_fft, fft_idx, alpha), criteria_post)

                #################Current search freqruency#################################
                fft_idx = np.zeros(y_fft.shape, dtype=np.bool)
                fft_idx[search_idx:shift_idx] = True

                criteria_cur = self.get_criteria(X, lag_cur, y_fft, fft_idx, alpha)

                if np.round(criteria_cur, decimals=3) < np.round(criteria_post, decimals=3) or search_idx == 0:  # found MSE increasing or the end of frequency
                    # print('local maximum found', shift_idx, end_idx, search_idx)
                    end_idx = search_idx
                    break

            #################update lag and the rest#########################
            logging.debug('update lag')
            fft_idx = np.zeros(y_fft.shape, dtype=np.bool)
            fft_idx[end_idx:(shift_idx + 1)] = True
            x_lag_fft_cur, lag_cur = self.findBestLag(X, y, fft_idx, [-1 * searchWidth] * X.shape[1],
                                                      [searchWidth] * X.shape[1], cpu=cpu)
            logging.info('final lag: {}'.format(lag_cur))
            fft_coef_sub = self.fft_fit(x_lag_fft_cur[fft_idx, :], y_fft[fft_idx])
            fft_pred_sub = np.zeros(y_fft.shape, dtype=np.complex)
            fft_pred_sub[fft_idx] = np.dot(x_lag_fft_cur[fft_idx, :], fft_coef_sub)
            fft_y_sub = np.zeros(y_fft.shape, dtype=np.complex)
            fft_y_sub[fft_idx] = y_fft[fft_idx]
            accr_cur = r2_score(np.real(np.fft.irfft(fft_y_sub)), np.real(np.fft.irfft(fft_pred_sub)))
            mse_cur = mean_squared_error(np.real(np.fft.irfft(fft_y_sub)), np.real(np.fft.irfft(fft_pred_sub)))
            ##################################################################
            logging.info('current model: {} to {}'.format(end_idx, shift_idx))
            print('current model: {} to {}'.format(end_idx, shift_idx))
            self.models.insert(0, model(start=end_idx,
                                          end=shift_idx,
                                          lags=tuple(lag_cur),
                                          coefs=fft_coef_sub,
                                          mse=mse_cur,
                                          r_square=accr_cur))

            shift_idx = end_idx - 1
            ##################Commission test to merge models#######################
            if len(self.models) >=2: #if there are more than 1 models
                cur_model = self.models[0]
                pre_model = self.models[1]

                ###################previous model#############################################
                end = pre_model.end
                start = pre_model.start
                print('previous model: ', start, end)
                fft_idx = np.zeros(y_fft.shape, dtype=np.bool)
                fft_idx[start:(end + 1)] = True

                x_lag_fft = self.getLagedFFT(X, pre_model.lags)
                S1 = self.fft_SSE(x_lag_fft[fft_idx, :], pre_model.coefs, y_fft[fft_idx])

                ###################current frequency#############################################
                end = cur_model.end
                start = cur_model.start
                fft_idx = np.zeros(y_fft.shape, dtype=np.bool)
                fft_idx[start:(end + 1)] = True

                x_lag_fft = self.getLagedFFT(X, cur_model.lags)

                S2 = self.fft_SSE(x_lag_fft[fft_idx, :], cur_model.coefs, y_fft[fft_idx])

                ###################overall frequency#############################################
                start = cur_model.start
                end = pre_model.end
                print('overall model: ', start, end)
                fft_idx = np.zeros(y_fft.shape, dtype=np.bool)
                fft_idx[start:(end + 1)] = True
                # fft_idx[(len(y_fft) - end):(len(y_fft) - start)] = True
                x_lag_fft_cur, lag_cur = self.findBestLag(X, y, fft_idx, [-1 * searchWidth] * X.shape[1], [searchWidth] * X.shape[1], cpu=cpu)

                fft_coef_sub = self.fft_fit(x_lag_fft_cur[fft_idx, :], y_fft[fft_idx])

                Sc = self.fft_SSE(x_lag_fft_cur[fft_idx, :], fft_coef_sub, y_fft[fft_idx])

                N1 = (cur_model.end - cur_model.start + 1) * 2
                N2 = (pre_model.end - pre_model.start + 1) * 2
                k = X.shape[1] * 2

                chow_score = (Sc - (S1 + S2))/k / ((S1 + S2)/(N1 + N2 - 2 * k))
                print('Sc, S1, S2, N1, N2, k', Sc, S1, S2, N1, N2, k)
                print('chow_test:', chow_score, scipy.stats.f.ppf(q=chow_q, dfn=k, dfd=(N1 + N2 - 2 * k)))
                if chow_score < 0 or chow_score < scipy.stats.f.ppf(q=chow_q, dfn=k, dfd=(N1 + N2 - 2 * k)): #merge two models
                    logging.info('merge model {} to {}'.format(start, end))
                    self.models = self.models[2:]
                    accr_cur = r2_score(np.real(np.fft.irfft(fft_y_sub)), np.real(np.fft.irfft(fft_pred_sub)))
                    mse = mean_squared_error(np.real(np.fft.irfft(fft_y_sub)), np.real(np.fft.irfft(fft_pred_sub)))
                    # fft_coef_sub = self.fft_fit(x_fft[fft_idx, :], y_fft[fft_idx])
                    self.models.insert(0, model(start=start,
                                                  end=end,
                                                  lags=tuple(lag_cur),
                                                  coefs=fft_coef_sub,
                                                  mse=mse,
                                                  r_square=accr_cur))


        return True

    # def predict_fft_seg(self, X):

    # def predict_seg(self, X):

#    def predict_fft(self, X):
#        pred = np.zeros(X.shape[0])
#        for i in range(X.shape[1]):
#            pred += np.fft.rfft(X[:, i]) * self.coef_[i]
#        return pred
#
#    def predict(self, X):
#        return np.fft.irfft(predict_fft(X))