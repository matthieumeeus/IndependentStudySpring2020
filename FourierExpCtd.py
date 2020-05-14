import numpy as np

experiments_sin = np.loadtxt('PlotsMay4/SinLosses.txt', delimiter=',')
experiments_sincos = np.loadtxt('PlotsMay4/SinCosLosses.txt', delimiter=',')
sincos_winners = 0

for i in range(20):
    print(i)
    sin_mean = np.mean(experiments_sin[i,:])
    sin_min = np.min(experiments_sin[i,:])
    sin_max = np.max(experiments_sin[i,:])
    print('For sine only: Mean = {} - Min = {} - Max = {}'.format(
        sin_mean, sin_min, sin_max))
    sincos_mean = np.mean(experiments_sincos[i,:])
    sincos_min = np.min(experiments_sincos[i,:])
    sincos_max = np.max(experiments_sincos[i,:])
    print('For sine & cosine: Mean = {} - Min = {} - Max = {}'.format(
        sincos_mean, sincos_min, sincos_max))
    if sincos_mean < sin_mean:
        perc = 100*np.round((sin_mean - sincos_mean)/sin_mean, 4)
        print('Sin Cos was better by {}%'.format(perc))
        sincos_winners += 1
    else:
        perc = 100*np.round((sincos_mean - sin_mean)/sincos_mean, 4)
        print('Sine was better by {}%'.format(perc))
    print('---------------------------------------------')

print(sincos_winners/20)