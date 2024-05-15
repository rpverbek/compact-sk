# ©, 2022, Sirris
# owner: FFNG

from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

V = [1413, 2966, 4393, 5156, 5653]
Q_min = [75, 100, 135, 155, 170]  # Q-min flow valve starts opening
H_min = [295, 1090, 2300, 3160, 3780]  # H at corresponding Q_min and speed

# polinomial interpolation: f_min(Q_min) = H_min
# f_min = KroghInterpolator(Q_min, H_min)
f_min = interpolate.interp1d(Q_min, H_min, kind='quadratic')
Q_min_interpolated = np.arange(min(Q_min), max(Q_min), 1)
H_min_interpolated = f_min(Q_min_interpolated)

Q_max = [225, 430, 630, 715, 750]  # Q-min flow valve starts opening
H_max = [190, 660, 1400, 2000, 2500]  # H at corresponding Q_min and speed

# polinomial interpolation: f_min(Q_max) = H_max
# f_max = KroghInterpolator(Q_max, H_max)
f_max = interpolate.interp1d(Q_max, H_max, kind='quadratic')
Q_max_interpolated = np.arange(min(Q_max), max(Q_max), 1)
H_max_interpolated =f_max(Q_max_interpolated)

Q_shutoff = [0, 0, 0, 0, 0]
H_shutoff = [300, 1110, 2360, 3230, 3860]

fs = {}
pump_curves = {}
q_interpolated = {}
for v, q, h in zip(V, zip(Q_shutoff, Q_min, Q_max), zip(H_shutoff, H_min, H_max)):
    fs[v] = interpolate.interp1d(q, h, kind='quadratic')
    q_interpolated[v] = np.arange(min(q),max(q), 1)
    pump_curves[v] = fs[v](q_interpolated[v])

V_BEP = [0, 5652]
Q_BEP = [0, 611.65]
H_BEP = [0, 2913.7]

f_BEP = interpolate.interp1d(Q_BEP, H_BEP, kind='linear')
Q_BEP_interpolated = np.arange(min(Q_BEP),max(Q_BEP), 1)
H_BEP_interpolated =f_BEP(Q_BEP_interpolated)


def plot_operating_range(ax=None, show_pump_speed=False, show_legend=True, markersize=None):
    ax = ax or plt.subplots(figsize=(12, 8))[1]
    ax.plot(Q_min_interpolated, H_min_interpolated, label='Q_min')
    ax.scatter(Q_min, H_min, s=markersize)
    ax.plot(Q_max_interpolated, H_max_interpolated, label='Q_max')
    ax.scatter(Q_max, H_max, s=markersize)
    ax.plot(Q_BEP_interpolated, H_BEP_interpolated, label='BEPs')
    ax.scatter(Q_BEP, H_BEP, s=markersize)
    for v, q_max, h_max in zip(V, Q_max, H_max):
        label = 'constant running speed' if v == V[-1] else None
        ax.plot(q_interpolated[v], pump_curves[v], color='black', label=label)
        if show_pump_speed:
            plt.text(q_max, h_max, f'{v} RPM', fontsize=11)
    if show_legend:
        ax.legend()
    ax.set_xlabel('Flow Q [m^3/h]')
    ax.set_ylabel('Head H [mWC]')
    return ax