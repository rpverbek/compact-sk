# ©, 2022, Sirris
# owner: FFNG

""" Visualisations of single frequency bands.

For plotting multiple frequency bands over time, use the functions provided in
<viz/spectrogram.py>.
"""
import matplotlib.pyplot as plt


def plot_frequency_band(x, y, peaks=None, ax=None, **kwargs):
    """Show frequency band for frequencies x and corresponding amplitudes y."""
    # create figure and axis if no axis was provided
    new_fig = True if ax is None else False
    fig = None
    if new_fig:
        fig, ax = plt.subplots(figsize=(12, 9))

    # plot frequency band
    ax.plot(x, y, label='signal', **kwargs)
    ax.set_xlabel('freqency [Hz]')
    ax.set_ylabel('acceleration')

    # plot peaks if provided
    if peaks is not None:
        ax.scatter(x[peaks], y[peaks], marker='x', color='black',
                   label=f'peak (n={len(peaks)})')

    ax.legend()
    return ax if new_fig else fig, ax

