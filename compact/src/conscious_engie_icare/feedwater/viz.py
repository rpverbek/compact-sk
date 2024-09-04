# ©, 2024, Sirris
# owner: FFNG
from conscious_engie_icare.viz.viz import get_controller
import seaborn as sns
import matplotlib.pyplot as plt
from ipywidgets import interact


def show_fingerprints(fingerprints_):
    """ Code adapted from conscious_engie_icare/viz/viz module for showing fingerprints in interactive way. """
    possible_oms = list(fingerprints_.keys())
    controller_om = get_controller({'widget': 'Dropdown',
                                    'options': [(_om, _om) for _om in possible_oms],
                                    'value': possible_oms[0],
                                    'description': 'Select the operating mode'})
    
    def make_plot(om):
        df_ = fingerprints_[om]
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.heatmap(df_, annot=True, fmt=".3f", ax=ax, cmap='Blues', vmin=0, vmax=0.01, cbar=False)
        ax.set_title(f'Vibration fingerprint @ OM {om}')
        ax.set_xlabel('component')
        fig.show()
    
    interact(make_plot, om=controller_om)