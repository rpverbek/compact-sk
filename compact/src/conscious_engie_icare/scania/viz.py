import matplotlib.pyplot as plt
from conscious_engie_icare.scania.dataset import get_attribute_columns
import matplotlib as mpl


def plot_cumulative_timeseries(df, vehicle_ids, attribute_columns=None):
    if attribute_columns is None:
        attribute_columns = get_attribute_columns(df)
    
    unique_features = set(int(col.split('_')[0]) for col in attribute_columns)
    fig, axes = plt.subplots(figsize=(6*len(unique_features), 4*len(vehicle_ids)),
                             nrows=len(vehicle_ids), ncols=len(unique_features), sharex=True)
    if len(vehicle_ids) > 1:
        for vehicle_id, axes_row in zip(vehicle_ids, axes):
            df_ = df[df.vehicle_id == vehicle_id].set_index('time_step')
            for feature, ax in zip(unique_features, axes_row):
                df_plot_ = df_[df_.columns[df_.columns.str.contains(str(feature))]]
                #df_plot_.plot(ax=ax, marker='o')
                ax.plot(df_plot_, marker='o')
                ax.set_xlabel('timestamp')
                ax.set_ylabel('Cumulative count')
                ax.set_title(feature)
    else:
        df_ = df[df.vehicle_id == vehicle_ids[0]].set_index('time_step')
        for feature, ax in zip(unique_features, axes):
            df_plot_ = df_[df_.columns[df_.columns.str.contains(str(feature))]]
            ax.plot(df_plot_, marker='o')
            ax.set_xlabel('abstract timestamp unit')
            ax.set_ylabel('cumulative count')
            ax.set_title(feature)
    fig.tight_layout()
    return fig, axes


def plot_V(V):
    cmap = mpl.cm.get_cmap("Blues")
    fig, ax = plt.subplots(figsize=(6, 8))
    nrows = V.shape[0]
    ncols = V.shape[1]
    title_ = "Performance matrix V" + f" ({nrows} x {ncols})"
    ax.set_title(title_, fontsize=20)
    # V.columns = BAND_COLUMNS
    im = ax.imshow(
        V,
        cmap=cmap,
        aspect='auto',
        interpolation='nearest',
        norm=mpl.colors.LogNorm(vmin=None, vmax=None),
        # extent=[0.25,30.25,nrows,0]
    )
    ax.tick_params(axis='y', labelrotation=90)
    return fig, ax