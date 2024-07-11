import matplotlib.pyplot as plt


def plot_cumulative_timeseries(df, vehicle_ids):
    unique_features = set(int(col.split('_')[0]) for col in ATTRIBUTE_COLUMNS)
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