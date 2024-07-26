# ©, 2024, Sirris
# owner: FFNG

def order_cluster_names(df, add_pump_to_cluster_name=True, col_name_clusters='cluster'):
    # rename cluster s.t. each name is unique across pumps,
    # e.g. there should be no OM 1 in pump 11 and pump 12 at the same time
    offset = {}
    tmp = 0
    for pump, group in df.groupby('pump'):
        offset[pump] = tmp
        n_unique_clusters = len(group[col_name_clusters].unique())
        tmp += n_unique_clusters
    series_offset = df['pump'].replace(offset)
    df[col_name_clusters] = df[col_name_clusters] + series_offset

    # extract mean RPM per cluster and keep track of corresponding pump
    df_grouped = df.groupby(col_name_clusters)
    mean_rpm = df_grouped.mean()['velocity <v-MP> [RPM]']
    corresponding_pump = df_grouped.first()['pump']

    # rename cluster according to asceding mean RPM,
    # i.e. lowest RPM --> Cluster0, second lowest RPM --> Cluster1
    sorted_c = mean_rpm.sort_values(ascending=True)
    om_to_pump = dict(list(zip(sorted_c.index, corresponding_pump.loc[sorted_c.index])))
    name_mapping = dict(list(zip(sorted_c.index, range(len(sorted_c)))))
    df[col_name_clusters] = df[col_name_clusters].replace(name_mapping)

    """
    # rename cluster from numeric to alphabetic
    alphabet_upper_case = list(map(chr, range(ord('A'), ord('Z')+1)))
    replace_oms = dict(list(zip(sorted(df[col_name_clusters].unique()), alphabet_upper_case)))
    om_to_pump = {replace_oms[k]: v for k, v in om_to_pump.items()}
    print(om_to_pump)
    # global_name_mapping = {k: v for k, v in om_to_pump.items()}
    df[col_name_clusters] = df[col_name_clusters].replace(to_replace=replace_oms)
    """

    # create unique name that includes pump type
    if add_pump_to_cluster_name:
        for pump, group in df.groupby('pump'):
            sorted_col_names = sorted(group[col_name_clusters].unique())
            for i, col_name in enumerate(sorted_col_names):
                name_mapping[col_name] = f'p{pump}_c{i}'
        df[col_name_clusters] = df[col_name_clusters].replace(name_mapping)

    return df, False