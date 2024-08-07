# ©, 2024, Sirris
# owner: FFNG

import pandas as pd
import numpy as np

def contextual_train_test_split(df_contextual, split_date):
    df_contextual_train = df_contextual[df_contextual.timestamp < split_date]
    df_contextual_test = df_contextual[df_contextual.timestamp >= split_date]
    return df_contextual_train, df_contextual_test


def vibration_train_test_split(df_V, split_date):
    meta_data = df_V[['timestamp', 'pump', 'location', 'direction']].copy()

    def time_based_train_test_split(_meta_data, split_date='2021-10-01'):
        return (_meta_data['timestamp'] > split_date).replace({False: 'train', True: 'test'})

    meta_data['time_based_set'] = time_based_train_test_split(meta_data, split_date=split_date)
    is_trainset = (meta_data['time_based_set'] == 'train')
    meta_data_train_set = meta_data[is_trainset]
    df_V_train = df_V[is_trainset]
    is_testset = (meta_data['time_based_set'] == 'test')
    meta_data_test_set = meta_data[is_testset]
    df_V_test = df_V[is_testset]
    return df_V_train, df_V_test, meta_data


def order_cluster_names(df, add_pump_to_cluster_name=False, col_name_clusters='cluster'):
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

    alphabet_upper_case = list(map(chr, range(ord('A'), ord('Z')+1)))
    replace_oms = dict(list(zip(sorted(df['cluster_kmeans'].unique()), alphabet_upper_case)))
    df['cluster_kmeans'] = df['cluster_kmeans'].replace(to_replace=replace_oms)

    return df, name_mapping


def assign_name_according_to_principle_component(pump, centroids):
    """ For each pump, assign the cluster names according to the first principal component. """
    arr = centroids[pump][:,0]
    # Get the indices that would sort the array in an descending order
    sorted_indices = np.flip(np.argsort(arr))
    # Create a list of indices in the desired order
    res = [list(sorted_indices).index(i) for i in range(len(arr))]
    res = dict(zip(range(len(res)), np.array(res) + 1))
    # res = {k: v for k, v in zip(range(len(res)), np.array(res) + 1)}
    # res = dict(res, **{np.nan: 0})
    res.update({np.nan: 0})
    return res


# CLUSTER_NAMES_DICT = {p:assign_name_according_to_principle_component(p) for p in PUMPS}
# SK TODO: automatic renaming of 
LETTERS = {
    1: {0:'unknown', 1:'C', 2:'F', 3:'I', 4:'K', 5:'N', 6:'O'},
    2: {0:'unknown', 1:'B', 2:'D', 3:'G', 4:'J', 5:'M', 6:'P'},
    3: {0:'unknown', 1:'A', 2:'E', 3:'H', 4:'L', 5:'Q'}
}


def create_merged_df(df_dist_pivot, process_view, pump, train=False, om_name='OM_v6', centroids=None):
    df_left = df_dist_pivot.copy().reset_index()
    df_left = df_left[df_left.pump == pump]
    df_left['timestamp'] = pd.to_datetime(df_left['timestamp'], utc=True).dt.tz_localize(None)
    df_left = df_left.sort_values(by='timestamp')
    df_right = process_view[pump].copy().reset_index()
    df_right['timestamp'] = pd.to_datetime(df_right['timestamp'], utc=True).dt.tz_localize(None)
    df_merged = pd.merge_asof(df_left, df_right, left_on='timestamp', right_on='timestamp', direction='backward')
    # SK TODO: need to take care of cluster names! 
    
    cluster_names_dict_ = assign_name_according_to_principle_component(pump, centroids)
    letters_ = LETTERS[pump]
    if train:
        df_merged['OM-replaced-name'] = df_merged['cluster_kmeans'] # !!!
    else:
        df_merged['OM-replaced-name'] = df_merged[om_name].replace(cluster_names_dict_)
        # print(df_merged['OM-replaced-name'].unique())
        df_merged['OM-replaced-name'] = df_merged['OM-replaced-name'].replace(letters_)
    df_merged['vibration view: closest fingerprint'] = df_merged['closest OM']
    df_merged['process view: closest operating mode'] = df_merged['OM-replaced-name']
    return df_merged