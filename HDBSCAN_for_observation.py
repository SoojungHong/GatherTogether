
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from sklearn.cluster import DBSCAN

from rwo_parser.Ingolstadt_data.GFRHadSignGroup import *


"""
clean dataframe to remove NaN, infinity or a value too large for dtype('float64')
"""
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def check_cluster_point(cluster):
    """
    #print('len(cluster) : ', len(cluster))
    # for i in range(len(cluster)): # this doesn't work since cluster dataframe has its own index
    sign_arr = []
    for row in cluster.iterrows():
        #print('row :', row[1]) # OK
        lon = row[1]["position::longitude_degrees"]
        lat = row[1]["position::latitude_degrees"]
        #["position::latitude_degrees"])#, row["position::longitude_degrees"]) #, row["HadSign"])
        #lat = cluster["position::latitude_degrees"][i]
        #lon = cluster["position::longitude_degrees"][i]
        point = simple_sign_df[(simple_sign_df["position::latitude_degrees"] == lat) & (simple_sign_df["position::longitude_degrees"] == lon)]
        #print("Sign : ", point["HadSign"])
        sign_arr.append( (point["HadSign"]).values[0])
    big_np_arr = sign_arr # ToDo : why this step is necessary?? #np.array(sign_arr)
    #print('test : ', big_np_arr)
    most_freq_sign_in_cluster = get_most_frequent_sign(big_np_arr)
    #print('cluster sign : ', most_freq_sign_in_cluster)
    """

    #for point in cluster:
    mean_point = (np.mean(cluster["position::latitude_degrees"]), np.mean(cluster["position::longitude_degrees"])) #, np.mean(cluster["position::altitude_meters"]))
  
    return tuple(mean_point)


def check_cluster_sign(cluster):
    sign_arr = []
    for row in cluster.iterrows():
        lon = row[1]["position::longitude_degrees"]
        lat = row[1]["position::latitude_degrees"]
        point = row[1]["HadSign"]
        sign_arr.append(point)
    big_np_arr = sign_arr
    most_freq_sign_in_cluster = get_most_frequent_sign(big_np_arr)
    print('cluster sign : ', most_freq_sign_in_cluster)

    return most_freq_sign_in_cluster


def hdbscan_clustering_using_lat_lon_alt_width_height():
    import hdbscan
    labels = ["position::longitude_degrees", "position::latitude_degrees", "position::altitude_meters", "details::classification::gfr_group", "HadSign", "details::width_meters", "details::height_meters"]
    clustering_labels = ["position::longitude_degrees", "position::latitude_degrees", "position::altitude_meters", "details::width_meters", "details::height_meters"]

    df = sign_df[labels]
    df["HadSign"] = pd.Categorical(df["HadSign"])  # change the type of the column
    df["sign_code"] = df["HadSign"].cat.codes  # To capture category code

    df_clean = clean_dataset(df[clustering_labels])

    # normalize
    column_maxes = df_clean.max()
    df_max = column_maxes.max()
    df_normalized = df_clean / df_max

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, gen_min_span_tree=True)
    db = clusterer.fit(df_normalized)

    df_normalized["HadSign"] = df["HadSign"]
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    df_clean["HadSign"] = df["HadSign"]
    clusters = pd.Series([df_clean[cluster_labels == n] for n in range(num_clusters)])
    clusters = clusters[:-1]  # FIX: remove empty last dataframe
    print("number of clusters : ", num_clusters)

    """
        centroid of all clusters 
    """
    centermost_points = clusters.map(check_cluster_point)
    all_signs = clusters.map(check_cluster_sign)
    lats, lons = zip(*centermost_points)
    rep_points = pd.DataFrame({"position::longitude_degrees": lons, "position::latitude_degrees": lats})
    rep_points["group"] = all_signs
    rep_points.to_csv("result_oct_29_3.csv")

    # visualize
    fig, ax = plt.subplots(figsize=[20, 20])
    color_labels = rep_points["group"].unique() # Get Unique continents

    # List of colors in the color palettes
    rgb_values = sns.color_palette("Set1", n_colors=len(color_labels))

    # Map continents to the colors
    color_map = dict(zip(color_labels, rgb_values))

    # plot
    groups = rep_points.groupby('group')
    for name, group in groups:
        ax.plot(group["position::longitude_degrees"], group["position::latitude_degrees"], marker='o', linestyle='',
                ms=2, label=name, color=color_map[name])
    ax.legend(fontsize=5, loc='upper right')

    ax.set_title("Sign observation clusters in Ingolstadt", fontsize=15)
    ax.set_xlabel('Longitude', fontsize=15)
    ax.set_ylabel('Latitude', fontsize=15)

    plt.show()

# experiment
hdbscan_clustering_using_lat_lon_alt_width_height()
