
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


def split_to_sub_clusters_get_centroids_lat(cluster):
    sign_arr = []
    sign_dictionary = {}
    sign_lat_dic = {}
    sign_lon_dic = {}
    for row in cluster.iterrows():
        lon = row[1]["position::longitude_degrees"]
        lat = row[1]["position::latitude_degrees"]
        sign_name = row[1]["HadSign"]
        sign_arr.append(sign_name)

        if sign_name in sign_dictionary:
            sign_dictionary[sign_name] = sign_dictionary[sign_name] + 1
            sign_lat_dic[sign_name] = float((sign_lat_dic[sign_name] + lat)/2)
            sign_lon_dic[sign_name] = float((sign_lon_dic[sign_name] + lat)/2)
        else:
            sign_dictionary[sign_name] = 1
            sign_lat_dic[sign_name] = lat
            sign_lon_dic[sign_name] = lon

    return list(sign_lat_dic.values())


def split_to_sub_clusters_get_centroids_lon(cluster):
    sign_arr = []
    sign_dictionary = {}
    sign_lat_dic = {}
    sign_lon_dic = {}
    for row in cluster.iterrows():
        lon = row[1]["position::longitude_degrees"]
        lat = row[1]["position::latitude_degrees"]
        sign_name = row[1]["HadSign"]
        sign_arr.append(sign_name)

        if sign_name in sign_dictionary:
            sign_dictionary[sign_name] = sign_dictionary[sign_name] + 1
            sign_lat_dic[sign_name] = float((sign_lat_dic[sign_name] + lat)/2)
            sign_lon_dic[sign_name] = float((sign_lon_dic[sign_name] + lon)/2)
        else:
            sign_dictionary[sign_name] = 1
            sign_lat_dic[sign_name] = lat
            sign_lon_dic[sign_name] = lon

    return list(sign_lon_dic.values())


def split_to_sub_clusters_get_centroids_alt(cluster):
    sign_arr = []
    sign_dictionary = {}
    sign_lat_dic = {}
    sign_lon_dic = {}
    sign_alt_dic = {}
    for row in cluster.iterrows():
        lon = row[1]["position::longitude_degrees"]
        lat = row[1]["position::latitude_degrees"]
        alt = row[1]["position::altitude_meters"]
        sign_name = row[1]["HadSign"]
        sign_arr.append(sign_name)

        if sign_name in sign_dictionary:
            sign_dictionary[sign_name] = sign_dictionary[sign_name] + 1
            sign_lat_dic[sign_name] = float((sign_lat_dic[sign_name] + lat) / 2)
            sign_lon_dic[sign_name] = float((sign_lon_dic[sign_name] + lon) / 2)
            sign_alt_dic[sign_name] = float((sign_alt_dic[sign_name] + alt) / 2)

        else:
            sign_dictionary[sign_name] = 1
            sign_lat_dic[sign_name] = lat
            sign_lon_dic[sign_name] = lon
            sign_alt_dic[sign_name] = alt

    return list(sign_alt_dic.values())


def split_to_sub_clusters_get_centroids_sign(cluster):
    sign_arr = []
    sign_dictionary = {}
    sign_lat_dic = {}
    sign_lon_dic = {}

    for row in cluster.iterrows():
        lon = row[1]["position::longitude_degrees"]
        lat = row[1]["position::latitude_degrees"]
        sign_name = row[1]["HadSign"]
        sign_arr.append(sign_name)

        if sign_name in sign_dictionary:
            sign_dictionary[sign_name] = sign_dictionary[sign_name] + 1
            sign_lat_dic[sign_name] = float((sign_lat_dic[sign_name] + lat) / 2)
            sign_lon_dic[sign_name] = float((sign_lon_dic[sign_name] + lat) / 2)
        else:
            sign_dictionary[sign_name] = 1
            sign_lat_dic[sign_name] = lat
            sign_lon_dic[sign_name] = lon

    return list(sign_dictionary.keys())

def dbscan_clustering_using_geo_then_gfr():
    labels = ["position::longitude_degrees", "position::latitude_degrees"]

    df = sign_df[labels]
    df_clean = clean_dataset(df)

    road_threshold = 0.003  # (in km)
    kms_per_rad = 6371.0088
    eps = road_threshold / kms_per_rad
    db = DBSCAN(eps=eps, min_samples=3, algorithm='ball_tree', metric='haversine').fit(np.radians(df_clean))

    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series([df[cluster_labels == n] for n in range(num_clusters)])
    clusters = clusters[:-1]  # FIX: remove empty last dataframe
    
    """
        centroid of all clusters 
    """

    centermost_lat_points = clusters.map(split_to_sub_clusters_get_centroids_lat)
    all_lats = []
    for index, value in centermost_lat_points.items():
        all_lats.extend(value) # use extend() instead of append() to keep one list

    centermost_lon_points = clusters.map(split_to_sub_clusters_get_centroids_lon)
    all_lons = []
    for index2, value2 in centermost_lon_points.items():
        all_lons.extend(value2)

    centermost_signs = clusters.map(split_to_sub_clusters_get_centroids_sign)
    all_signs = []
    for index3, value3 in centermost_signs.items():
        all_signs.extend(value3)

    rep_points = pd.DataFrame({"position::longitude_degrees": all_lons, "position::latitude_degrees": all_lats, "group":all_signs})
    rep_points.to_csv("dbscan_cluster_with_subclustering.csv")

    fig, ax = plt.subplots(figsize=[20, 20])  # [10, 6])
    color_labels = rep_points["group"].unique()     # Get Unique continents
    rgb_values = sns.color_palette("Set1", n_colors=len(color_labels))
    color_map = dict(zip(color_labels, rgb_values)) # Map continents to the colors

    groups = rep_points.groupby('group')
    for name, group in groups:
        ax.plot(group["position::longitude_degrees"], group["position::latitude_degrees"], marker='o', linestyle='',
                ms=4, label=name, color=color_map[name])
    ax.legend(fontsize=6, loc='upper right')

    ax.set_title("Sign observation clusters in Ingolstadt", fontsize=15)
    ax.set_xlabel('Longitude', fontsize=15)
    ax.set_ylabel('Latitude', fontsize=15)

    plt.show()

dbscan_clustering_using_geo_then_gfr()
