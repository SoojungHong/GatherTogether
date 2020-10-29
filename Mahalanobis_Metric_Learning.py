import pandas as pd
import math
import random
from math import sqrt


"""
data :
1. road sign gt data
2. observation data
"""

sign_path = "C:/Users/shong/PycharmProjects/sandbox_p2g-sensor-aggregation-and-conflation/rwo_parser/Ingolstadt_data/signs.csv" #sign_aug3031_merged.csv"
sign_df = pd.read_csv(sign_path)    # 11680 * 17

road_sign_gt_path = "C:/Users/shong/PycharmProjects/sandbox_p2g-sensor-aggregation-and-conflation/rwo_parser/Ingolstadt_data/road_sign_gt.csv"
sign_gt_df = pd.read_csv(road_sign_gt_path) # 325 * 10

obs_labels = ["position::longitude_degrees","position::latitude_degrees","position::altitude_meters","details::height_meters","details::width_meters"]
gt_labels = ["lon_0","lat_0","properties_alt_m","properties_height_m","properties_width_m"]

sign_df = sign_df[obs_labels]
sign_gt_df = sign_gt_df[gt_labels]

"""
1. first measure the haversine distance with lat and lon
2. if the distance less than 3m, compare the altitude
3. ToDo : how to apply height and width 
"""


"""
Sort dataframe with lon and lat and alt
"""
sorted_sign_df = sign_df.sort_values(by=["position::longitude_degrees","position::latitude_degrees","position::altitude_meters"])
sorted_gt_df = sign_gt_df.sort_values(by=["lon_0","lat_0","properties_alt_m"])


"""
Haversine distance of two geo-coordinates
"""

# haversine formula
def haversine(lat1, lon1, lat2, lon2):
    # distance between latitudes
    # and longitudes
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0

    # convert to radians
    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0

    # apply formulae
    a = (pow(math.sin(dLat / 2), 2) +
         pow(math.sin(dLon / 2), 2) *
         math.cos(lat1) * math.cos(lat2));
    rad = 3958#6371  #the units are kilometers - 6,371,000, the units are meters - 3,958
    c = 2 * math.asin(math.sqrt(a))
    return rad * c


# calculate euclidean distance
def euclidean_distance(a, b):
    return sqrt((a - b) ** 2)


def normalize(df):
    column_maxes = df.max()
    df_max = column_maxes.max()
    df_normalized = df / df_max
    print('normalized dataframe : ', df_normalized)
    return df_normalized


def runMMC():

    # Run MMC
    from metric_learn import MMC

    """
    Learn MMC (Mahalanobis Metrics for Clustering) Model 
    """
    mmc = MMC()
    mmc.fit(pairs, y)  # learn the MMC model
    print("Mahalanobis Matrix : ", mmc.get_mahalanobis_matrix())


def create_dataframe():
    training_pairs_positive = []
    training_pairs_negative = []
    y = []
    pairs = []
    count = 0
    p_df = pd.DataFrame(columns=["lon_0", "lat_0", "properties_alt_m", "position::longitude_degrees",
                                 "position::latitude_degrees", "position::altitude_meters"])

    n_df = pd.DataFrame(columns=["lon_0", "lat_0", "properties_alt_m", "position::longitude_degrees",
                                 "position::latitude_degrees", "position::altitude_meters"])

    for row in sorted_gt_df.iterrows():  # row - tuple
        lon = row[1]["lon_0"]
        lat = row[1]["lat_0"]
        alt = row[1]["properties_alt_m"]
        gt_tuple = [lon, lat, alt]

        for obs_row in sorted_sign_df.iterrows():
            obs_lon = obs_row[1]["position::longitude_degrees"]
            obs_lat = obs_row[1]["position::latitude_degrees"]
            obs_alt = obs_row[1]["position::altitude_meters"]
            obs_tuple = [obs_lon, obs_lat, obs_alt]

            dist = haversine(lat, lon, obs_alt, obs_lon)  # if rad is 3958, the distance in meter, if rad is 6371, the distance is in km
            alt_dist = euclidean_distance(alt, obs_alt)  # unit is meter
            # print('diff alt in meter : ', alt_dist)
            if dist < 3:
                if alt_dist > 3:
                    # synthesize the training data
                    pair = [gt_tuple, obs_tuple]
                    row1 = {"lon_0": lon, "lat_0": lat, "properties_alt_m": alt, "position::longitude_degrees": obs_lon,
                            "position::latitude_degrees": obs_lat, "position::altitude_meters": obs_alt}
                    training_pairs_negative.append(pair)
                    n_df = n_df.append(row1, ignore_index=True)
                    pairs.append(pair)
                    y.append(-1)

                    random_diff = float(random.uniform(-2.0, 2.0))
                    modified_obs_tuple = (obs_lon, obs_lat, alt + random_diff)
                    modified_pair = [gt_tuple, modified_obs_tuple]
                    m_row1 = {"lon_0": lon, "lat_0": lat, "properties_alt_m": alt,
                              "position::longitude_degrees": obs_lon,
                              "position::latitude_degrees": obs_lat, "position::altitude_meters": alt + random_diff}

                    training_pairs_positive.append(modified_pair)
                    pairs.append(modified_pair)
                    p_df = p_df.append(m_row1, ignore_index=True)
                    y.append(1)

            elif dist > 3 and count < 2000:
                row3 = {"lon_0": lon, "lat_0": lat, "properties_alt_m": alt, "position::longitude_degrees": obs_lon,
                        "position::latitude_degrees": obs_lat, "position::altitude_meters": obs_alt}
                n_df = n_df.append(row3, ignore_index=True)
                y.append(-1)
                count = count + 1
            elif count > 2000:
                break

    print('Debug : ')  # create new dataframe for positive pairs
    print('p_df : ', p_df)
    print('n_df : ', n_df)
    p_df.to_csv('positive_pairs.csv')
    n_df.to_csv('negative_pairs.csv')
    print(y)


if __name__ == "__main__":

    create_dataframe()

    positive_pairs = pd.read_csv('positive_pairs.csv')
    negative_pairs = pd.read_csv('negative_pairs.csv')

    labels = ["lon_0", "lat_0", "properties_alt_m", "position::longitude_degrees","position::latitude_degrees", "position::altitude_meters"]

    positive_pairs = positive_pairs[labels] # remove id column
    negative_pairs = negative_pairs[labels]

    normalized_p_pairs = normalize(positive_pairs)
    normalized_n_pairs = normalize(negative_pairs)

    pairs = []
    y = []

    # construct with pairs
    for row in normalized_p_pairs.iterrows():
        print('row : ', row)
        lon = row[1]["lon_0"]
        lat = row[1]["lat_0"]
        alt = row[1]["properties_alt_m"]
        gt_tuple = [lon, lat, alt]

        obs_lon = row[1]["position::longitude_degrees"]
        obs_lat = row[1]["position::latitude_degrees"]
        obs_alt = row[1]["position::altitude_meters"]
        obs_tuple = [obs_lon, obs_lat, obs_alt]

        pair = [gt_tuple, obs_tuple]
        pairs.append(pair)
        y.append(1)


    for row2 in normalized_n_pairs.iterrows():
        print('row2 : ', row2)
        lon2 = row2[1]["lon_0"]
        lat2 = row2[1]["lat_0"]
        alt2 = row2[1]["properties_alt_m"]
        gt_tuple2 = [lon2, lat2, alt2]

        obs_lon2 = row2[1]["position::longitude_degrees"]
        obs_lat2 = row2[1]["position::latitude_degrees"]
        obs_alt2 = row2[1]["position::altitude_meters"]
        obs_tuple2 = [obs_lon2, obs_lat2, obs_alt2]

        pair2 = [gt_tuple2, obs_tuple2]
        pairs.append(pair2)
        y.append(-1)


    print('debug : pairs >> ', pairs)
    print('debug : y >> ', y)

    # Run MMC
    from metric_learn import MMC

    """
    Learn MMC (Mahalanobis Metrics for Clustering) Model 
    """
    mmc = MMC()
    mmc.fit(pairs, y)  # learn the MMC model
    print("Mahalanobis Matrix : ", mmc.get_mahalanobis_matrix())
