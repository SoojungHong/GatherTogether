"""
Mahalanobis Metric for Clustering (MMC)
http://contrib.scikit-learn.org/metric-learn/generated/metric_learn.MMC.html#metric_learn.MMC
"""

from metric_learn import MMC

pairs = [[[1.2, 7.5], [1.3, 1.5]],
         [[6.4, 2.6], [6.2, 9.7]],
         [[1.3, 4.5], [3.2, 4.6]],
         [[6.2, 5.5], [5.4, 5.4]]]

# in this task we want points where the first feature is close to be closer to each other,
# no matter how close the second feature is

y = [1, 1, -1, -1]

"""
Learn MMC (Mahalanobis Metrics for Clustering) Model 
"""
mmc = MMC()
mmc.fit(pairs, y) # learn the MMC model

"""
Return the decision function used to classify the pairs
"""
print("debug 1: ", mmc.decision_function(pairs))


"""
Returns a copy of the Mahalanobis matrix learned by the metric learner
"""
print("debug 2: ", mmc.get_mahalanobis_matrix())


"""
Returns a function that takes as input two 1D arrays and outputs the learned metric score on these two points.
"""
f = mmc.get_metric()
print("debug 3: ", f)


"""
Predicts the learned metric between input pairs
"""
example_pairs = [[[1.2, 7.5], [1.3, 8.5]]]#[1.2, 7.5] # error - ValueError: 3D array of formed tuples expected by MMC.
print("debug 4 : ", mmc.predict(example_pairs))

