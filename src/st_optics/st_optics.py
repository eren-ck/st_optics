# -*- coding: utf-8 -*-
"""
ST_OPTICS - fast scalable implementation of ST OPTICS
            scales also to memory by splitting into frames
            and merging the clusters together
"""

# Author: Eren Cakmak <eren.cakmak@uni-konstanz.de>
#
# License: MIT

import numpy as np
import math
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import OPTICS
from sklearn.utils import check_array
from tqdm import tqdm

class ST_OPTICS():
    """
    A class to perform the ST_OPTICS clustering
    Parameters
    ----------
    eps1 : float, default=np.inf
        The spatial density threshold (maximum spatial distance) between 
        two points to be considered related. Only used when method cluster method is DBSCAN
    eps2 : float, default=10
        The temporal threshold (maximum temporal distance) between two 
        points to be considered related.
    min_samples : int, default=5
        The number of samples required for a core point.
    xi : float, between 0 and 1, optional (default=0.05)
    cluster_method : str, optional (default='xi')
    metric : string default='euclidean'
        The used distance metric - more options are
        ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
        ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘rogerstanimoto’, ‘sqeuclidean’,
        ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘yule’.
    n_jobs : int or None, default=-1
        The number of processes to start -1 means use all processors 
    Attributes
    ----------
    labels : array, shape = [n_samples]
        Cluster labels for the data - noise is defined as -1
    References
    ----------
    Ankerst, M., Breunig, M. M., Kriegel, H. P., & Sander, J. (1999). OPTICS: ordering points to identify the clustering structure. ACM Sigmod record, 28(2), 49-60.
    """
    def __init__(self,
                 eps1=np.inf,
                 eps2=10,
                 min_samples=5,
                 max_eps=np.inf,
                 metric='euclidean',
                 cluster_method='xi',
                 xi=0.05,
                 n_jobs=-1):
        self.eps1 = eps1
        self.eps2 = eps2
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.cluster_method = cluster_method
        self.xi = xi
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X):
        """
        Apply the ST OPTICS algorithm 
        ----------
        X : 2D numpy array with
            The first element of the array should be the time 
            attribute as float. The following positions in the array are 
            treated as spatial coordinates. The structure should look like this [[time_step1, x, y], [time_step2, x, y]..]
            For example 2D dataset:
            array([[0,0.45,0.43],
            [0,0.54,0.34],...])
        Returns
        -------
        self
        """
        # check if input is correct
        X = check_array(X)

        if not self.eps1 > 0.0 or not self.eps2 > 0.0 or not self.min_samples > 0.0:
            raise ValueError('eps1, eps2, minPts must be positive')

        n, m = X.shape

        # Compute sqaured form Euclidean Distance Matrix for 'time' attribute and the spatial attributes
        time_dist = pdist(X[:, 0].reshape(n, 1),
                          metric=self.metric)
        euc_dist = pdist(X[:, 1:], metric=self.metric)

        # filter the euc_dist matrix using the time_dist
        time_filter = math.pow(10, m)
        dist = np.where(time_dist <= self.eps2, euc_dist, time_filter)

        # speeds up the ST OPTICS
        if np.isinf(self.max_eps):
            self.max_eps = time_filter - 1
        if np.isinf(self.eps1):
            self.eps1 = time_filter - 1

        op = OPTICS(eps=self.eps1,
                    min_samples=self.min_samples,
                    metric='precomputed',
                    max_eps=self.max_eps,
                    cluster_method=self.cluster_method,
                    xi=self.xi,
                    n_jobs=self.n_jobs)
        op.fit(squareform(dist))

        self.labels = op.labels_
        self.reachability = op.reachability_
        self.ordering = op.ordering_
        self.core_distances = op.core_distances_
        self.predecessor = op.predecessor_
        self.cluster_hierarchy = op.cluster_hierarchy_

        return self

    def fit_frame_split(self, X, frame_size, frame_overlap=None):
        """
        Apply the ST OPTICS algorithm with splitting it into frames 
        Merging is still not optimal resulting in minor errors in 
        the overlapping area. In this case the input data has to be 
        sorted for by time. 
        ----------
        X : 2D numpy array with
            The first element of the array should be the time (sorted by time)
            attribute as float. The following positions in the array are 
            treated as spatial coordinates. The structure should look like this [[time_step1, x, y], [time_step2, x, y]..]
            For example 2D dataset:
            array([[0,0.45,0.43],
            [0,0.54,0.34],...])
        frame_size : float, default= None
            If not none the dataset is split into frames and merged aferwards
        frame_overlap : float, default=eps2
            If frame_size is set - there will be an overlap between the frames
            to merge the clusters afterwards 
        Returns
        -------
        self
        """
        # check if input is correct
        X = check_array(X)

        # default values for overlap
        if frame_overlap == None:
            frame_overlap = self.eps2

        if not self.eps1 > 0.0 or not self.eps2 > 0.0 or not self.min_samples > 0.0:
            raise ValueError('eps1, eps2, minPts must be positive')

        if not frame_size > 0.0 or not frame_overlap > 0.0 or frame_size < frame_overlap:
            raise ValueError(
                'frame_size, frame_overlap not correctly configured.')

        # unique time points
        time = np.unique(X[:, 0])

        labels = None
        right_overlap = 0
        max_label = 0

        for i in tqdm(range(0, len(time), (frame_size - frame_overlap + 1))):
            for period in [time[i:i + frame_size]]:
                frame = X[np.isin(X[:, 0], period)]

                self.fit(frame)

                # match the labels in the overlaped zone
                # objects in the second frame are relabeled
                # to match the cluster id from the first frame
                if not type(labels) is np.ndarray:
                    labels = self.labels
                else:
                    frame_one_overlap_labels = labels[len(labels) -
                                                      right_overlap:]
                    frame_two_overlap_labels = self.labels[0:right_overlap]

                    mapper = {}
                    for i in list(
                            zip(frame_one_overlap_labels,
                                frame_two_overlap_labels)):
                        mapper[i[1]] = i[0]

                    # clusters without overlapping points are ignored
                    ignore_clusters = set(self.labels) - set(
                        frame_two_overlap_labels)
                    # recode them to the value -99
                    new_labels_unmatched = [
                        i if i not in ignore_clusters else -99
                        for i in self.labels
                    ]

                    # objects in the second frame are relabeled to match the cluster id from the first frame
                    new_labels = np.array([
                        mapper[i] if i != -99 else i
                        for i in new_labels_unmatched
                    ])

                    # delete the right overlap
                    labels = labels[0:len(labels) - right_overlap]
                    # change the labels of the new clustering and concat
                    labels = np.concatenate((labels, new_labels))

                right_overlap = len(X[np.isin(X[:, 0],
                                              period[-frame_overlap + 1:])])

        # rename labels with -99
        labels[labels == -99] = -1

        self.labels = labels

        return self
