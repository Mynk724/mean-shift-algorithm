
from sklearn.cluster import MeanShift
from sklearn.datasets import load_iris

def mean_shift_clustering():
    # Load built-in Iris dataset
    data = load_iris()
    X = data.data

    # Apply Mean Shift
    ms = MeanShift()
    labels = ms.fit_predict(X)

    print("Cluster Centers:")
    print(ms.cluster_centers_)
    print("Number of Clusters:", len(set(labels)))

if __name__ == "__main__":
    mean_shift_clustering()
