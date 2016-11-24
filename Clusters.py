import pandas as pd
import numpy as np
from scipy.stats import variation
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan
from sys import exit
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import TradeExplorer
plt.rcParams['font.size'] = 14
from SliceData import rca
from SliceData import cmd_lookup

space = TradeExplorer.CountrySpace(2011,'comtrade_2011_2dg.tsv',feature='export_destination')
X = space.distance()
X = StandardScaler().fit_transform(X)

def get_cluster(k, X):

    km = KMeans(n_clusters=k, random_state=1)
    km.fit(X)
    cluster_labels_km = pd.Series(km.labels_)
    cluster_counts_km = cluster_labels_km.value_counts()

    agglom = AgglomerativeClustering(n_clusters=k)
    agglom.fit(X)
    cluster_labels_agglom = pd.Series(agglom.labels_)
    cluster_counts_agglom = cluster_labels_agglom.value_counts()

    cluster_counts = pd.concat([cluster_counts_km,cluster_counts_agglom],axis=1)
    var = min(variation(cluster_labels_agglom),variation(cluster_counts_agglom))
    return var, cluster_counts


def hdbscan_counts(X):
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                gen_min_span_tree=False, leaf_size=40,
                                metric='precomputed', min_cluster_size=2, min_samples=None, p=None)
    clusterer.fit(X)
    cluster_labels = pd.Series(clusterer.labels_)
    return cluster_labels.value_counts(), cluster_labels

hdbscan_counts, labels = hdbscan_counts(X)

print pd.concat([space.ctry_names,labels],axis=1)
print hdbscan_counts
exit()
rca = rca('comtrade_2010_2dg.tsv',yr=2010)
drop = [1,3]

rca = rca[rca.cmdCode != 1]
rca = rca[rca.cmdCode != 3]
print rca.head()
exit()
rca = rca.pivot(index = 'cmdCode', columns='fromCode', values='rca')



def components(n, rca):
    pca = PCA(n_components=n)
    components = pca.fit_transform(rca)
    components = pd.DataFrame(components)
    cmd_code = pd.Series(rca.index.values,name='cmdCode')
    components = pd.concat([cmd_code,components],axis=1)
    components = components.set_index('cmdCode')
    return components

rca = components(2, rca)



color_palette = sns.color_palette('deep', 8)
cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in clusterer.labels_]
cluster_member_colors = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, clusterer.probabilities_)]
plt.scatter(rca[0],rca[1],s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
plt.show()
exit()

def components(n, rca):
    pca = PCA(n_components=n)
    components = pca.fit_transform(rca)
    components = pd.DataFrame(components)
    cmd_code = pd.Series(rca.index.values,name='cmdCode')
    components = pd.concat([cmd_code,components],axis=1)
    components = components.set_index('cmdCode')
    return components

def plot_components(n,rca):
    pca = PCA(n_components=n)
    pca.fit_transform(rca)
    plt.plot(pca.explained_variance_ratio_)
    plt.show()

#plot_components(4,rca)


def get_cluster(k, n, rca):
    X = components(n, rca)

    km = KMeans(n_clusters=k, random_state=1)
    km.fit(X)
    cluster_labels_km = pd.Series(km.labels_)
    cluster_counts_km = cluster_labels_km.value_counts()

    agglom = AgglomerativeClustering(n_clusters=k)
    agglom.fit(X)
    cluster_labels_agglom = pd.Series(agglom.labels_)
    cluster_counts_agglom = cluster_labels_agglom.value_counts()

    cluster_counts = pd.concat([cluster_counts_km,cluster_counts_agglom],axis=1)
    var = min(variation(cluster_labels_agglom),variation(cluster_counts_agglom))
    return var, cluster_counts

lowest_var = np.inf
best_counts = pd.DataFrame()
best_n = 0
best_k = 0
for n in range(4,5):
    for k in range(10,80):
        var, cluster_counts = get_cluster(k, n, rca)
        if var < lowest_var:
            lowest_var = var
            best_counts = cluster_counts
            best_n = n
            best_k = k

print 'best number of clusters: ',best_k
print 'best number of components: ',best_n
print best_counts


def score(k,al):
    if al=='km':
        km = KMeans(n_clusters=k, random_state=1)
        km.fit(X)
        return metrics.silhouette_score(X, km.labels_)
    else:
        assert al=='agglom'
        agglom = AgglomerativeClustering(n_clusters=k)
        agglom.fit(X)
        return metrics.silhouette_score(X, agglom.labels_)

def plot_score(al):
    k_range = range(2,30)
    scores = []
    for k in k_range:
        scores.append(score(k,al))
    plt.plot(k_range, scores, label = al)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Coefficient')
    plt.grid(True)

#plot_score('km')
#plot_score('agglom')
#plt.show()


