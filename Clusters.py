import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

dataset = 'comtrade_2015_2dg.tsv'
path = 'data'

with open('%s/%s' % (path, dataset), mode='rU') as f:
    df = pd.read_csv(f, sep='\t', header=0)

countries = set(df['toTitle'].tolist())

df_pivoted = pd.DataFrame(index = countries)



for ctry in countries:
    df_subset = df[df['toTitle'] == ctry]
    pivoted = df_subset.pivot(index='fromTitle', columns='cmdCode', values='tradeValue')
    df_pivoted.merge(pivoted, how = 'outer', left_index=True, right_index=True)

print pivoted.head()

#km = KMeans(n_clusters=8, random_state=1)
#km.fit(X)