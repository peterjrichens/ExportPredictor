import pandas as pd


data_path = 'data'
with open('%s/comtrade_2000_2015_2dg.tsv' % data_path, mode='rU') as f:
    df = pd.read_csv(f, sep='\t', header=0)


print df.columns
print 'number of rows = %d' % len(df)
print df.source.value_counts()
print df.yr.value_counts()

