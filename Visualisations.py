import pandas as pd
from SliceData import *
from sys import exit
import matplotlib.pyplot as plt
from DownloadData import get_wb

with open('UN Comtrade Country List.csv', mode='rU') as f:
    ctry_codes = pd.read_csv(f, sep=',')

ctry_codes = ctry_codes[['Country Code','Country Name English','ISO3-digit Alpha']]
gdp = get_wb('NY.GDP.PCAP.PP.KD',2010,2011)
gdp = gdp[gdp.year=='2010']


rca = rca('comtrade_2010_2dg.tsv',yr=2010)

average_RCA = rca[['hasRCA','fromCode']].groupby('fromCode',as_index=False).mean().head(200)
average_RCA = average_RCA.sort('hasRCA')
average_RCA = average_RCA.merge(ctry_codes, left_index=True,right_on='Country Code')


average_RCA = average_RCA.merge(gdp, left_on='ISO3-digit Alpha',right_on='iso')

average_RCA.plot.scatter(x='hasRCA', y='value')
plt.show()

#within countries, over time?
