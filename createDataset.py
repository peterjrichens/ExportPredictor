import os
import pandas as pd
import sys

data_path = 'data'

imports_2dg = []
for file in os.listdir(data_path):
    if ('imports' in file) and ('2dg' in file) and (file[0] != '.'):
        imports_2dg.append(file)

exports_2dg = []
for file in os.listdir(data_path):
    if ('exports' in file) and ('2dg' in file) and (file[0] != '.'):
        exports_2dg.append(file)

keepColumns = ['rt3ISO','rtCode','rtTitle','pt3ISO','ptCode','ptTitle','aggrLevel',
               'cmdCode','cmdDescE','yr','period','rgDesc','TradeValue']

imports_2dg_df = pd.DataFrame(columns = keepColumns)
for i,file in enumerate(imports_2dg):
    with open('%s/%s' % (data_path, file), mode = 'rU') as f:
        print 'appending %d/%d: %s' % (i+1, len(imports_2dg), file)
        df = pd.read_csv(f, sep = '\t', header = 0, usecols = keepColumns)
        imports_2dg_df = imports_2dg_df.append(df)

int_cols = ['rtCode','ptCode','aggrLevel','cmdCode','yr']
imports_2dg_df[int_cols] = imports_2dg_df[int_cols].apply(lambda x: x.astype(int))
imports_2dg_df.rename(columns={'rt3ISO': 'toISO', 'rtCode': 'toCode', 'rtTitle': 'toTitle',
                               'pt3ISO': 'fromISO', 'ptCode': 'fromCode', 'ptTitle': 'fromTitle'}, inplace=True)

exports_2dg_df = pd.DataFrame(columns = keepColumns)
for i,file in enumerate(exports_2dg):
    with open('%s/%s' % (data_path, file), mode = 'rU') as f:
        print 'appending %d/%d: %s' % (i+1, len(exports_2dg), file)
        df = pd.read_csv(f, sep = '\t', header = 0, usecols = keepColumns)
        exports_2dg_df = exports_2dg_df.append(df)

exports_2dg_df[int_cols] = exports_2dg_df[int_cols].apply(lambda x: x.astype(int))
exports_2dg_df.rename(columns={'rt3ISO': 'fromISO', 'rtCode': 'fromCode', 'rtTitle': 'fromTitle',
                               'pt3ISO': 'toISO', 'ptCode': 'toCode', 'ptTitle': 'toTitle'}, inplace=True)

print 'merging import and export datasets'
comtrade_2dg = imports_2dg_df.merge(exports_2dg_df,how = 'outer',
            on = ['fromISO','fromCode','fromTitle',
                  'toISO','toCode','toTitle',
                  'aggrLevel','cmdCode','cmdDescE',
                  'yr','period'])

comtrade_2dg['source'] = comtrade_2dg.rgDesc_x
comtrade_2dg.ix[pd.isnull(comtrade_2dg.source), 'source'] = comtrade_2dg.rgDesc_y
comtrade_2dg.rename(columns={'TradeValue_x': 'reportedByImporter', 'TradeValue_y': 'reportedByExporter'}, inplace=True)
comtrade_2dg['tradeValue'] = comtrade_2dg.reportedByImporter
comtrade_2dg.ix[comtrade_2dg.source == 'Export', 'tradeValue'] = comtrade_2dg.reportedByExporter
comtrade_2dg = comtrade_2dg.drop(['rgDesc_x','rgDesc_y'], axis = 1)

print 'saving to csv'
comtrade_2dg.to_csv('%s/comtrade_2dg.tsv' % data_path, index=None, sep='\t', encoding='utf-8')
comtrade_2000_2015_2dg = comtrade_2dg[comtrade_2dg.yr >= 2000]
comtrade_2000_2015_2dg.to_csv('%s/comtrade_2000_2015_2dg.tsv' % data_path, index=None, sep='\t', encoding='utf-8')

print 'done.'
