# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import itertools
import unicodedata


def get_headings(dataset, path = 'data'):
    '''
    :param dataset: dataset to check column names
    :param path: location of dataset
    :return: list of headings
    '''
    with open('%s/%s' % (path, dataset), mode='rU') as f:
        headings = f.readline()
    return headings

def read_data(dataset, path = 'data'):
    if dataset[-3:] == 'csv':
        with open('%s/%s' % (path, dataset), mode='rU') as f:
            df = pd.read_csv(f, sep=',', header=0)
    else:
        with open('%s/%s' % (path, dataset), mode='rU') as f:
            df = pd.read_csv(f, sep='\t', header=0)
    return df

def ctry_lookup(value, code = True):
    if code: # use code to lookup name
        with open('UN Comtrade Country List.csv', mode='rU') as f:
            nested_list = [row.split(',') for row in f]
            for row in nested_list:
                if row[0] == value:
                    return row[1].decode('utf8')
    else: # use name to lookup code
        with open('UN Comtrade Country List.csv', mode='rU') as f:
            nested_list = [row.split(',') for row in f]
            for row in nested_list:
                if row[1] == value:
                    return row[0]

def cmd_lookup(value, code = True):
    with open('classificationHS.json') as f:
        HS_codes = json.load(f)
    if code: # use code to lookup description
        for row in HS_codes[u'results']:
            if row['id'] == value:
                return row['text']
    else: # use description to lookup code
        for row in HS_codes[u'results']:
            if row['text'] == value:
                return row['id']

def filter(dataset, keep_cols = [], filter_on=[], conditions=[], path ='data'):
    '''
    :param dataset: name of file to import
    :param keep_cols: list of columns to keep
    :param filter_on: list of columns on which to filter, eg. ["yr", "TradeValue"]
    :param filter_on: conditions relating to each column in filter_on. eg. ["> 2010", "> 100000"]
    :param path: location of input file
    :return: sliced dataframe
    '''
    df = read_data(dataset, path = path)
    if len(keep_cols) != 0:
        df = df[keep_cols]
    assert len(conditions) == len(filter_on)
    for i in range(len(conditions)):
        df = eval('df[df.'+eval('%s[%d]' % (filter_on, i))+eval('%s[%d]' % (conditions, i))+']')
    return df

def ctry_cmd_shares(dataset, yr = 2015, path = 'data'):
    df = filter(dataset, filter_on = ['yr'], conditions = ['==%d' % yr], path = path)
    cmds = set(df.cmdCode)
    ctries = set(df.fromCode)
    ctry_cmd_totals = df.groupby(['fromCode', 'cmdCode']).apply(lambda x: pd.Series(dict(
            exportValue=x.tradeValue.sum()
    ))).reset_index()
    ctry_totals = df.groupby(['fromCode']).apply(lambda x: pd.Series(dict(
            exportTotal=x.tradeValue.sum()
    ))).reset_index()
    ctry_cmd_shares = ctry_cmd_totals.merge(ctry_totals, on = 'fromCode', right_index = True)
    ctry_cmd_shares = ctry_cmd_shares.set_index(['fromCode', 'cmdCode'])
    ctry_cmd_shares['exportShare'] = ctry_cmd_shares['exportValue'] / ctry_cmd_shares['exportTotal']
    # fill in missing commodities
    complete_index = list(itertools.product(ctries, cmds))
    ctry_cmd_shares = ctry_cmd_shares.reindex(complete_index, fill_value=0)
    ctry_cmd_shares.reset_index(inplace=True)
    ctry_cmd_shares['hasExport'] = (ctry_cmd_shares['exportShare'] > 0).values.astype(np.int)
    return ctry_cmd_shares[['fromCode','cmdCode','exportShare', 'hasExport']]

def ctry_matrix(dataset, yr=2015, cols = 'cmd', path = 'data', imports = False):
    records = filter(dataset, keep_cols = ['fromCode', 'toCode', 'yr', 'tradeValue', 'cmdCode'],
                     filter_on=['yr'],
                     conditions=['==%d' % yr])
    if imports:
        records.rename(columns={'toCode': 'rows'}, inplace=True)
    else:
        records.rename(columns={'fromCode': 'rows'}, inplace=True)
    if cols == 'cmd':
        records.rename(columns={'cmdCode': 'cols'}, inplace=True)
    else:
        assert cols == 'ctry'
        if imports:
            records.rename(columns={'fromCode': 'cols'}, inplace=True)
        else:
            records.rename(columns={'toCode': 'cols'}, inplace=True)

    records_totals = records.groupby(['rows', 'cols']).apply(lambda x: pd.Series(dict(
        tradeValue=x.tradeValue.sum()
    ))).reset_index()

    matrix = records_totals.pivot(index='rows', columns='cols', values='tradeValue')
    if cols == 'ctry':
        ctries_all = list(set(matrix.index.values) | set(matrix.columns.values))
        matrix = matrix.reindex(ctries_all, fill_value=0)
    matrix = matrix.fillna(value=0)
    matrix = matrix.loc[:, (matrix != 0).any(axis=0)] # remove columns with all zeros
    matrix = matrix.loc[(matrix != 0).any(axis=1)] # remove rows with all zeros
    return matrix

def save(df, save_as, csv = False, path = 'data'):
    '''
    :param df: df to save
    :param save_as: file name
    :param csv: save as csv. default is tsv
    :param path: location to save
    '''
    if csv:
        df.to_csv('%s/%s.csv' % (path, save_as), index=None, sep=',', encoding='utf-8')
    else:
        df.to_csv('%s/%s.tsv' % (path, save_as), index=None, sep='\t', encoding='utf-8')

if __name__ == "__main__":
    # this won't be run when imported
    print get_headings('comtrade_2dg.tsv')

    df = filter('comtrade_2dg.tsv', keep_cols = ['fromCode', 'toCode', 'yr', 'tradeValue', 'cmdCode'],
                filter_on = ['yr'],
                conditions = ['==2010'])

    save(df, 'comtrade_2010_2dg')