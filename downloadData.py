# -*- coding: utf-8 -*-

import sys
import json
import getTradeDataFromComtrade
import pandas as pd
import time
import os
import csv
import unicodedata
import requests


# get list of HS codes
with open('classificationHS.json') as f:
    HS_codes = json.load(f)

hs_codes = {}
for row in HS_codes[u'results']:
    code = str(row['id'])
    text = row['text']
    if code.isdigit():
        hs_codes[code] = text[text.find(' - ') + 3:] # remove numeric code from text label

hs_codes_2dg = {}
hs_codes_4dg = {}
hs_codes_6dg = {}
for code, text in hs_codes.iteritems():
    if len(code) <= 2:
        hs_codes_2dg[code] = text
    elif len(code) == 4 < 10000:
        hs_codes_4dg[code] = text
    else:
        hs_codes_6dg[code] = text

hs_codes_all =  list(hs_codes.keys())
hs_codes_all.sort()
hs_list_4dg =  list(hs_codes_4dg.keys())
hs_list_4dg.sort()
hs_list_6dg =  list(hs_codes_6dg.keys())
hs_list_6dg.sort()
hs_list_2dg =  list(hs_codes_2dg.keys())
hs_list_2dg.sort()
hs_list_4dg.sort()

# get list of country codes
comapi=getTradeDataFromComtrade.ComtradeApi()
country_codes = comapi._ctry_codes[['Country Code', 'Country Name English', 'ISO3-digit Alpha',
                                   'Start Valid Year', 'End Valid Year']]
country_codes.columns = ['code', 'name', 'iso', 'start', 'end']
country_codes.set_index('code')
country_codes = country_codes[country_codes.iso != 'NULL']

countries_all = [str(ctry) for ctry in country_codes.code.values.tolist()]
def get_name(code):
    unicode_name = ''.join(country_codes['name'][country_codes['code'] == int(code)].values)
    return unicodedata.normalize('NFKD', unicode_name).encode('ascii','ignore')

def ctry_validity(code):
    '''
    returns the range of years for which the country is valid. input ctry code string
    '''
    start_year = ''.join(country_codes['start'][country_codes['code'] == int(code)].values)
    end_year = ''.join(country_codes['end'][country_codes['code'] == int(code)].values)
    return range(int(start_year), int(end_year) + 1)

def get_ctry_index(name):
    '''
    inputs country name, returns its index in country_codes
    '''
    for i, bool in enumerate(country_codes['name'] == name):
        if bool == True:
            return i

#sys.exit("script stopped")

# Paratetrs for getComtradeData():
    # @param comcodes - array/list of HS country codes as strings
    # @param reporter - array/list of comtrade countries as list or use 'all' for
    # all countries
    # @param partner - same as above
    # @param years -array of years as integers
    # @param freq - 'A' or 'M' for year or monthly data respectively
    # @param rg='1' for imports and '2' for exports - only tested for imports
    # @param fmt='json' - default
    #@param rowmax=50000. No reason to change this. This is the max value.

def getComtradeAll(hs_list, ctry_list, rg, yrs, freq):
    aggregation_level = str(len(hs_list[0]))
    if not os.path.isfile('api_calls.csv'):
        with open("api_calls.csv", "w") as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['file_name', 'hs_codes', 'api_string', 'time_stamp', 'row_count'])
    if freq not in ('A', 'M'):
        print "did not understand freqency parameter."
        return
    if rg not in ('1', '2'):
        print "did not understand rg parameter. imports = '1', exports = '2'"
        return
    yr_ref = '_'.join([str(yrs[0]), str(yrs[-1])])
    # drop countries that don't exist in year range specificed
    ctry_list = [ctry for ctry in ctry_list if not set(ctry_validity(ctry)).isdisjoint(yrs)]
    for c in range(0,len(ctry_list),5):
        ctry_start = c
        if c + 5 < len(ctry_list):
            ctry_end = c + 5
            ctry_ref = "%s_to_%s" % (get_name(ctry_list[ctry_start]), get_name(ctry_list[c + 4]))
        else:
            ctry_end = None
            ctry_ref = "%s_to_%s" % (get_name(ctry_list[ctry_start]), get_name(ctry_list[-1]))
        if rg == '2':
            if freq == 'A':
                ref = 'annual_exports_%s_%s' % (yr_ref, ctry_ref)
            if freq == 'M':
                ref = 'monthly_exports_%s_%s' % (yr_ref, ctry_ref)
        if rg == '1':
            if freq == 'A':
                ref = 'annual_imports_%s_%s' % (yr_ref, ctry_ref)
            if freq == 'M':
                ref = 'monthly_imports__%s_%s' % (yr_ref, ctry_ref)
        for i in range(0,len(hs_list),20):
            start = i
            if i + 20 < len(hs_list):
                end = i + 20
            else:
                end = None
            if end != None:
                hs_str = '%s to %d' % (hs_list[start], int(hs_list[end]) - 1)
            else:
                hs_str = '%s to %s' % (hs_list[start], hs_list[-1])
            print "getting data for HS codes %s, countries %s" % (hs_str, ctry_ref)
            if i == 0:
                df, s, time_stamp = comapi.getComtradeData(comcodes = hs_list[start:end], partner = ctry_list[ctry_start:ctry_end], rg = [rg], years = yrs, freq = freq)
                rows = len(df)
            else:
                df_new, s, time_stamp = comapi.getComtradeData(comcodes = hs_list[start:end], partner = ctry_list[ctry_start:ctry_end], rg = [rg], years = yrs, freq = freq)
                rows = len(df_new)
                print "appending to df.."
                try:
                    df = df.append(df_new)
                except:
                    if len(df) == 0:
                        df = df_new
                print "df length is now: %d" % len(df)
            with open("api_calls.csv", "a") as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['%s_%sdg' % (ref, aggregation_level), hs_str, s, time_stamp, rows])
        print "saving %s to csv.." % ref
        df.to_csv('data/%s_%sdg.tsv' % (ref, aggregation_level), index = None, sep ='\t', encoding = 'utf-8')

if not os.path.exists('data'):
    os.makedirs('data')

#start_index = 0
last_ctry = 'Ukraine' # input name of last country downloaded
start_index = get_ctry_index(last_ctry) + 1

#getComtradeAll(hs_list_2dg, countries_all[start_index:], '1', range(1991, 1996), 'A')
#getComtradeAll(hs_list_2dg, countries_all[start_index:], '1', range(1996, 2001), 'A')
#getComtradeAll(hs_list_2dg, countries_all[start_index:], '1', range(2001, 2006), 'A')
#getComtradeAll(hs_list_2dg, countries_all[start_index:], '1', range(2006, 2011), 'A')
#getComtradeAll(hs_list_2dg, countries_all[start_index:], '1', range(2011, 2016), 'A')

#getComtradeAll(hs_list_2dg, countries_all[start_index:], '2', range(1986, 1991), 'A')

#getComtradeAll(hs_list_2dg, countries_all[start_index:], '2', range(1991, 1996), 'A')
#getComtradeAll(hs_list_2dg, countries_all[start_index:], '2', range(1996, 2001), 'A')
#getComtradeAll(hs_list_2dg, countries_all[start_index:], '2', range(2001, 2006), 'A')
#getComtradeAll(hs_list_2dg, countries_all[start_index:], '2', range(2006, 2011), 'A')
#getComtradeAll(hs_list_2dg, countries_all[start_index:], '2', range(2011, 2016), 'A')



# re-run requests that didn't return data
def get_failed():
    '''
    returns a list of tuples: (api_string, hs_code, file_name) for failed requests
    '''
    with open('api_calls.csv', mode='rU') as f:
        api_calls = pd.read_csv(f, sep=',')
    failed = []
    total_rows = api_calls.groupby(['api_string', 'hs_codes', 'file_name']).sum()
    for entry in total_rows.itertuples():
        if entry[1] == 0:
            failed.append(entry[0])
    return failed

failed = get_failed()

for api, hs_str, fname in failed:
    print "getting %s, HS codes %s" % (fname, hs_str)
    print "calling api: %s" % api
    time.sleep(1)
    r = requests.get(r'%s' % (api))
    data = r.json()
    df_new = pd.DataFrame(data['dataset'])
    print "returned df length: %d" % len(df_new)
    time_stamp = time.ctime()
    if len(df_new) > 0:
        # append new data if not already in fname
        with open("data/%s.tsv" % fname, "r") as f:
            df_old = pd.read_csv(f, sep='\t', header=0)
            duplicates = df_new.isin(df_old).all(axis = 1)
            if True in duplicates.values:
                print "HS codes %s already in %s, not appending." % (hs_str, fname)
                continue
        print "appending and saving %s" % fname
        df = df_new.append(df_old)
        df.to_csv('data/%s.tsv' % fname, index = None, sep ='\t', encoding = 'utf-8')
        # append to api_calls.csv
        with open("api_calls.csv", "a") as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([fname, hs_str, api, time_stamp, len(df_new)])

if len(get_failed()) == 0:
    print 'no failed api requests'

#sys.exit("script stopped")


# verify that row counts in downloaded files and 'api_calls.csv' are consistent
with open('api_calls.csv', mode='rU') as f:
    api_calls = pd.read_csv(f, sep=',')
    total_rows = api_calls.groupby(['file_name']).sum()
for fname, row_count in total_rows.itertuples():
    with open('data/%s.tsv' % fname, mode='rU') as f:
        data = list(csv.reader(f, delimiter='\t'))
        if len(data) - 1 != row_count:
            print 'WARNING: %s has %d rows but api_calls.csv suggests %d' % (fname, len(data) - 1, row_count)


# verify that downloaded files have expected HS codes
downloaded = [file for file in os.listdir('data')]
for file in downloaded:
    with open('data/%s' % file, mode = 'rU') as f:
        aggregation_level = file[-7]
        df = pd.read_csv(f, sep = '\t', header = 0)
        codes_found = set(df['cmdCode'])
        codes_not_found = []
        for code in globals()['hs_list_%sdg' % aggregation_level]:
            if int(code) not in codes_found and code != '77': # HS code 77 is 'resvered for future use'
                codes_not_found.append(int(code))
        if len(codes_not_found) > 35:
            print 'WARNING: %s is missing following hs codes:' % file
            print codes_not_found
        if (len(codes_not_found) > 15) and (max(codes_not_found) - min(codes_not_found) < 20):
            print 'WARNING: %s is missing following hs codes:' % file
            print codes_not_found

# check for duplicate rows
for file in downloaded:
    with open('data/%s' % file, mode = 'rU') as f:
        df = pd.read_csv(f, sep='\t', header=0)
        duplicates = df.duplicated()
        for index, value in duplicates.iteritems():
            if value:
                print 'WARNING: %s line %d is duplicated' % (file, index)
