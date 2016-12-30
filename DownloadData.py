# -*- coding: utf-8 -*-
import collections
import csv
import os
import time
import unicodedata
from datetime import datetime, timedelta
import pandas as pd
import requests



def get_ctry_codes_df():
    with open('UN Comtrade Country List.csv', mode='rU') as f:
        ctry_codes = pd.read_csv(f, sep=',', header=0)
    ctry_codes = ctry_codes[['Country Code', 'Country Name English','ISO2-digit Alpha', 'ISO3-digit Alpha',
                                   'Start Valid Year', 'End Valid Year']]
    ctry_codes.columns = ['code', 'name', 'iso2', 'iso', 'start', 'end']
    ctry_codes['code'] = ctry_codes['code'].apply(str)
    ctry_codes = ctry_codes[ctry_codes.code != '0'] #remove 'world'
    remove = [ # countries/territories that don't interest me
        'AIA', # Anguilla
        'ANT', # Antartica
        'ATA', # Antartica
        'BVT', # Bouvet Island
        'IOT', # British Indian Ociean Terr.
        'CXR', # Christmas island
        'CCK', # Cook islands
        'COK', # Cocos is
        'FLK', # Falklands
        'ATF', # South Antartic Terr
        'HMD', # Heard and McDonald Island
        'VAT', # Vatican
        'MYT', # Mayotte
        'MSR', # Montserrat
        'NIU', # Nieu
        'NFK', # Norfolk Islands
        'PCN', # Pitcairn
        'SHN', # Saint Helena
        'SPM', # Saint Pierre and Miguelon
        'SGS', # South Georgia and the South Sandwich Islands
        'TKL', # Tokelau
        'UMI', # US Minor Outlying Islands
        'WLF', # Walls and Futuna
        'ESH'  # Western Sahara
    ]
    for iso in remove:
        ctry_codes = ctry_codes[ctry_codes.iso != iso]
    ctry_codes.set_index('code')
    ctry_codes['name'] = ctry_codes['name'].apply(lambda x: x.decode('utf-8'))
    ctry_codes = ctry_codes[pd.notnull(ctry_codes.iso)]
    ctry_codes['start'] = ctry_codes['start'].apply(lambda x: int(x))
    ctry_codes['end'] = ctry_codes['end'].apply(lambda x: int(x))
    return ctry_codes

def ctry_tup(col, years):
    '''
    :param col: col in ctry_codes_df to return - 'code', 'name', 'iso', 'start', 'end'
    :param years: list of years as integers. countries that didn't exist for whole of the year range are removed
    :return: col as sequence of strings
    '''
    country_codes = get_ctry_codes_df()
    for year in years:
        country_codes = country_codes[country_codes.start <= year]
        country_codes = country_codes[country_codes.end >= year]
    countries_list = [str(ctry) for ctry in country_codes[col].values.tolist()]
    return tuple(countries_list)


def get_name(code):
    country_codes = get_ctry_codes_df()
    unicode_name = ''.join(country_codes['name'][country_codes['code'] == code].values)
    return unicodedata.normalize('NFKD', unicode_name).encode('ascii','ignore')

def get_ctry_index(name, ctry_list):
    '''
    inputs country name, returns its index in ctry_list (list of ctry codes)
    '''
    country_codes = get_ctry_codes_df()
    try:
        df_index = country_codes[country_codes.name == name].index.tolist()[0]
    except:
        country_codes['unicodename'] = country_codes['code'].apply(lambda x: get_name(x))
        df_index = country_codes[country_codes.unicodename == name].index.tolist()[0]
    code = country_codes.code[df_index]
    for i, ctry_code in enumerate(ctry_list):
        if ctry_code == code:
            return i
    print "could not find index for %s" % name


def get_start_index(rg, hs_list, ctry_list, years, num_com):
    #return 0,0  # uncomment for emergency restart
    with open("api_calls.csv", mode='rU') as f:
        api_calls = pd.read_csv(f, sep=',', header=0)
    api_calls['rg'] = api_calls['api_string'].apply(lambda x: x[x.find('rg=') + 3:x.find('rg=') + 4])
    api_calls = api_calls[api_calls.rg == rg]
    yr_string = "%2C".join([str(yr) for yr in years])
    api_calls['current_yrs'] = api_calls['api_string'].apply(lambda x: yr_string in x)
    api_calls = api_calls[api_calls.current_yrs]
    api_calls['hs_digits'] = api_calls['file_name'].apply(lambda x: x[-3:-2])
    api_calls = api_calls[api_calls.hs_digits == str(len(hs_list[0]))]
    api_calls = api_calls[api_calls.row_count != 0]  # remove failed api requests
    try:
        last_ref = collections.OrderedDict.fromkeys(api_calls['file_name'].values).items()[-2][0] #last completed file
        current_ref = collections.OrderedDict.fromkeys(api_calls['file_name'].values).items()[-1][0]
        ctry_end = last_ref.rfind('_')
        ctry_start = last_ref[:ctry_end].rfind('_') + 1
        last_ctry = last_ref[ctry_start:ctry_end]
        last_calls = api_calls[api_calls.file_name == current_ref]
        cmd_start_index = (last_calls.shape[0])*num_com
    except IndexError:
        cmd_start_index, ctry_start_index = 0,0
        return ctry_start_index, cmd_start_index
    except Exception:
        cmd_start_index, ctry_start_index = len(ctry_list),len(hs_list)
        return ctry_start_index, cmd_start_index
    if cmd_start_index >= len(hs_list):
        cmd_start_index = 0
        ctry_start_index = get_ctry_index(last_ctry, ctry_list) + 6
    else:
        if last_ctry == None:
            ctry_start_index = 0
        else:
            ctry_start_index = get_ctry_index(last_ctry, ctry_list) + 1
    return ctry_start_index, cmd_start_index


def buildApiString(freq,years,ctry_list,rg,hs_list):
    s = "http://comtrade.un.org/api/get?type=C"
    s = "%s&freq=%s&px=HS" % (s, freq)
    if freq == 'A':
        s = "%s&ps=%s" % (s, "%2C".join([str(yr) for yr in years]))
    else:
        mons = [["%s%02d" % (yr, mon) for mon in range(1, 13)]
                for yr in years]
        s = "%s&ps=%s" % (s, "%2C".join(mons[0]))
    s = "%s&r=all" % s
    s = "%s&p=%s" % (s, "%2C".join(ctry_list))
    s = "%s&rg=%s" % (s, "%2C".join(rg))
    s = "%s&cc=%s" % (s, "%2C".join(hs_list))
    s = "%s&fmt=json&max=50000&head=M" % s
    return s

def apiCall(s):
    print "api call: %s" % s
    time_stamp = time.ctime()
    try:
        r = requests.get(r'%s' % (s))
    except Exception:
        try:
            r = requests.get(r'%s' % (s), verify=False)
        except requests.exceptions.ConnectionError:
            print "Connection refused. Will try again later."
            return [], time_stamp
    try:
        data = r.json()
        df = pd.DataFrame(data['dataset'])
        print "returned %r, length: %r" % (type(df), len(df))
        if len(df) == 0:
            # wait 1 sec and try again. if that fails wait 1 hour.
            time.sleep(1)
            try:
                r = requests.get(r'%s' % (s), verify=False)
                data = r.json()
                df = pd.DataFrame(data['dataset'])
            except Exception:
                one_hour_from_now = datetime.now() + timedelta(hours=1)
                print "Reached hourly call limit. Sleeping until %s" % format(one_hour_from_now, '%H:%M:%S')
                time.sleep(3600)
    except ValueError:
        one_hour_from_now = datetime.now() + timedelta(hours=1)
        print "Reached hourly call limit. Sleeping until %s" % format(one_hour_from_now, '%H:%M:%S')
        time.sleep(3600)
        df = []
    except Exception:
        print "No data to parse. Will try again later."
        df = []
    return df, time_stamp

def getComtrade(hs_list, ctry_list, rg, yrs, freq = 'A', num_com = 20):
    '''
    :param hs_list: array/list of HS country codes as strings
    :param ctry_list: list of country codes as strings
    :param rg: rg='1' for imports and '2' for exports - only tested for imports
    :param yrs: array of years as integers
    :param freq: 'A' or 'M' for year or monthly data respectively
    :param num_com: Number of commodities for each api call. No reason to change this. This is the max value.
    :return:
    '''
    if not os.path.exists('data'):
        os.makedirs('data')
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

    ctry_start_index = get_start_index(rg, hs_list, ctry_list, yrs, num_com)[0]
    ctry_sublist = ctry_list[ctry_start_index:]
    for c in range(0,len(ctry_sublist),5):
        ctry_start = c
        if c + 5 < len(ctry_sublist):
            ctry_end = c + 5
            ctry_ref = "%s_to_%s" % (get_name(ctry_sublist[ctry_start]), get_name(ctry_sublist[c + 4]))
        else:
            ctry_end = None
            ctry_ref = "%s_to_%s" % (get_name(ctry_sublist[ctry_start]), get_name(ctry_sublist[-1]))
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
        cmd_start_index = get_start_index(rg, hs_list, ctry_list, yrs, num_com)[1]
        for i in range(cmd_start_index,len(hs_list),num_com):
            start = i
            if i + num_com < len(hs_list):
                end = i + num_com
            else:
                end = None
            if end != None:
                hs_str = '%s to %d' % (hs_list[start], int(hs_list[end]) - 1)
            else:
                hs_str = '%s to %s' % (hs_list[start], hs_list[-1])
            print "getting data for HS codes %s, countries %s" % (hs_str, ctry_ref)
            if i == 0:
                s = buildApiString(freq, yrs, ctry_sublist[ctry_start:ctry_end], rg, hs_list[start:end])
                df, time_stamp = apiCall(s)
                rows = len(df)
                if rows > 0:
                    print "creating %s text file" % ref
                    df.to_csv('data/%s_%sdg.tsv' % (ref, aggregation_level), index = None, sep ='\t', encoding = 'utf-8')
            else:
                s = buildApiString(freq, yrs, ctry_sublist[ctry_start:ctry_end], rg, hs_list[start:end])
                df, time_stamp = apiCall(s)
                rows = len(df)
                try:
                    if rows > 0:
                        print "appending to %s text file" % ref
                        with open('data/%s_%sdg.tsv' % (ref, aggregation_level), 'a') as f:
                            df.to_csv(f, header=False, index = None, sep ='\t', encoding = 'utf-8')
                except:
                    if not os.path.isfile('data/%s_%sdg.tsv' % (ref, aggregation_level)):
                        df.to_csv('data/%s_%sdg.tsv' % (ref, aggregation_level), index = None, sep ='\t', encoding = 'utf-8')
            with open("api_calls.csv", "a") as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['%s_%sdg' % (ref, aggregation_level), hs_str, s, time_stamp, rows])
        recall_failed()


def hs_dict():
    import json
    with open('classificationHS.json') as f:
        json = json.load(f)
    hs_codes_dict = {}
    for row in json[u'results']:
        code = str(row['id'])
        text = row['text']
        if code.isdigit():
            hs_codes_dict[code] = text[text.find(' - ') + 3:] # remove numeric code from text label
    return hs_codes_dict

def hs_list(aggregation_level):
    '''
    :param aggregation_level: = 2 or 4 or 6. Number of hs code digist.
    :return: sorted list of hs codes
    '''
    hs_all = hs_dict()
    hs_subset = {}
    for code, text in hs_all.iteritems():
        if len(code) <= aggregation_level and len(code) >=aggregation_level-1:
            hs_subset[code] = text
    hs_list = list(hs_subset.keys())
    hs_list.sort()
    return hs_list

def get_failed():
    # returns a list of tuples: (api_string, hs_code, file_name) for failed requests
    with open('api_calls.csv', mode='rU') as f:
        api_calls = pd.read_csv(f, sep=',')
    failed = []
    total_rows = api_calls.groupby(['api_string', 'hs_codes', 'file_name']).sum()
    for entry in total_rows.itertuples():
        if entry[1] == 0:
            failed.append(entry[0])
    return failed


def recall_failed():
    while len(get_failed()) > 0:
        for api, hs_str, fname in get_failed():
            print "getting missing data: %s, HS codes %s" % (fname, hs_str)
            df_new, time_stamp = apiCall(api)
            if len(df_new) > 0:
                # append new data if not already in fname
                with open("data/%s.tsv" % fname, "r") as f:
                    df_old = pd.read_csv(f, sep='\t', header=0)
                    duplicates = df_new.isin(df_old).all(axis=1)
                    if True in duplicates.values:
                        print "HS codes %s already in %s, not appending." % (hs_str, fname)
                        continue
                print "appending and saving %s" % fname
                df = df_new.append(df_old)
                df.to_csv('data/%s.tsv' % fname, index=None, sep='\t', encoding='utf-8')
                # append to api_calls.csv
                with open("api_calls.csv", "a") as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow([fname, hs_str, api, time_stamp, len(df_new)])
    print "All api requests sucessfull."

def getComtradeAll(hs_list, start_yr, end_yr, imports = True,freq = 'A', num_com=20):
    ctries = ctry_tup('code', range(start_yr,end_yr+1))
    if imports:
        if end_yr - start_yr < 5:
            getComtrade(hs_list, ctries, '1', range(start_yr, end_yr+1),freq=freq,num_com=num_com)
        else:
            for i,yr in enumerate(range(start_yr,end_yr,5)):
                getComtrade(hs_list, ctries, '1', range(yr, yr+5),freq=freq,num_com=num_com)
    if not imports:
        if end_yr - start_yr < 5:
            getComtrade(hs_list, ctries, '2', range(start_yr, end_yr+1),freq=freq,num_com=num_com)
        else:
            for i,yr in enumerate(range(start_yr,end_yr,5)):
                getComtrade(hs_list, ctries, '2', range(yr, yr+5),freq=freq,num_com=num_com)

def longest_consecutive_range(num_list):
    # inputs list of integers, returns largest range of consecutive integers
    # used by data_checks to check missing hs_codes
    from itertools import groupby
    from operator import itemgetter
    num_list.sort()
    longest_range = []
    for k, g in groupby(enumerate(num_list), lambda (i, x): i - x):
            range = map(itemgetter(1), g)
            if len(range) > len(longest_range):
                longest_range = range
    return longest_range


def data_checks(path):
    # verify that row counts in downloaded files and 'api_calls.csv' are consistent
    with open('api_calls.csv', mode='rU') as f:
        api_calls = pd.read_csv(f, sep=',')
        total_rows = api_calls.groupby(['file_name']).sum()
    for fname, row_count in total_rows.itertuples():
        try:
            with open('%s/%s.tsv' % (path,fname), mode='rU') as f:
                data = list(csv.reader(f, delimiter='\t'))
                if len(data) - 1 != row_count:
                    print 'WARNING: %s has %d rows but api_calls.csv suggests %d' % (fname, len(data) - 1, row_count)
        except:
            print 'WARNING: %s/%s.tsv not found' % (path,fname)

    # verify that downloaded files have expected HS codes
    downloaded = [file for file in os.listdir(path)]
    dont_check = []
    for file in downloaded:
        with open('%s/%s' % (path,file), mode='rU') as f:
            aggregation_level = file[file.index('dg.')-1:file.index('dg.')]
            df = pd.read_csv(f, sep='\t', header=0)
            try:
             # will fail for any files in path not generated by this script
                codes_found = set(df['cmdCode'])
                codes_not_found = []
                for code in hs_list(aggregation_level):
                    if int(code) not in codes_found and code != '77':  # HS code 77 is 'reserved for future use'
                        codes_not_found.append(int(code))
                if len(codes_not_found) > 0:
                    missing_range = longest_consecutive_range(codes_not_found)
                    if len(missing_range) > 20:
                        print 'WARNING: %s is missing %d consecutive hs codes (%d to %d)' % (file,len(missing_range),
                                                                                 missing_range[0],missing_range[-1])
            except:
                dont_check.append(file)
                print '%s not checked' % file

    # check for duplicate rows
    check = [file for file in downloaded if file not in dont_check]
    for file in check:
        with open('%s/%s' % (path,file), mode='rU') as f:
            df = pd.read_csv(f, sep='\t', header=0)
            duplicates = df.duplicated()
            for index, value in duplicates.iteritems():
                if value:
                    print 'WARNING: %s line %d is duplicated' % (file, index)

# Country data from the World Bank
def get_ctry_data():
    # returns a dataframe with each country's latitude, longitude, region and current income classification
    import wbdata
    ctry_dict_list = wbdata.search_countries('')
    ctry_data = pd.DataFrame(columns=['iso','country','latitude','longitude','region','incomeLevel'])
    for ctry_dict in ctry_dict_list:
        ctry_data = ctry_data.append({'iso': ctry_dict['id'],'country': ctry_dict['name'],
                                'latitude': ctry_dict['latitude'],'longitude':ctry_dict['longitude'],
                                'region':ctry_dict['region']['value'],'incomeLevel':ctry_dict['incomeLevel']['value']},
                                ignore_index=True)
    ctry_data = ctry_data[ctry_data.latitude != ''] # remove non-countries
    ctry_data = ctry_data[pd.notnull(ctry_data.iso)] # remove non-countries
    ctry_data = ctry_data.reset_index()
    return ctry_data

def get_wb(indictor, start_year, end_year):
    import wbdata
    import datetime
    start_date = datetime.date(year=start_year, month=1, day=1)
    end_date = datetime.date(year=end_year, month=1, day=1)
    series = wbdata.get_data(indicator = indictor, country = ctry_tup('iso',range(start_year,end_year)),
                     data_date=(start_date,end_date),pandas=True)
    df = series.to_frame(name = 'value').reset_index()
    df.rename(columns={'date': 'year'}, inplace=True)
    df = get_ctry_data()[['iso', 'country']].merge(df,on='country') # necessary to have ISO codes in dataframe
    return df

if __name__ == "__main__":
    getComtradeAll(hs_list(4), 1996, 2015)
    getComtradeAll(hs_list(4), 1996, 2015, imports = False)
    data_checks(path='data')

