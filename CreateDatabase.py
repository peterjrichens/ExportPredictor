import os
import json
import pandas as pd
from sqlalchemy import create_engine
from dbModel import db
from config import SQLALCHEMY_DATABASE_URI
from DownloadData import get_ctry_codes_df, get_ctry_data

engine = create_engine(SQLALCHEMY_DATABASE_URI)
db.create_all()

def clean_df_db_dups(df, tablename, engine, dup_cols=[],
                         filter_continuous_col=None, filter_categorical_col=None):
    """
    source: https://github.com/ryanbaumann/Pandas-to_sql-upsert/blob/master/to_sql_newrows.py

    Remove rows from a dataframe that already exist in a database
    Required:
        df : dataframe to remove duplicate rows from
        engine: SQLAlchemy engine object
        tablename: tablename to check duplicates in
        dup_cols: list or tuple of column names to check for duplicate row values
    Optional:
        filter_continuous_col: the name of the continuous data column for BETWEEEN min/max filter
                               can be either a datetime, int, or float data type
                               useful for restricting the database table size to check
        filter_categorical_col : the name of the categorical data column for Where = value check
                                 Creates an "IN ()" check on the unique values in this column
    Returns
        Unique list of values from dataframe compared to database table
    """
    args = 'SELECT %s FROM %s' % (', '.join(['"{0}"'.format(col) for col in dup_cols]), tablename)
    args_contin_filter, args_cat_filter = None, None
    if filter_continuous_col is not None:
        if df[filter_continuous_col].dtype == 'datetime64[ns]':
            args_contin_filter = """ "%s" BETWEEN Convert(datetime, '%s')
                                          AND Convert(datetime, '%s')""" % (filter_continuous_col,
                                                                            df[filter_continuous_col].min(),
                                                                            df[filter_continuous_col].max())

    if filter_categorical_col is not None:
        args_cat_filter = ' "%s" in(%s)' % (filter_categorical_col,
                                            ', '.join(["'{0}'".format(value) for value in
                                                       df[filter_categorical_col].unique()]))

    if args_contin_filter and args_cat_filter:
        args += ' Where ' + args_contin_filter + ' AND' + args_cat_filter
    elif args_contin_filter:
        args += ' Where ' + args_contin_filter
    elif args_cat_filter:
        args += ' Where ' + args_cat_filter

    df.drop_duplicates(dup_cols, keep='last', inplace=True)
    df = pd.merge(df, pd.read_sql(args, engine), how='left', on=dup_cols, indicator=True)
    df = df[df['_merge'] == 'left_only']
    df.drop(['_merge'], axis=1, inplace=True)
    return df

def country_db():
    df = get_ctry_codes_df()[['code', 'name', 'iso2', 'iso']]
    df.code = df['code'].apply(lambda x: int(x))
    ctry_data = get_ctry_data()[['iso', 'latitude', 'longitude', 'region', 'incomeLevel']]
    ctry_data.replace({'iso': {'ABW':'ARB','ROU':'ROM','TLS':'TMP'}}, inplace=True) # different ISO codes for Aruba, Romania and Timor-Leste
    ctry_data.rename(columns = {'incomeLevel': 'income_level'}, inplace=True)
    df = df.merge(ctry_data, how='left', on='iso')
    try:
        df = clean_df_db_dups(df, 'countries', engine, dup_cols=['code'])
        df.to_sql(name='countries', if_exists='append', con=engine, index=False)
    except ValueError as exception:
        print exception

def commodity_db():
    with open('classificationHS.json') as data_file:
        dict = json.load(data_file)['results']
    df = pd.DataFrame.from_dict(dict)
    df.columns = ['code', 'parent', 'name']
    parents = df[df.code.map(len) == 2][['code', 'name']]
    df = df[df.code.map(len) == 4]
    df = df[df.code.str.isnumeric() == True]
    df.code = df['code'].apply(lambda x: int(x))
    df.name = df.name.apply(lambda x: x[7:])
    df = df.merge(parents, how='left', left_on='parent',right_on='code', suffixes=('','_parent'))
    df = df.drop('code_parent', axis=1)
    df.name_parent = df.name_parent.apply(lambda x: x[5:])
    df.parent = df['parent'].apply(lambda x: int(x))
    section_map = {
        tuple(range(1,6)): 'Animals & animal products',
        tuple(range(6,15)): 'Vegetable products',
        tuple(range(15,25)): 'Foodstuffs',
        tuple(range(25,28)): 'Mineral products',
        tuple(range(28,39)): 'Chemicals and allied industries',
        tuple(range(39,41)): 'Plastics & rubber',
        tuple(range(41,44)): 'Hides, skins, leather & fur',
        tuple(range(44,50)): 'Wood and wood/paper products',
        tuple(range(50,64)): 'Textiles',
        tuple(range(64,68)): 'Footwear & headgear',
        tuple(range(68,72)): 'Stone & glass',
        tuple(range(72,84)): 'Metals',
        tuple(range(84,86)): 'Machinery/electrical',
        tuple(range(86,90)): 'Transportation',
        tuple(range(90,100)): 'Miscellaneous'
    }
    df['name_1dg'] = df.parent.apply(lambda x: [section_map[key] for key in section_map.keys() if x in key][0])
    df.rename(columns={'parent': 'code_2dg', 'name_parent': 'name_2dg'}, inplace=True)
    try:
        df = clean_df_db_dups(df, 'commodities', engine, dup_cols=['code'])
        df.to_sql(name='commodities', if_exists='append', con=engine, index=False)
    except ValueError as exception:
        print exception


def buildDatabase(aggregation_level ='4dg', start_year = 1996, end_year = 2015, postgres=True, csv=False,
                  sample = False, max_rows = 50000000, data_path = 'data', remove_oil = False):
    '''
    :param aggregation_level: HS product code aggregation ('2dg', '4dg' or '6dg'). postgres db designed for 4 digit
    :param start_year: 1986 <= integer <= 2011 ending in 1 or 6
    :param end_year: 1990 <= integer <= 2015 ending in 0 or 5
    :param postgres: add output to postgres database
    :param csv: save output to text file(s) of length <= max_rows
    :param sample: False saves all data (to multiple files if necessary). True saves a sample of length max_rows
    :param max_rows: maximum number of rows per file
    :param data_path: location of the tsv files generated by DownloadData.py
    :param remove_oil: remove mineral fuels, oils, distillation products
    '''
    years = [str(yr) for yr in range(start_year, end_year, 5)]
    for year in years:
        imports = []
        exports = []
        for file in os.listdir(data_path):
            file_start_yr = file[file.find('_',10)+1:file.find('_',10)+5]
            if ('imports' in file) and (aggregation_level in file) and (file_start_yr == year) and (file[0] != '.'):
                imports.append(file)
            if ('exports' in file) and (aggregation_level in file) and (file_start_yr == year) and (file[0] != '.'):
                exports.append(file)

        if len(imports) == 0:
            print 'No files matching those parameters found in %s' % data_path
            return

        if sample:
            # only use first 10 files
            if len(imports) > 10:
                imports = imports[:10]
            if len(exports) > 10:
                exports = exports[:10]

        headers = ['AltQuantity','CIFValue', 'FOBValue', 'GrossWeight', 'IsLeaf', 'NetWeight', 'TradeQuantity',
                   'TradeValue', 'aggrLevel', 'cmdCode', 'cmdDescE', 'cstCode', 'cstDesc', 'estCode', 'motCode',
                   'motDesc', 'period', 'periodDesc', 'pfCode', 'pt3ISO', 'pt3ISO2', 'ptCode', 'ptCode2',
                   'ptTitle', 'ptTitle2', 'qtAltCode', 'qtAltDesc', 'qtCode', 'qtDesc', 'rgCode', 'rgDesc',
                   'rt3ISO', 'rtCode', 'rtTitle', 'yr']

        keepColumns = ['rt3ISO','rtCode','rtTitle','pt3ISO','ptCode','ptTitle','aggrLevel',
                       'cmdCode','cmdDescE','yr','period','rgDesc','TradeValue']

        imports_df = pd.DataFrame(columns = keepColumns)
        for i,file in enumerate(imports):
            with open('%s/%s' % (data_path, file), mode = 'rU') as f:
                print 'appending %d/%d: %s' % (i+1, len(imports), file)
                try:
                    df = pd.read_csv(f, sep = '\t', header=0, usecols=keepColumns)
                except ValueError:
                    df = pd.read_csv(f, sep = '\t', header=None, names=headers, usecols=keepColumns)
                imports_df = imports_df.append(df)

        imports_df = imports_df.dropna(axis=0, how='all')
        imports_df = imports_df.dropna(axis=0, how='any', subset=['cmdCode'])
        int_cols = ['rtCode','ptCode','aggrLevel','cmdCode','yr']
        imports_df[int_cols] = imports_df[int_cols].apply(lambda x: x.astype(int))
        imports_df.rename(columns={'rt3ISO': 'toISO', 'rtCode': 'toCode', 'rtTitle': 'toTitle',
                                       'pt3ISO': 'fromISO', 'ptCode': 'fromCode', 'ptTitle': 'fromTitle'}, inplace=True)


        exports_df = pd.DataFrame(columns = keepColumns)
        for i,file in enumerate(exports):
            with open('%s/%s' % (data_path, file), mode = 'rU') as f:
                print 'appending %d/%d: %s' % (i+1, len(exports), file)
                try:
                    df = pd.read_csv(f, sep = '\t', header = 0, usecols = keepColumns)
                except ValueError:
                    df = pd.read_csv(f, sep='\t', header=None, names=headers, usecols=keepColumns)
                exports_df = exports_df.append(df)
        exports_df = exports_df.dropna(axis=0, how='any')
        exports_df = exports_df.dropna(axis=0, how='any', subset=['cmdCode'])
        exports_df[int_cols] = exports_df[int_cols].apply(lambda x: x.astype(int))
        exports_df.rename(columns={'rt3ISO': 'fromISO', 'rtCode': 'fromCode', 'rtTitle': 'fromTitle',
                                       'pt3ISO': 'toISO', 'ptCode': 'toCode', 'ptTitle': 'toTitle'}, inplace=True)

        for yr in range(int(year),int(year)+5):
            imports_yr = imports_df[imports_df.yr == yr]
            exports_yr = exports_df[exports_df.yr == yr]


            print 'merging import and export datasets for %d' % yr
            comtrade = imports_yr.merge(exports_yr,how = 'outer',
                        on = ['fromISO','fromCode','fromTitle',
                              'toISO','toCode','toTitle',
                              'aggrLevel','cmdCode','cmdDescE',
                              'yr','period'])

            comtrade['source'] = comtrade.rgDesc_x
            comtrade.ix[pd.isnull(comtrade.source), 'source'] = comtrade.rgDesc_y
            comtrade.rename(columns={'TradeValue_x': 'reportedByImporter', 'TradeValue_y': 'reportedByExporter'}, inplace=True)
            comtrade['tradeValue'] = comtrade.reportedByImporter
            comtrade.ix[comtrade.source == 'Export', 'TradeValue'] = comtrade.reportedByExporter
            comtrade = comtrade.drop(['rgDesc_x','rgDesc_y'], axis = 1)
            comtrade.yr = comtrade.yr.map(int)

            remove = [  # countries/territories that don't interest me
                'AIA',  # Anguilla
                'ANT',  # Antartica
                'ATA',  # Antartica
                'BVT',  # Bouvet Island
                'IOT',  # British Indian Ociean Terr.
                'CXR',  # Christmas island
                'CCK',  # Cook islands
                'COK',  # Cocos is
                'FLK',  # Falklands
                'ATF',  # South Antartic Terr
                'HMD',  # Heard and McDonald Island
                'VAT',  # Vatican
                'MYT',  # Mayotte
                'MSR',  # Montserrat
                'NIU',  # Nieu
                'NFK',  # Norfolk Islands
                'PCN',  # Pitcairn
                'SHN',  # Saint Helena
                'SPM',  # Saint Pierre and Miguelon
                'SGS',  # South Georgia and the South Sandwich Islands
                'TKL',  # Tokelau
                'UMI',  # US Minor Outlying Islands
                'WLF',  # Walls and Futuna
                'ESH'  # Western Sahara
            ]
            keep_cols = ['fromCode', 'fromISO', 'toCode', 'toISO', 'yr', 'tradeValue', 'cmdCode']
            comtrade = comtrade[keep_cols]
            comtrade = comtrade.dropna(axis=0, how='any', subset=['fromISO'])
            comtrade = comtrade.dropna(axis=0, how='any', subset=['toISO'])
            comtrade = comtrade.dropna(axis=0, how='any', subset=['tradeValue'])
            comtrade = comtrade[comtrade.tradeValue != 0]

            for iso in remove:
                comtrade = comtrade[comtrade.fromISO != iso]
                comtrade = comtrade[comtrade.toISO != iso]

            digits = int(aggregation_level[0])

            if remove_oil:
                comtrade['oil'] = comtrade.cmdCode.apply(lambda x: str(x).zfill(digits)[:2] == '27')
                comtrade = comtrade[comtrade.oil == False]
                comtrade = comtrade.drop('oil', axis=1)

            if postgres:
                db.create_all()
                country_db()
                commodity_db()
                comtrade = comtrade[['fromCode', 'toCode', 'cmdCode', 'yr', 'tradeValue']]
                comtrade.columns = ['origin', 'destination', 'cmd', 'yr', 'value']
                print 'Checking dataframe against database'
                new_rows = clean_df_db_dups(comtrade, 'comtrade', engine, dup_cols=['origin', 'destination', 'cmd', 'yr'],
                                            filter_categorical_col='yr')
                print 'Inserting new rows into database'
                new_rows.to_sql(name='comtrade', if_exists='append', con=engine, index=False)

            if csv:
                save_as = 'comtrade_%d_%d_%s' % (start_year,end_year,aggregation_level)
                if sample:
                    save_as = 'comtrade_%d_%d_%s_sample' % (start_year,end_year,aggregation_level)
                    if len(comtrade) <= max_rows:
                        save_with_warning(comtrade, data_path, save_as)
                    save_with_warning(comtrade.sample(n=max_rows), data_path, save_as)

                dataframes = []
                for start_row in range(0,len(comtrade),max_rows):
                    end_row = start_row + max_rows
                    if end_row < len(comtrade):
                        dataframes.append(comtrade[start_row:end_row])
                    else:
                        dataframes.append(comtrade[start_row:])

                for i, df in enumerate(dataframes):
                    save_as_iter = '%s_%d' % (save_as, i)
                    save_with_warning(df, data_path, save_as_iter)
                    print 'saved %s' % save_as_iter


def save(data, path, file_stub):
    if 'sample' in file_stub:
        data.to_csv('%s/%s.tsv' % (path,file_stub), index=None, sep='\t', encoding='utf-8')
        return
    for num, df in enumerate(data):
        df.to_csv('%s/%s_%d.tsv' % (path, file_stub, num), index=None, sep='\t', encoding='utf-8')

def save_with_warning(data, path, file_stub):
    for file in os.listdir(path):
        if file_stub in file:
            response = raw_input('Overwrite %s? y_train/n\n' % file_stub)
            if response == 'y_train':
                if 'sample' in file_stub:
                    save(data, path, file_stub)
            break
    save(data, path, file_stub)

if __name__ == "__main__":

    buildDatabase(aggregation_level ='4dg', start_year = 1996, end_year = 2015, postgres=True, csv=False, data_path='data')
