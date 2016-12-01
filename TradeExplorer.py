import collections
from os.path import isfile
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata
from sklearn.metrics.pairwise import cosine_similarity

def get_ctry_codes_df():
    with open('UN Comtrade Country List.csv', mode='rU') as f:
        ctry_codes = pd.read_csv(f, sep=',', header=0)
    ctry_codes = ctry_codes[['Country Code', 'Country Name English', 'ISO3-digit Alpha',
                                   'Start Valid Year', 'End Valid Year']]
    ctry_codes.columns = ['code', 'name', 'iso', 'start', 'end']
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

def get_ctries_dict():
    ctries = collections.OrderedDict()
    for row in get_ctry_codes_df().itertuples():
        code = row[1]
        name = row[2]
        iso = row[3]
        ctries[code] = {'name': name, 'iso': iso}
    return ctries

def get_cmds(aggregation_level,labels_as_keys=False):
    import json
    with open('classificationHS.json') as f:
        json = json.load(f)
    cmds = collections.OrderedDict()
    for row in json[u'results']:
        code = str(row['id'])
        text = row['text']
        if code.isdigit() and (len(code) <= aggregation_level and len(code) >=aggregation_level-1):
            cmds[code.zfill(aggregation_level)] = text[text.find(' - ') + 3:]  # remove numeric code from string
    if labels_as_keys:
        cmds = {y:x for x,y in cmds.iteritems()}
    return cmds


class Country(object):
    def __init__(self, code):
        self.code = code
        self.name = get_ctries_dict()[code]['name']
        self.iso = get_ctries_dict()[code]['iso']


class Commodity(object):
    def __init__(self, code):
        self.code = code
        self.name = get_cmds(len(code))[code]


def get_data(fname, path='data'):
    if fname[-3:] == 'csv':
        with open('%s/%s' % (path, fname), mode='rU') as f:
            df = pd.read_csv(f, sep=',', header=0)
    else:
        assert fname[-3:] == 'tsv'
        with open('%s/%s' % (path, fname), mode='rU') as f:
            df = pd.read_csv(f, sep='\t', header=0)
    cols = ['fromCode', 'toCode', 'yr', 'tradeValue', 'cmdCode']
    for col in cols:
        assert col in df.columns.values
    df = df[cols]
    df = df[~pd.isnull(df.tradeValue)]
    return df

def check_missing(df):
    try:
        assert not df.isnull().values.any()
    except AssertionError:
        print df[df.isnull().any(axis=1)].head()
        raise

def check_zeros(df,rows=True,columns=False):
    if rows:
        zero_rows = df[df.sum(axis=1) == 0]
        try:
            assert zero_rows.empty
        except AssertionError:
            print 'df contains all zero rows.'
            print zero_rows.head()
            raise
    if columns:
        zero_cols = df.ix[:, (df == 0).all()]
        try:
            assert zero_cols.empty
        except AssertionError:
            print 'df contains all zero columns.'
            print zero_cols.head()
            raise


class Exports(object):

    def __init__(self, fname, path='data'):
        assert isinstance(fname, str)
        assert '.' in fname # file name should include extension
        df = get_data(fname, path)
        self.origin = df.fromCode.rename('origin')
        self.destination = df.toCode.rename('destination')
        self.cmd = df.cmdCode.rename('cmd')
        self.value = df.tradeValue.rename('value')
        self.world_total = self.value.sum()
        self.yr = df.yr.rename('yr')
        self.raw = pd.concat([self.origin, self.destination, self.cmd, self.yr, self.value], axis=1)

    def by_origin(self, share = False):
        df = pd.concat([self.origin, self.yr, self.value], axis=1)
        df = df.groupby(['origin','yr']).sum().reset_index()
        if share:
            df.value = df.value / self.world_total
        return df

    def by_destination(self, share = False):
        df = pd.concat([self.destination, self.yr, self.value], axis=1)
        df = df.groupby(['destination','yr']).sum().reset_index()
        if share:
            df.value = df.value / self.world_total
        return df

    def by_origin_and_cmd(self, share=True, rca=False):
        df = pd.concat([self.origin, self.cmd, self.yr, self.value], axis=1)
        cols = df.columns.values
        df = df.groupby(['origin', 'cmd', 'yr']).sum().reset_index()
        if share or rca:
            df = df.merge(self.by_origin(), on='origin', suffixes=('', '_t'))
            df.value = df.value/df.value_t
            df = df[cols]
        if rca:
            df = df.merge(self.by_cmd(share=True), on='cmd', suffixes=('', '_t'))
            df.value = df.value/df.value_t
            df = df[cols]
        df = df.sort_values(['yr','origin','cmd'])
        assert not df.duplicated(['origin', 'cmd', 'yr']).values.any()
        return df

    def by_cmd(self, share = False):
        df = pd.concat([self.cmd, self.yr, self.value], axis=1)
        df = df.groupby(['cmd','yr']).sum().reset_index()
        if share:
            df.value = df.value / self.world_total
        return df

    def by_origin_and_destination(self, share=False, intensity=False):
        df = pd.concat([self.origin, self.destination, self.yr, self.value], axis=1)
        cols = df.columns.values
        df = df.groupby(['origin', 'destination', 'yr']).sum().reset_index()
        if share:
            # share of origin's exports that go to destination
            assert not intensity
            df = df.merge(self.by_origin(), on='origin', suffixes=('', '_origin_total'))
            df.value = df.value/df.value_origin_total
        if intensity:
            # Intensity_ij = Intensity_ji = (Eij + Eji)/(Ei + Ej)
            assert not share
            df = df.merge(self.by_origin(), on='origin', suffixes=('', '_origin_total'))
            df = df.merge(self.by_origin(), left_on='destination', right_on='origin',
                          suffixes=('', '_destination_total'))
            df_copy = df
            df = df.merge(df_copy, left_on=['origin', 'destination'],
                                   right_on=['destination', 'origin'], suffixes=('', '_imports'))
            df.value = (df.value + df.value_imports) / (df.value_origin_total + df.value_destination_total)
        df = df[cols]
        df = df.sort_values(['yr','origin','destination'])
        return df


class ProductSpace(object):

    def __init__(self, yr, fname, path='data'):
        '''
        :param yr: year as integer
        :param fname: dataset filename including extension
        :param path: location of filename
        :param country: country code as integer. default is none (all countries)
        '''
        assert isinstance(yr, int)
        assert isinstance(fname, str)
        assert '.' in fname # file name should include extension
        self.yr = yr
        self.fname = fname
        self.path = path

        df = Exports(fname, path).by_origin_and_cmd(rca=True)
        df = df[df.yr==yr]
        df = df[df.value!=0]
        self.matrix = df.pivot(index='origin',columns='cmd',values='value').fillna(0)
        codes = self.matrix.columns.values
        self.aggregation_level = max([len(str(code)) for code in codes])
        codes = [str(code).zfill(self.aggregation_level) for code in codes]
        self.labels = {i: get_cmds(self.aggregation_level)[code] for i, code in enumerate(codes)}
        self.rca_correlation_matrix = np.corrcoef(self.matrix, rowvar=0)

    def show_graph(self, country=None, cut_off=0.2):
        rows, cols = np.where(self.rca_correlation_matrix > cut_off)
        edges = zip(rows.tolist(), cols.tolist())
        graph = nx.Graph()
        graph.add_edges_from(edges)
        pos = nx.fruchterman_reingold_layout(graph)
        plt.figure(figsize=(24, 12))
        if country is not None:
            country_name = get_ctries_dict()[str(country)]['name']
            plt.title('Product space for %s in %d, HS %d-digit classification' % (country_name, self.yr, self.aggregation_level))
            country_rca = self.matrix.loc[country]
            scaler = country_rca.sum()/50000
            nodes_show = [i for i,rca in enumerate(country_rca) if rca>1]
            nodes_grey_out = [i for i,rca in enumerate(country_rca) if rca<=1]
            node_size = [rca/scaler for i,rca in enumerate(country_rca) if rca>1]
            labels = {}
            for key,value in self.labels.iteritems():
                if key in nodes_show:
                    labels[key] = value

            nx.draw_networkx_nodes(graph, pos,
                                   nodelist=nodes_grey_out,
                                   node_color='grey',
                                   node_size=50,
                                   alpha=0.2)
            nx.draw_networkx_nodes(graph, pos,
                                   nodelist=nodes_show,
                                   node_color='r',
                                   node_size=node_size,
                                   alpha=0.6)
            nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
            nx.draw_networkx_labels(graph, pos, labels, font_size=12)
        else:
            plt.title('Product space in %d, HS %d-digit classification' % (self.yr, self.aggregation_level))
            nx.draw(graph, pos, labels=self.labels, with_labels=True)
        plt.show()

class CountrySpace(object):

    def __init__(self, yr, fname, path='data', feature='rca'):
        '''
        :param yr: year as integer
        :param fname: dataset filename including extension
        :param path: location of filename
        '''
        assert isinstance(yr, int)
        assert isinstance(fname, str)
        assert '.' in fname  # file name should include extension
        self.yr = yr
        self.fname = fname
        self.path = path
        self.countries = ProductSpace(yr, fname, path).matrix.index.values
        self.commodities = ProductSpace(yr, fname, path).matrix.columns.values
        self.ctry_names = pd.Series([get_ctries_dict()[str(ctry)]['name'] for ctry in self.countries])
        self.labels = {i: get_ctries_dict()[str(ctry)]['name'] for i, ctry in enumerate(self.countries)}
        assert feature in ['rca', 'export_destination', 'intensity']
        self.feature = feature
        if feature == 'rca':
            df = Exports(fname, path).by_origin_and_cmd(rca=True)
            df = df[df.yr == yr]
            self.feature_matrix = df.pivot(index='origin', columns='cmd', values='value').fillna(0)
            self.feature_matrix = self.feature_matrix.reindex(index=self.countries, columns=self.commodities, fill_value=0)
            check_zeros(self.feature_matrix)
            check_missing(self.feature_matrix)
            self.correlation_matrix = np.corrcoef(self.feature_matrix.as_matrix(), rowvar=0)
        if feature == 'export_destination':
            df = Exports(fname, path).by_origin_and_destination(share=True)
            df = df[df.yr == yr]
            self.feature_matrix = df.pivot(index='origin', columns='destination', values='value').fillna(0)
            self.feature_matrix = self.feature_matrix.reindex(index=self.countries, columns=self.countries, fill_value=0)
            check_zeros(self.feature_matrix)
            check_missing(self.feature_matrix)
            self.correlation_matrix = np.corrcoef(self.feature_matrix.as_matrix(), rowvar=0)
        if feature == 'intensity':
            df = Exports(fname, path).by_origin_and_destination(intensity=True)
            df = df[df.yr == yr]
            self.feature_matrix = df.pivot(index='origin', columns='destination', values='value').fillna(0)
            size = self.feature_matrix.shape[0]
            assert size == self.feature_matrix.shape[1] # should be square
            self.feature_matrix.values[[np.arange(size)] * 2] = 0 # fills diagonal with zeros (intensity with self)
            self.feature_matrix = self.feature_matrix.reindex(index=self.countries, columns=self.countries, fill_value=0)
            check_missing(self.feature_matrix)
            self.correlation_matrix = np.corrcoef(self.feature_matrix.as_matrix(), rowvar=0)

    def distance(self, metric='euclidean'):
        distance_matrix = squareform(pdist(self.feature_matrix.as_matrix(), metric))
        distance_matrix = pd.DataFrame(data=distance_matrix, index=self.countries, columns=self.countries)
        return distance_matrix

    def similarity(self):
        features = self.feature_matrix
        similarity = cosine_similarity(features.as_matrix())
        np.fill_diagonal(similarity, 0)
        similarity = pd.DataFrame(data=similarity,index=self.countries, columns=self.countries)
        check_missing(similarity)
        return similarity

    def neighbours(self, k):
        # returns {country:[k nearest neighbours]}
        if self.feature == 'intensity':
            similarity = self.feature_matrix.as_matrix()
        else:
            similarity = self.similarity().as_matrix()
        neighbours = {}
        for country, sim_array in zip(self.countries,similarity):
            neighbours[country] = [self.countries[np.argsort(-sim_array)[:k]]]
        return neighbours

    def rca_neighbours(self, k):
        # returns the rca rank across commodities of k nearest neighbours for each country
        # index = ['origin', 'cmd']. columns = rca_neighbour_1 to rca_neighbour_k
        neighbours = pd.DataFrame.from_dict(self.neighbours(k), orient='index')
        neighbours.columns = ['neighbours']
        rca = ProductSpace(self.yr, self.fname, self.path).matrix
        cmds = rca.columns
        ctries = rca.index
        for ctry in ctries:
            rca.loc[ctry] = rankdata(rca.loc[ctry])
        rca = pd.DataFrame(MinMaxScaler().fit_transform(rca.T).T, index=ctries,columns=cmds)
        for i in range(k):
            rca.columns = cmds
            tups = zip(*[['%s_neighbour_%d' % (self.feature, i+1)] * rca.shape[1], rca.columns])
            rca.columns = pd.MultiIndex.from_tuples(tups, names=['neighbour', 'cmd'])
            neighbours['neighbour'] = neighbours.neighbours.apply(lambda x: x[i])
            neighbours = neighbours.merge(rca, left_on='neighbour', right_index=True)
            neighbours = neighbours.drop('neighbour', axis=1)
        neighbours = neighbours.drop('neighbours', axis=1)
        neighbours.columns = pd.MultiIndex.from_tuples(neighbours.columns, names=['', 'cmd'])
        neighbours = neighbours.stack()
        neighbours.index.rename(names=['origin', 'cmd'], inplace=True)
        neighbours.sortlevel(level=0, axis=0, inplace=True)
        neighbours = neighbours.reindex(index=neighbours.index, method='ffill')
        return neighbours

    def get_distance(self,country_1,country_2,metric='euclidean'):
        return self.distance(metric).loc[country_1,country_2]

    def get_closest(self, country, n=5,metric='euclidean'):
        s = self.distance(metric)[country]
        s = s.sort_values()
        s = s.rename('Countries closest to %s based on %s' % (get_ctries_dict()[str(country)]['name'], self.feature))
        s.index = s.index.map(lambda x: get_ctries_dict()[str(x)]['name'])
        return s[1:n+1]

    def similarity_rank(self,n=30,descending=True):
        similarity = self.similarity().stack()
        similarity = similarity[similarity != 0]  # remove pairs with self
        similarity = similarity.sort_values()
        if descending:
            similarity = similarity.sort_values(ascending=False)
        similarity = similarity.head(n)
        similarity.MultiIndex = similarity.index
        similarity.index = similarity.MultiIndex.map(lambda x: (get_ctries_dict()[str(x[0])]['name'], get_ctries_dict()[str(x[1])]['name']))
        return similarity

    def show_graph(self, country=None, cut_off=0.2):
        graph = nx.Graph()
        node_list = self.labels.keys()
        graph.add_nodes_from(node_list)
        rows, cols = np.where(self.correlation_matrix > cut_off)
        edges = zip(rows.tolist(), cols.tolist())
        graph.add_edges_from(edges)
        pos = nx.fruchterman_reingold_layout(graph)
        plt.figure(figsize=(24, 12))
        plt.title('Country space in %d based on %s' % (self.yr, self.feature))
        nx.draw(graph, pos, labels=self.labels, with_labels=True)
        plt.show()

class CmdCtryMatrix(object):

    def __init__(self, yr, fname, path='data'):
        '''
        :param yr: year as integer
        :param fname: dataset filename including extension
        :param path: location of filename
        '''
        assert isinstance(yr, int)
        assert isinstance(fname, str)
        assert '.' in fname  # file name should include extension
        self.yr = yr
        self.fname = fname
        self.path = path
        self.RCA = ProductSpace(yr, fname, path).matrix.transpose()
        self.commodities = self.RCA.index.values
        self.countries = self.RCA.columns.values
        self.has_export = (self.RCA > 0).applymap(lambda x: 1 if x else 0)

    def market_share(self,rows='commodities'):
        assert rows in ['commodities', 'countries']
        market_share = Exports(self.fname,self.path).by_origin_and_cmd(share=False)
        market_share = market_share[market_share.yr==self.yr].pivot(index='cmd',columns='origin',values='value').fillna(0)
        cmd_totals = Exports(self.fname,self.path).by_cmd(share=False)
        cmd_totals = cmd_totals[cmd_totals.yr==self.yr].drop('yr',axis=1)
        cmd_totals = pd.concat([cmd_totals.value]*len(self.countries),axis=1)
        cmd_totals.columns = self.countries
        cmd_totals.index = self.commodities
        market_share = market_share/cmd_totals
        assert market_share.sum(axis=1).all() == 1
        if rows == 'countries':
            market_share = market_share.transpose()
        return market_share

class SingleYearDataset(object):

    def __init__(self, yr, fname, path='data', k=5):
        '''
        :param yr: year as integer
        :param fname: dataset filename including extension
        :param path: location of filename
        '''
        assert isinstance(yr, int)
        assert isinstance(fname, str)
        assert '.' in fname  # file name should include extension
        self.yr = yr
        self.fname = fname
        self.path = path
        self.k = k
        self.matrices = CmdCtryMatrix(self.yr, self.fname, self.path)
        self.has_export = self.matrices.has_export.stack().sort_index(level=['origin','cmd'])
        self.index = self.has_export.index
        self.similarity_features = ['rca','export_destination','intensity']
        self.m = len(set(self.has_export.index.get_level_values('cmd'))) # number of commodities
        self.n = len(set(self.has_export.index.get_level_values('origin'))) # number of countries
        self.observations = self.m*self.n # number of commodities x number of countries
        assert self.has_export.shape[0] == self.observations


    def join_features(self):
        df = self.has_export
        df = df.reorder_levels(['origin', 'cmd'])
        for similarity_feature in self.similarity_features:
            neighbours = CountrySpace(yr=self.yr, fname=self.fname, path=self.path, feature=similarity_feature).rca_neighbours(self.k)
            assert neighbours.shape[0] == self.observations
            df = pd.concat([df,neighbours], axis=1, join_axes=[df.index])
        df = df.rename(columns={0:'has_export'})
        df['yr'] = self.yr
        try:
            assert df.shape[0] == self.observations
        except AssertionError:
            print 'dataframe has %d rows, expected %d' % (df.shape[0], self.observations)
            raise
        check_missing(df)
        return df

    def save(self,fname,save_path='data'):
        df = self.join_features()
        print 'saving %s, rows: %d' % (fname,len(df))
        df.to_csv('%s/%s' % (save_path,fname), sep=',')


class FinalDataset(object):

    def __init__(self, yr, base_yr, path='data'):
        '''
        :param yr: year to train PredictiveModels. # TODO: extend to multiple years
        :param base_yr: filters dataset  - only commodities that are not exported in base_yr are kept
        :param path: location of data files
        '''
        assert isinstance(yr, int)
        assert isinstance(base_yr, int)
        self.yr = yr
        self.path = path
        with open('%s/%s' % (path, 'mldataset_%d_4dg.csv' % base_yr), mode='rU') as f:
            df = pd.read_csv(f, sep=',', header=0, usecols=['cmd','origin','has_export'])
        df.index = [df.cmd, df.origin]
        self.has_export_in_base_year = df

    def build(self):
        with open('%s/%s' % (self.path, 'mldataset_%d_4dg.csv' % self.yr), mode='rU') as f:
            iter_csv = pd.read_csv(f, sep=',', header=0, iterator=True, chunksize=50000)
            df = pd.concat(iter_csv, ignore_index=True)
        df.index = [df.cmd, df.origin]
        df = df.merge(self.has_export_in_base_year,on=['cmd','origin'],suffixes=('', '_base'))
        df = df[df.has_export_base == 0]
        df = df.drop('has_export_base',axis=1)
        return df

    def save(self,save_as):
        df = self.build()
        print 'saving %s, rows: %d' % (save_as,len(df))
        df.to_csv('%s/%s.csv' % (self.path,save_as), sep=',')
        sample = df.sample(frac=0.05, random_state=1)
        print 'saving %s, rows: %d' % ('sample_'+save_as,len(sample))
        sample.to_csv('%s.csv' % ('sample_'+save_as), sep=',')


def save_final_dataset(yr, base_yr, k):
    SingleYearDataset(base_yr, 'comtrade_%d_4dg.tsv' % base_yr, k=k).save('mldataset_%d_4dg.csv' % base_yr)
    SingleYearDataset(yr, 'comtrade_%d_4dg.tsv' % yr, k=k).save('mldataset_%d_4dg.csv' % yr)
    FinalDataset(yr,base_yr).save('mldataset_4dg')

if __name__ == "__main__":
    save_final_dataset(2015, 2011, 200)

