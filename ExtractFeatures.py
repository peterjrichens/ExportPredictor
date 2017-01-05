import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric
from math import radians
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from CreateDatabase import clean_df_db_dups
from config import SQLALCHEMY_DATABASE_URI
from sqlalchemy import create_engine
engine = create_engine(SQLALCHEMY_DATABASE_URI)

db_yrs = pd.read_sql('select distinct yr from comtrade', engine).values
DB_START_YR = min(db_yrs)[0]
DB_END_YR = max(db_yrs)[0]
DB_YR_RANGE = range(DB_START_YR, DB_END_YR+1)

# definition of new export:
NB_YRS_NOT_EXPORTED = 3 # number of consecutive years a product must not be exported to be considered a potential new export
NB_YRS_EXPORTED = 3 # number of consecutive years a potential export must be exported to be considered a new export
NEW_EXPORT_START_YR = DB_START_YR + NB_YRS_NOT_EXPORTED # cannot be classified as new export before this year
NEW_EXPORT_END_YR = DB_END_YR - NB_YRS_EXPORTED + 1 # cannot be classified as new export after this date
NEW_EXPORT_YR_RANGE = range(NEW_EXPORT_START_YR, NEW_EXPORT_END_YR+1)

FEATURES = ['rca', 'imports', 'export_destination', 'import_origin', 'intensity', 'distance']

'''
The TARGET is defined at the country-product-year level: whether country i starts exporting product p at time t. To
minimise marginal exports and noise in the data, a country is considered to export a product only after three
consecutive years.

FEATURES are computed as follows:
    - Construct a matrix with rows for each potential new export (one row for country i, product p
      at time t). Column j of the matrix represents the j'th nearest neighbour of country i based on
      the particular distance metric. The cell values are j's comparative advantage in product p (RCA rank).
    - Use Linear Discriminant Analysis to reduce this matrix to vector. The comparative advantage matrix of
      i's neighbours is projected on to the most discriminative direction with respect to i's new
      exports in the previous time period (lagged y).
    - This procedure is repeated for each distance/similarity metric, which are what the feature names refer to:

        rca:                The cosine of the angle between the revealed comparative advantage product vectors of
                            country i and country j. Rationale: aligned RCA vectors suggests two countries have similar
                            underlying productive capabilities.

        imports:            The cosine of the angle between the import product vectors of country i and
                            country j. Rationale: aligned import vectors suggest similar productive capabilities
                            or position in global supply chains.

        export_destination: The cosine of the angle between the export destination vectors of country i and
                            country j. Rationale: aligned export destination vectors suggest countries have access to
                            similar markets.

        import_origin:      The cosine of the angle between the import origin vectors of country i and
                            country j. Rationale: aligned import origin vectors suggest countries occupy similar
                            positions in global supply chains.

        intensity:          The bilateral trade intensity between country i and country j, a proxy for economic
                            integration. Rationale: countries with close trade relationships may share other
                            characteristics and are more likely to learn from each other.

        distance:           The geographic distance between country i and country j. Rationale: neighbouring countries
                            may share other characteristics and are more likely to learn from each other.

    - The final two features are aggregated lagged dependent variables:

        average_origin:     The average discovery rate across products for country i in the previous time period.

        average_cmd:        The average discovery rate across countries for product p in the previous time period.

'''
def has_export(yr_range=DB_YR_RANGE):
    # returns dataframe. rows: origin-commodity, columns: years
    query = '''select countries.code as origin, commodities.code as cmd
                from countries cross join commodities
                order by origin, cmd
            '''
    has_export = pd.read_sql(query, engine)
    for yr in yr_range:
        query = '''select origin, cmd, 1 as has_export_%s
                   from comtrade
                   where yr = %d
                   group by origin, cmd
                    ''' % (yr, yr)
        df = pd.read_sql(query, engine)
        df['has_export_%s'%yr] = df['has_export_%s'%yr].apply(lambda x: int(x))
        has_export = has_export.merge(df, how='left', on=['origin', 'cmd']).fillna(0)
    for col in [col for col in has_export.columns if 'has_export' in col]:
        has_export[col] = has_export[col].apply(lambda x: int(x))
    return has_export

def has_rca(yr_range=DB_YR_RANGE):
    # returns dataframe. rows: origin-commodity, columns: years
    # Country i is said to 'export' product p if RCA > 1 or: X_i,p / X_i > X_world,p / X_world
    query = '''select countries.code as origin, commodities.code as cmd
                from countries cross join commodities
                order by origin, cmd
            '''
    has_export = pd.read_sql(query, engine)
    print "finding 'exports'..."
    for yr in tqdm(yr_range):
        query = '''
                            select origin, cmd, 1 as has_export_%s
                            from
                            (select origin, cmd,
                            ((0.0+value)/sum(value) over w1)/((0.0+sum(value) over w2)/sum(value) over ()) as rca
                            from (
                              select origin, cmd, sum(value) as value
                              from comtrade
                              where yr = %d
                              group by origin, cmd
                            ) as sum_over_destination

                            window w1 as (
                                partition by origin
                            ), w2 as (
                                partition by cmd
                            )
                            order by origin, cmd) as t
                            where rca >= 1
                            ''' % (yr, yr)
        df = pd.read_sql(query, engine)
        df['has_export_%s'%yr] = df['has_export_%s'%yr].apply(lambda x: int(x))
        has_export = has_export.merge(df, how='left', on=['origin', 'cmd']).fillna(0)
    for col in [col for col in has_export.columns if 'has_export' in col]:
        has_export[col] = has_export[col].apply(lambda x: int(x))
    return has_export

def get_new_exports(has_export_df):
    '''
    Loops through has_export_df looking for origin-cmd combinations not exported in for 3 (by default) consecutive years, then checks
    if they are exported for 3 (by default) consecutive years. Returns df with index: (origin, cmd, year), values:
    0: cmd was not exported by origin for at least 3 consecutive years and was not exported for 3 consecutive years starting in year
    1: cmd was not exported by origin for at least 3 consecutive years and was exported for 3 consecutive years starting in year
    missing: cmd already exported for at least 3 consecutive years
    '''
    has_export_columns = [col for col in has_export_df.columns if 'has_export' in col]
    new_exports = {}
    print "finding 'new exports'..."
    for _, row in tqdm(has_export_df.iterrows()):
        origin = row['origin']
        cmd = row['cmd']
        check_forward = False
        no_exports_counter = 0
        exports_counter = 0
        for t, col in enumerate(has_export_columns):
            if np.isnan(row[col]):
                break # do not consider countries with some years missing (eg. Montenegro, South Sudan)
            has_export = (row[col] == 1)
            yr = DB_START_YR+t
            if check_forward:
                if has_export:
                    exports_counter += 1
                else: # new export not maintained, reset counter and continue to check forward
                    exports_counter = 0
                    continue
                if exports_counter < NB_YRS_EXPORTED:
                    if yr == NEW_EXPORT_END_YR: # cannot be exported for 3 consecutive years
                        new_exports[(origin, cmd)] = {'new_export_%s' % yr: 0 for yr in NEW_EXPORT_YR_RANGE}
                    continue
                else: # exported for 3 consecutive years
                    export_run_start = yr-exports_counter+1
                    new_exports[(origin, cmd)] = {'new_export_%s' % yr: 0 for yr in range(NEW_EXPORT_START_YR, export_run_start)}
                    new_exports[(origin, cmd)]['new_export_%s' % export_run_start] = 1
                    break # start next origin-cmd

            if no_exports_counter < NB_YRS_NOT_EXPORTED:
                if has_export:
                    exports_counter += 1
                    no_exports_counter = 0
                    if exports_counter < NB_YRS_EXPORTED:
                        continue
                    else:
                        break # already exported for 3 consecutive years: ignore and start next origin-cmd
                else: # not exported
                    exports_counter = 0
                    no_exports_counter += 1
                    continue
            else: # not exported for at least 3 consecutive years
                if not has_export:
                    if yr == NEW_EXPORT_END_YR: # cannot be exported for 3 consecutive years
                        new_exports[(origin, cmd)] = {'new_export_%s' % yr: 0 for yr in NEW_EXPORT_YR_RANGE}
                        break
                    continue
                else: # found a new export, now check next two years
                    exports_counter = 1
                    check_forward = True
    df = pd.DataFrame.from_dict(new_exports, orient='index')
    ordered_cols = ['new_export_%s' % yr for yr in NEW_EXPORT_YR_RANGE]
    df = df[ordered_cols]
    tuples = list(zip(*[['year']*len(NEW_EXPORT_YR_RANGE), NEW_EXPORT_YR_RANGE]))
    df.columns = pd.MultiIndex.from_tuples(tuples)
    df = df.stack()
    df.columns = ['new_export']
    df.index.rename(['origin', 'cmd', 'year'], inplace=True)
    df.reset_index(inplace=True)
    df['new_export'] = df.new_export.apply(lambda x: int(x))
    return df

def origin_cmd_average(new_exports, origin_cmd):
    # returns the average export discovery rate by year, either:
    #   across commoidities for particular countries: if origin_cmd = 'origin'
    #  or across countries or particular commodities: if origin_cmd = 'cmd'
    if origin_cmd == 'origin':
        average = new_exports.groupby(['origin', 'year']).mean()
        average.drop('cmd', axis=1, inplace=True)
    if origin_cmd == 'cmd':
        average = new_exports.groupby(['cmd', 'year']).mean()
        average.drop('origin', axis=1, inplace=True)
    average.rename(columns = {'new_export': '%s_average' % origin_cmd}, inplace=True)
    return average


class FeaturesByYr(object):
    # class of feature extraction methods parameterised by start and end year

    def __init__(self, start_yr, end_yr=None):
        assert isinstance(start_yr, int)
        self.start_yr = start_yr
        if end_yr is not None:
            assert isinstance(end_yr, int)
            self.end_yr = end_yr
        else:
            self.end_yr = start_yr


    def rca(self):
        # returns revealed comparative advanage at origin-commodity level
        query = '''
                    select origin, cmd,
                    ((0.0+value)/sum(value) over w1)/((0.0+sum(value) over w2)/sum(value) over ()) as rca
                    from (
                      select origin, cmd, sum(value) as value
                      from comtrade
                      where (yr >= %d and yr <= %d)
                      group by origin, cmd
                    ) as sum_over_destination

                    window w1 as (
                        partition by origin
                    ), w2 as (
                        partition by cmd
                    )
                    order by origin, cmd
                    ''' % (self.start_yr, self.end_yr)
        df = pd.read_sql(query, engine)
        return df

    def export_destination(self):
        # returns export destination shares at origin-destination level
        query = '''
                    select origin, destination,
                    (0.0+value)/sum(value) over (partition by origin) as export_destination
                    from (
                      select origin, destination, sum(value) as value
                      from comtrade
                      where (yr >= %d and yr <= %d)
                      group by origin, destination
                    ) as sum_over_cmd

                    order by origin, destination
                    ''' % (self.start_yr, self.end_yr)
        df = pd.read_sql(query, engine)
        return df

    def imports(self):
        # returns import product shares at origin-commodity level
        query = '''
                    select coalesce(t1.destination, t2.origin) as origin,
                    cmd,
                    (0.0+value)/sum(value) over (partition by origin) as imports
                    from (
                      select destination, cmd, sum(value) as value
                      from comtrade
                      where (yr >= %d and yr <= %d)
                      group by destination, cmd
                    ) t1
                     full outer join (
                      select distinct origin
                      from comtrade
                      where (yr >= %d and yr <= %d)
                    ) t2
                    on
                      (t1.destination = t2.origin)

                    order by origin, cmd
                    ''' % (self.start_yr, self.end_yr, self.start_yr, self.end_yr)
        df = pd.read_sql(query, engine)
        return df

    def import_origin(self):
        # returns import origin shares at country-partner level
        query = '''
                    select coalesce(t1.destination, t2.origin) as origin,
                    t1.origin as destination,
                    (0.0+value)/sum(value) over (partition by t1.destination) as import_origin
                    from (
                      select destination, origin, sum(value) as value
                      from comtrade
                      where (yr >= %d and yr <= %d)
                      group by destination, origin
                    ) as t1
                    full outer join (
                      select distinct origin
                      from comtrade
                      where (yr >= %d and yr <= %d)
                    ) t2
                    on
                      (t1.destination = t2.origin)

                    order by origin, destination
                    ''' % (self.start_yr, self.end_yr, self.start_yr, self.end_yr)
        df = pd.read_sql(query, engine)
        return df

    def intensity(self):
        # returns trade intensity at country-partner level
        query = '''
                    select coalesce(t1.origin, t2.destination) as origin,
                    coalesce(t1.destination, t2.origin) as destination,
                    (coalesce(t1.value, 0.0) + coalesce(t2.value, 0.0))/
                    (coalesce(sum(t1.value) over (partition by t1.origin), 0.0) +
                    coalesce(sum(t2.value) over (partition by t1.destination), 0.0)) as intensity
                    from (
                      select origin, destination, sum(value) as value
                      from comtrade
                      where (yr >= %d and yr <= %d)
                      group by origin, destination
                    ) t1
                    full outer join (
                      select origin, destination, sum(value) as value
                      from comtrade
                      where (yr >= %d and yr <= %d)
                      group by origin, destination
                    ) t2
                    on
                      (t1.origin = t2.destination and t1.destination = t2.origin)
                    order by origin, destination
                    ''' % (self.start_yr, self.end_yr, self.start_yr, self.end_yr)
        df = pd.read_sql(query, engine)
        return df


    def adjacency_matrix(self, feature):
        # converts long feature dataframes to country adjacency matrices
        assert feature in FEATURES
        if feature == 'distance':
            query = '''
                     select t1.origin, latitude, longitude
                     from
                     (select distinct origin from comtrade where (yr >= %d and yr <= %d)) as t1, countries
                     where t1.origin=countries.code
                    ''' % (self.start_yr, self.end_yr)
            df = pd.read_sql(query, engine)
            df['latitude'] = df.latitude.map(lambda x: radians(x))
            df['longitude'] = df.longitude.map(lambda x: radians(x))
            df.set_index('origin',inplace=True)
            destination = df.index.rename('destination')
            dist = DistanceMetric.get_metric('haversine')
            return pd.DataFrame(dist.pairwise(df[['latitude','longitude']])*6378.1,index=df.index, columns=destination).fillna(40000)
        if feature == 'intensity':
            return self.intensity().pivot(index='origin', columns='destination', values='intensity').fillna(0)
        if feature == 'rca':
            matrix = self.rca().pivot(index='origin', columns='cmd', values='rca').fillna(0)
        if feature == 'export_destination':
            matrix = self.export_destination().pivot(index='origin', columns='destination',
                                                                 values='export_destination').fillna(0)
        if feature == 'imports':
            matrix = self.imports().pivot(index='origin', columns='cmd',
                                                                 values='imports').fillna(0)
        if feature == 'import_origin':
            matrix = self.import_origin().pivot(index='origin', columns='destination',
                                                                 values='import_origin').fillna(0)

        origin = matrix.index
        destination = origin.rename('destination')
        matrix = cosine_similarity(matrix.as_matrix())
        np.fill_diagonal(matrix, 0)
        return pd.DataFrame(matrix, index=origin, columns=destination)


    def neighbour_dict(self, feature):
        # returns {country:[neighbours sorted by feature]}
        assert feature in FEATURES
        similarity = self.adjacency_matrix(feature)
        countries = similarity.index.values
        neighbours = {}
        for country, sim_array in zip(countries, similarity.as_matrix()):
            neighbours[country] = [countries[np.argsort(-sim_array)]]
        return neighbours


    def rca_neighbours(self, feature):
        # returns the rca rank across commodities of sorted feature-neighbours for each country
        # index = ['origin', 'cmd']. columns = rca of neighbour_1 to rca of neighbour_N,
        # values = rca rank of origin's neighbour for cmd
        neighbours = pd.DataFrame.from_dict(self.neighbour_dict(feature), orient='index')
        neighbours.columns = ['neighbours']
        rca = self.rca().pivot(index='origin', columns='cmd', values='rca').fillna(0)
        cmds = rca.columns
        ctries = rca.index
        for ctry in ctries:
            rca.loc[ctry] = rankdata(rca.loc[ctry])
        rca = pd.DataFrame(MinMaxScaler().fit_transform(rca.T).T, index=ctries, columns=cmds)
        for i in range(len(ctries)):
            rca.columns = cmds
            tups = zip(*[['%s_neighbour_%d' % (feature, i + 1)] * rca.shape[1], cmds])
            rca.columns = pd.MultiIndex.from_tuples(tups, names=['neighbour', 'cmd'])
            neighbours['neighbour'] = neighbours.neighbours.apply(lambda x: x[i])
            neighbours = neighbours.merge(rca, how='left', left_on='neighbour', right_index=True)
            neighbours = neighbours.drop('neighbour', axis=1)
        neighbours = neighbours.drop('neighbours', axis=1)
        neighbours.columns = pd.MultiIndex.from_tuples(neighbours.columns, names=['', 'cmd'])
        neighbours = neighbours.stack()
        neighbours = neighbours.fillna(0)
        neighbours.index.rename(names=['origin', 'cmd'], inplace=True)
        neighbours.sortlevel(level=0, axis=0, inplace=True)
        neighbours = neighbours.reindex(index=neighbours.index, method='ffill')
        return neighbours

def build_ml_db(new_exports, db_name):
    # builds postgres database containing final machine learning dataset
    origin_average = origin_cmd_average(new_exports, 'origin')
    cmd_average = origin_cmd_average(new_exports, 'cmd')

    query_database = pd.read_sql('select count(*) as nb_rows, max(year) as last_yr from %s' % db_name, engine)
    nb_rows = query_database.nb_rows.values[0]
    last_yr = query_database.last_yr.values[0]
    if nb_rows == 0:
        start = NEW_EXPORT_START_YR
    else:
        start = last_yr + 1
    for yr in range(start, NEW_EXPORT_END_YR+1):
        train_by_yr = new_exports[new_exports.year == yr]
        for feature in FEATURES:
            print 'getting neighbours based on %s for %d to %d' % (feature, yr-1, yr+1)
            # Features offset by one year lag (actual target cannot be used for LDA)
            # eg. new_exports in 2013-2015 (yr,yr+1,yr+2) are explained by features in 2012-2014 (yr-1 to yr+1)
            rca_neighbours = FeaturesByYr(yr - 1, yr + 1).rca_neighbours('%s' % feature)
            train_by_yr = train_by_yr.merge(rca_neighbours, how='left', left_on=['origin', 'cmd'], right_index=True)
            train_by_yr.dropna(axis=0, how='all', subset=[col for col in train_by_yr.columns if 'neighbour' in col], inplace=True) # cmds with no exports by any country
            train_by_yr = lda(train_by_yr)
            train_by_yr.rename(columns={0: '%s' %feature}, inplace=True)
        train_by_yr = merge_origin_cmd_averages(train_by_yr, origin_average, cmd_average)
        train_by_yr['year'] = train_by_yr['year'] + 2 # name period (yr,yr+1,yr+2) as last year not first year
        train_by_yr = clean_df_db_dups(train_by_yr, db_name, engine, dup_cols=['origin', 'cmd', 'year'])
        if len(train_by_yr) > 0:
            print 'appending training data for %d to %d to database' % (yr - 1, yr + 1)
            train_by_yr.to_sql(name=db_name, if_exists='append', con=engine, index=False)

    # generate X_test set for prediction
    test = new_exports[new_exports.year == NEW_EXPORT_END_YR]
    for feature in FEATURES:
        print 'getting neighbours based on %s for %d to %d' % (feature, NEW_EXPORT_END_YR, DB_END_YR)
        rca_neighbours = FeaturesByYr(NEW_EXPORT_END_YR, DB_END_YR).rca_neighbours('%s' % feature)
        test = test.merge(rca_neighbours, how='left', left_on=['origin', 'cmd'], right_index=True)
        test.dropna(axis=0, how='all', subset=[col for col in test.columns if 'neighbour' in col], inplace=True) # cmds with no exports by any country
        test = lda(test)
        test.rename(columns={0: '%s' %feature}, inplace=True)
    test = test[test.new_export == 0] # restrict to origin-cmd pairs never exported for 3 consecutive years
    test['year'] = test['year'] + 1 # want to generate predictions for next period
    test = merge_origin_cmd_averages(test, origin_average, cmd_average)
    test['year'] = test['year'] + 2 # name period (yr,yr+1,yr+2) as last year not first year
    test = clean_df_db_dups(test, db_name, engine, dup_cols=['origin', 'cmd', 'year'])
    test = test.drop('new_export', axis=1)
    if len(test) > 0:
        print 'appending X_test data for %d to %d to database' % (NEW_EXPORT_END_YR, DB_END_YR)
        test.to_sql(name=db_name, if_exists='append', con=engine, index=False)

def lda(df, target_col='new_export'):
    # reduces the dimensionality of (rca_neighbour_1 .. rca_neighbour_N) by projecting it to the most discriminative direction
    target = df[target_col]
    features_to_reduce = df[[col for col in df.columns if 'neighbour' in col]]
    other_features = df[[col for col in df.columns if 'neighbour' not in col and col != target_col]]
    lda = LinearDiscriminantAnalysis()
    lda.fit(features_to_reduce, df[target_col])
    component = pd.DataFrame(data=lda.transform(features_to_reduce), index=features_to_reduce.index)
    return pd.concat([target, other_features, component], axis=1)

def merge_origin_cmd_averages(df, origin_average, cmd_average):
    # merges df with lagged export discovery rate averaged at ctry and cmd level
    for col in ['origin', 'cmd', 'year']:
        df[col] = df[col].apply(lambda x: int(x))
    df['year'] = df.year - 1 # merge on previous year to get laggged value of new_export averaged across ctries/cmds
    df = df.join(origin_average, how='left', on=['origin','year'],)
    df['origin_average'].fillna(0, inplace=True)
    df = df.join(cmd_average, how='left', on=['cmd','year'])
    df['cmd_average'].fillna(0, inplace=True)
    df['year'] = df.year + 1
    return df


if __name__ == "__main__":

    has_export = has_export()
    new_exports = get_new_exports(has_export)
    build_ml_db(new_exports, 'mldataset')

    has_rca = has_rca()
    new_rca = get_new_exports(has_rca)
    build_ml_db(new_rca, 'mldataset2')


