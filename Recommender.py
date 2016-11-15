import pandas as pd
import graphlab
import WrangleData
import sys

train = WrangleData.ctry_cmd_shares('comtrade_2010_2dg.tsv', yr = 2010)
train = train[train.exportShare>0]
train = graphlab.SFrame(train)

# Train Models
factor = graphlab.recommender.factorization_recommender.create(
    train, user_id='fromCode', item_id='cmdCode',target='hasExport')

cmd_sim1 = graphlab.recommender.item_similarity_recommender.create(
    train, user_id='fromCode', item_id='cmdCode',target='hasExport', similarity_type='jaccard')

cmd_sim2 = graphlab.recommender.item_similarity_recommender.create(
    train, user_id='fromCode', item_id='cmdCode',target='hasExport', similarity_type='cosine')

cmd_sim3 = graphlab.recommender.item_similarity_recommender.create(
    train, user_id='fromCode', item_id='cmdCode',target='hasExport', similarity_type='pearson')


recommenders = [factor, cmd_sim1,cmd_sim2,cmd_sim3 ]


# Recommendations
def get_recomm(ctry_name, n, models = recommenders):
    ctry_code = int(WrangleData.ctry_lookup(ctry_name, code=False))
    for model in models:
        recomm = model.recommend(users =[ctry_code],k=n).to_dataframe()
        print 'Recommendations for %s using %s:' % (ctry_name, model)
        for i in range(len(recomm)):
            print '%d. %s'  % (i + 1, WrangleData.cmd_lookup(str(recomm.cmdCode[i])))


get_recomm('Uganda', 5)

sys.exit()
# Evaluate
test = sliceData.ctry_cmd_shares('comtrade_2015_2dg.tsv', yr = 2015)
test = test[test.exportShare>0]
test = graphlab.SFrame(test)
model_performance = graphlab.compare(test, recommenders)
#graphlab.show_comparison(model_performance,recommenders)
