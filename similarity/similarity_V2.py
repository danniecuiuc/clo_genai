'''something wrong with the v1 since with pca, the feature explanation is not available. here we let them enter bloomberg id 
    and use it to extract target information, maybe do some scalar work and directly do nearest neighbor and calculate contribution
    then ranked but get some neighbor out with threshold (if not close enough)'''
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# load data to df
def load(path):
    return pd.read_excel(path)

# get feature for further use
def features(df):

    feature_cols = [
        "MVOC", "MVOC (w/o X)", "Attach", "Thickness", "Diversity",
        "WAS %ile", "% <80", "% <50", "% CCC", "Equity NAV", "Eq Last Payment",
        "Junior OC", "Excess Spread", "Deal WACC", "AAA Coupon",
        "DM TC", "DM to RP+24", "NC Yrs left", "RP Yrs left"
        ]
    
    return df[feature_cols].copy(), feature_cols

def target(df, idcol,tarid):# as mentioned before here they enter id and used it for extraction.
    
    row = df[df[idcol] == tarid]

    if row.empty:
        raise ValueError("Bloomberg ID not found")
    
    targetdf,fecol = features(row)
    target = targetdf.values
    
    return target[0], row.index[0]

def preprocess(df, feature):

    x = df[feature].values
    scaler = StandardScaler()
    xsc = scaler.fit_transform(x)

    return xsc, scaler

def knn(dataset, target, k): # both are preprocess or scaled data

    model = NearestNeighbors(n_neighbors=k, metric='euclidean')
    model.fit(dataset)
    dist, ind = model.kneighbors(target)# since before has been set to target[0]

    return dist, ind

# for explanation, says contribution, since we have euclidean distance, we can directly calculate with 
# feature of neighbor - feature of target with square value and normalized
def contribution(dataset,target,id,features):
    contrib = {}

    for i in id:
        diff = (dataset[i] - target[0])**2
        total = np.sum(diff) + 1e-12 # for knn, self is contain and sum is 0

        pairs = list(zip(features, diff/total))

        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

        filtered = [(f, v) for f, v in pairs_sorted if v > 0.1]

        contrib[i] = filtered

    return contrib


def similarity(df, idcol, targetid,k): #idcol should be bloomberg id
    
    ddf, feacol = features(df)
    dataset, scaler = preprocess(df,feacol)
    tar, tarid = target(df, idcol,targetid)
    tarsc = scaler.transform(tar.reshape(1,-1))

    dist, ind = knn(dataset,tarsc,k)
    ranked_raw = df.iloc[ind[0]].copy()
    ranked_raw["distance"] = dist[0]

    contrib = contribution(dataset,tarsc,ind[0],feacol)
    ranked_clean = ranked_raw[[idcol, "Collateral manager", "distance"]].copy()
    ranked_clean["top_features"] = ranked_clean.index.map(contrib)

    return ranked_raw, ranked_clean


