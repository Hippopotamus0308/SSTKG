import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from sklearn.manifold._t_sne import TSNE

dd = pd.read_csv('../data_processing/data/geoLL.csv')
import json

locRes = []
for i in range(len(dd)):
    loc = dd['tag__loc'][i]
    array = json.loads(loc.replace("'", '"'))
    locRes.append({'pk':dd['uri'][i],'loc':array})

loc_test = []
for i in range(100):
    loc_test.append(locRes[i]['loc'])

coords = np.array(loc_test)

mds = MDS(n_components=2)
coords_mds = mds.fit_transform(coords)


tsne = TSNE(n_components=2)
coords_tsne = tsne.fit_transform(coords)
for i in range(6):
    print(locRes[i]['loc'])
    print(coords_tsne[i])
    print('------------')
selling = pd.read_csv('../data_processing/data/2023-1/spend_patterns.csv')
selling_record = selling['spend_by_day'][0]

selling_re = [float(x) for x in selling_record[1:-1].split(',')]
selling_records = []
for i in range(4):
    list = []
    for j in range(7):
        list.append(selling_re[j+i*7])
    selling_records.append(list)