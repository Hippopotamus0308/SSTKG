import placekey as pk
from jina import Document, DocumentArray
import pandas as pd
import numpy as np
import umap

def spatial_embedding(pks,fname):
    locs = []
    for p in pks:
        loc = pk.placekey_to_geo(p)
        locs.append(loc)
    coords = np.array(locs)
    reducer = umap.UMAP(verbose=True)
    coords_final = reducer.fit_transform(coords)
    print(coords_final[0])
    # tsne = TSNE(n_components=2, verbose=1)
    # coords_final = tsne.fit_transform(coords)
    # mds = MDS(n_components=2,verbose=1)
    # coords_final = mds.fit_transform(coords)
    # spe = DocumentArray()

    # for i in range(len(coords_final)):
    #     spe.append(Document(uri = pks[i], 
    #             tags = {'loc':locs[i], 'emb':[coords_final[i][0],coords_final[i][1]]}))
    # spe.save_csv(f'./spe/spe_{fname}.csv',flatten_tags = True)       
    # return spe

selling = pd.read_csv('../../../datasets/spend-ohio/2023-1/spend_patterns.csv')
pks = selling['placekey']
pks = pks[:50]
spatial_embedding(pks,'2023-1')
#print(pks[0])