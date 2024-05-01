# from sklearn.manifold import MDS,TSNE
import placekey as pk
from jina import Document, DocumentArray
import pandas as pd
import numpy as np
import umap
import umap.plot
from sklearn.preprocessing import StandardScaler

def spatial_embedding(pks,fname):
    locs = []
    for p in pks:
        loc = pk.placekey_to_geo(p)
        locs.append(loc)
    coords = np.array(locs)
    coords_rad = np.radians(coords)

    x = np.cos(coords_rad[:, 0]) * np.cos(coords_rad[:, 1])
    y = np.cos(coords_rad[:, 0]) * np.sin(coords_rad[:, 1])
    z = np.sin(coords_rad[:, 0])
    coords_3d = np.vstack([x, y, z]).T

    scaler = StandardScaler()
    coords_3d_scaled = scaler.fit_transform(coords_3d)

    reducer = umap.UMAP(verbose=True)
    coords_final = reducer.fit_transform(coords_3d_scaled)
    print(coords_final[0])

    # tsne = TSNE(n_components=2, verbose=1)
    # coords_final = tsne.fit_transform(coords)
    # mds = MDS(n_components=2,verbose=1)
    # coords_final = mds.fit_transform(coords)

    spe = DocumentArray()

    for i in range(len(coords_final)):
        spe.append(Document(uri = pks[i], 
                tags = {'loc':locs[i], 'emb':[coords_final[i][0],coords_final[i][1]]}))
    spe.save_csv(f'./spe/spe_{fname}.csv',flatten_tags = True)       
    return spe

for i in range(1,13):
    selling = pd.read_csv(f'../../../datasets/spend-ohio/2022-{i}/spend_patterns.csv')
    pks = selling['placekey']
    # pks = pks[:50]
    spatial_embedding(pks,f'2022-{i}')
    print(f"2022-{i} finished")

for i in range(2,5):
    selling = pd.read_csv(f'../../../datasets/spend-ohio/2023-{i}/spend_patterns.csv')
    pks = selling['placekey']
    # pks = pks[:50]
    spatial_embedding(pks,f'2023-{i}')
    print(f"2023-{i} finished")
#print(pks[0])
