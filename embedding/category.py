import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from jina import Document, DocumentArray

def category_embedding(pks,cds,fname):

    naics_codes = []
    for i in cds:
        naics_codes.append(str(int(i)))
    # Create the cumulative codes
    sectors = [code[:2] for code in naics_codes]
    subsectors = [code[:3] for code in naics_codes]
    industry_groups = [code[:4] for code in naics_codes]
    naics_industries = [code[:5] for code in naics_codes]
    national_industries = [code for code in naics_codes]

    # Encode the cumulative codes as integers
    encoder = LabelEncoder()
    encoded_sectors = encoder.fit_transform(sectors)
    encoded_subsectors = encoder.fit_transform(subsectors)
    encoded_industry_groups = encoder.fit_transform(industry_groups)
    encoded_naics_industries = encoder.fit_transform(naics_industries)
    encoded_national_industries = encoder.fit_transform(national_industries)

    # Define the embedding layers
    embedding_dim = 6  
    sector_embedding = tf.keras.layers.Embedding(input_dim=len(np.unique(encoded_sectors)), output_dim=embedding_dim)
    subsector_embedding = tf.keras.layers.Embedding(input_dim=len(np.unique(encoded_subsectors)), output_dim=embedding_dim)
    industry_group_embedding = tf.keras.layers.Embedding(input_dim=len(np.unique(encoded_industry_groups)), output_dim=embedding_dim)
    naics_industry_embedding = tf.keras.layers.Embedding(input_dim=len(np.unique(encoded_naics_industries)), output_dim=embedding_dim)
    national_industry_embedding = tf.keras.layers.Embedding(input_dim=len(np.unique(encoded_national_industries)), output_dim=embedding_dim)

    # Get the embeddings for each cumulative code
    sector_embedded = sector_embedding(encoded_sectors)
    subsector_embedded = subsector_embedding(encoded_subsectors)
    industry_group_embedded = industry_group_embedding(encoded_industry_groups)
    naics_industry_embedded = naics_industry_embedding(encoded_naics_industries)
    national_industry_embedded = national_industry_embedding(encoded_national_industries)

    # Take a weighted sum of the embeddings
    weights = tf.constant([0.55, 0.25, 0.1, 0.05, 0.05])  # The importance weights
    naics_embedding = weights[0] * sector_embedded + weights[1] * subsector_embedded + weights[2] * industry_group_embedded + weights[3] * naics_industry_embedded + weights[4] * national_industry_embedded

    print(naics_embedding[3].numpy().tolist())

    cat = DocumentArray()

    for i in range(len(naics_embedding)):
        cat.append(Document(uri = pks[i], 
                tags = {'emb':naics_embedding[i].numpy().tolist()}))
    cat.save_csv(f'./cat/cat_{fname}.csv',flatten_tags = True)       
    return cat


selling = pd.read_csv(f'../../../datasets/spend-ohio/ohiopoi.csv')
pks = selling['placekey']
# pks = pks[:50]
cds = selling['naics_code']
# cds = cds[:50]
print(str(int(cds[1])))
category_embedding(pks,cds,'2023-1')
