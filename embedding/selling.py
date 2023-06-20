from jina import Document, DocumentArray
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import datetime

def selling_embedding(selling, folder_name='2023-1'):
    #folder_name = '2023-1'
    year, month = folder_name.split('-')
    date = datetime.datetime(int(year), int(month), 1)
    formatted_date = date.strftime('%Y-%m-01')

    all_records = []
    for entity in selling:

        selling_re = [float(x) for x in entity[1:-1].split(',')]
        #print(selling_re)
        selling_records = []

        for i in range(4):
            lists = []
            for j in range(7):
                lists.append(selling_re[j+i*7])
            selling_records.append(lists)

        selling_records = np.array(selling_records)
        all_records.append(selling_records)
    

    all_records = np.array(all_records)
    all_records = all_records.reshape((all_records.shape[0], all_records.shape[1], all_records.shape[2],1))
    train_data = all_records[:80]
    train_targets = all_records[:80]
    val_data = all_records[80:]
    val_targets = all_records[80:]

    embedding_dim = 50

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(train_data.shape[1], train_data.shape[2])))
    #model.add(RepeatVector(7))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(LSTM(50, return_sequences=True))
    model.add(TimeDistributed(Dense(7)))

    model.compile(optimizer='adam', loss='mse')
    model.fit(train_data, train_targets, epochs=50, verbose=0, batch_size=32,
        validation_data=(val_data, val_targets))
    embedding_model = Model(inputs=model.inputs, outputs=model.layers[0].output)
    embeddings = embedding_model.predict(all_records)    

    print(len(embeddings))
    print(len(embeddings[0]))
    print(embeddings[0][0])
    print(embeddings[0][1])

def record(date_str):
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    week_number = date.isocalendar()[1]
    print(week_number)


# selling = pd.read_csv('../../../datasets/spend-ohio/2023-1/spend_patterns.csv')
# selling_record = selling['spend_by_day'][:100]
# # print(selling_record[0])
# # print(type(selling_record))
# selling_embedding(selling_record)