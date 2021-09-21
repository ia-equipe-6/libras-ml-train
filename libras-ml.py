import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

ds_libras = pd.read_csv('libras_dataset.csv')

dscolnames = ds_libras.columns[2:]

encoder = LabelEncoder()
encoder.fit(ds_libras['WORD'])
ds_libras['WORD'] = encoder.fit_transform(ds_libras['WORD'])

np.save("words.encoder", encoder.classes_)

columns_train = ds_libras[dscolnames]
result_train = ds_libras["WORD"]

print(f"Train data: {columns_train.shape}, labels: {result_train.shape}")

model = Sequential()

model.add(Dense(1000, 
    activation="relu",
    input_dim=439,
    input_shape=(439,)
    )
)

model.add(Dense(1000, 
    activation="tanh",
    input_dim=439,
    input_shape=(439,)
    )
)

model.add(Dense(11, 
    activation="softmax"
    )
)

model.summary()

#loss: https://www.tensorflow.org/api_docs/python/tf/keras/losses
#optimizers: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/

model.compile(optimizer="adam", 
              #loss="mse",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"]
             )

history = model.fit(columns_train, 
                    result_train, 
#                   verbose=0, 
                    epochs=110, 
                    batch_size=32
                   )


model.save("model")