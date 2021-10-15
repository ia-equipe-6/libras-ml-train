import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

#TRACK_MODE = True

MODEL_PATH = "model"
WORDS_ENCODER = "words_encoder"
ARQUIVO_ENTRADA = "libras_dataset.csv"

#MODEL_PATH = "model_track" if TRACK_MODE else "model"
#WORDS_ENCODER = "words_encoder_track" if TRACK_MODE else "words_encoder"
#ARQUIVO_ENTRADA = "libras_dataset_track.csv" if TRACK_MODE else "libras_dataset.csv"

ACTIVATION = "relu"
#ACTIVATION = "tanh"

#MODEL_PATH += "_" + ACTIVATION
#WORDS_ENCODER += "_" + ACTIVATION

ds_libras = pd.read_csv(ARQUIVO_ENTRADA)

dscolnames = ds_libras.columns[1:]
inputSize = len(dscolnames)

encoder = LabelEncoder()
encoder.fit(ds_libras['WORD'])
ds_libras['WORD'] = encoder.fit_transform(ds_libras['WORD'])

np.save(WORDS_ENCODER, encoder.classes_)
outputSize = len(encoder.classes_)

columns_train = ds_libras[dscolnames]
result_train = ds_libras["WORD"]

print(f"Train data: {columns_train.shape}, labels: {result_train.shape}")

model = Sequential()

model.add(Dense(10000, 
    activation=ACTIVATION,
    input_dim=inputSize,
    input_shape=(inputSize,)
    )
)

model.add(Dense(outputSize, 
    activation="softmax"
    )
)

model.summary()

#loss: https://www.tensorflow.org/api_docs/python/tf/keras/losses
#optimizers: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/

model.compile(optimizer="adam", 
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"]
             )

history = model.fit(columns_train, 
                    result_train, 
#                   verbose=0, 
                    epochs=150, 
                    batch_size=32
                   )


model.save(MODEL_PATH)