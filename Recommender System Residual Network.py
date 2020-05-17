
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten, Dense, Concatenate
from keras.layers import Dropout, BatchNormalization, Activation
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

# load in the data
df = pd.read_csv('very_small_rating.csv')

N = df.userId.max() + 1 # number of users
M = df.movie_idx.max() + 1 # number of movies

# split into train and test
df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

# initialize variables
K = 10 # latent dimensionality
mu = df_train.rating.mean()
epochs = 15
reg = 0. # regularization penalty


# keras model
u = Input(shape=(1,))
m = Input(shape=(1,))
u_embedding = Embedding(N, K,embeddings_regularizer=l2(reg))(u) # (N, 1, K)
m_embedding = Embedding(M, K,embeddings_regularizer=l2(reg))(m) # (N, 1, K)


##### main branch
u_bias = Embedding(N, 1,embeddings_regularizer=l2(reg))(u) # (N, 1, 1)
m_bias = Embedding(M, 1,embeddings_regularizer=l2(reg))(m) # (N, 1, 1)
x = Dot(axes=2)([u_embedding, m_embedding]) # (N, 1, 1)
x = Add()([x, u_bias, m_bias])
x = Flatten()(x) # (N, 1)


##### side branch
u_embedding = Flatten()(u_embedding) # (N, K)
m_embedding = Flatten()(m_embedding) # (N, K)
y = Concatenate()([u_embedding, m_embedding]) # (N, 2K)
#First Layer
y = Dense(400)(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = Dropout(0.5)(y)
#Hidden layer
y = Dense(100)(y) 
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = Dense(1)(y)


##### merge the matrix factorization layer and the ANN layers together
x = Add()([x, y])

model = Model(inputs=[u, m], outputs=x)
model.compile(
  loss='mse',
  # optimizer='adam',
  # optimizer=Adam(lr=0.01),
  optimizer=SGD(lr=0.08, momentum=0.9),
  metrics=['mse'],
)

r = model.fit(
  x=[df_train.userId.values, df_train.movie_idx.values],
  y=df_train.rating.values - mu,
  epochs=epochs,
  batch_size=128,
  validation_data=(
    [df_test.userId.values, df_test.movie_idx.values],
    df_test.rating.values - mu
  )
)


print("Mean Squared Error:",r.history['mse'][len(r.history['mse'])-1],"Val_MSE",r.history['val_mse'][len(r.history['val_mse'])-1])
print("UserID: 980 would rate the movieID: 47 as below")
print(model.predict([np.array([980]),np.array([47])]) + mu)

