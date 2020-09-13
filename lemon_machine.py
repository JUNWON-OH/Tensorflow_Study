#%%
import pandas as pd
import tensorflow as tf

# Set Data
lemonade = pd.read_csv("lemonade.csv")
lemonade_indep = lemonade[["온도"]]
lemonade_dep = lemonade[["판매량"]]

# Build Model
X = tf.keras.layers.Input(shape=[1])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss="mse")

# Fit Model
model.fit(lemonade_indep, lemonade_dep, epochs=10000, verbose=0)

# Use Model
print(model.predict([[15]]))

# %%
