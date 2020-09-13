import pandas as pd
import tensorflow as tf

# Set Data
boston = pd.read_csv("boston.csv")
boston_indep = boston[
    [
        "crim",
        "zn",
        "indus",
        "chas",
        "nox",
        "rm",
        "age",
        "dis",
        "rad",
        "tax",
        "ptratio",
        "b",
        "lstat",
    ]
]
boston_dep = boston[["medv"]]

# Build Model
X = tf.keras.layers.Input(shape=[13])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss="mse")

# Fit Model
model.fit(boston_indep, boston_dep, epochs=1000, verbose=0)

# Use Model
print(model.predict(boston_indep[0:5]))
print(boston_dep[0:5])

# Check Model
print(model.get_weights())
