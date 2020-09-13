import pandas as pd

# Read Data from CSV files

boston_path = "boston.csv"
boston = pd.read_csv(boston_path)

lemonade_path = "lemonade.csv"
lemonade = pd.read_csv(lemonade_path)

iris_path = "iris.csv"
iris = pd.read_csv(iris_path)

# Check Data by Shape

print(boston.shape)
print(lemonade.shape)
print(iris.shape)

# Print Column

print(boston.columns)
print(lemonade.columns)
print(iris.columns)

lemon_indep = lemonade[["온도"]]
lemon_dep = lemonade[["판매량"]]
print(lemon_indep.shape, lemon_dep.shape)

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
print(boston_indep.shape, boston_dep.shape)

iris_indep = iris[["꽃잎길이", "꽃잎폭", "꽃받침길이", "꽃받침폭"]]
iris_dep = iris[["품종"]]
print(iris_indep.shape, iris_dep.shape)

print(lemonade.head())
print(boston.head())
print(iris.head())