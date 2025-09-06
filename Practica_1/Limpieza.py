import pandas as pd

df = pd.read_csv("Practica 1/base vgsales.csv", encoding="latin-1")

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

print("Valores nulos por columna:\n", df.isnull().sum())
df = df.dropna()

# Eliminamos datos duplicados si es que los hay
df = df.drop_duplicates()

# Guardamos el dataset
df.to_csv("vgsales.csv", index=False)

print("Dataset limpio guardado como vgsales.csv")