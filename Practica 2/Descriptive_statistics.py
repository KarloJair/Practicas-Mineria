import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("vgsales.csv")

# Descripción general
print(df.info())
print(df.describe())

# Frecuencias de las categorías
print(df['Platform'].value_counts())
# Obteniendo la frecuencias de las plataformas podemos ver que las que más juegos tienen son la DS y la PS2, ambos con más de 2000 juegos.

print(df['Genre'].value_counts())
# El genero mas comun es Action, seguido por Sports.

print(df['Publisher'].value_counts())
# El publisher con más juegos es Electronic Arts, con más de 1000 juegos.

print("\nVentas por genero (Millones): ")
ventas_por_genero = df.groupby("Genre")["Global_Sales"].sum().sort_values(ascending=False)
print(ventas_por_genero)

print("\nVentas por plataforma (Millones): ")
ventas_por_plataforma = df.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False)
print(ventas_por_plataforma)

print("\nVentas por año (Millones): ")
ventas_por_año = df.groupby("Year")["Global_Sales"].sum()
print(ventas_por_año)
