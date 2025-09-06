import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

theme = sns.set_style("whitegrid")
df = pd.read_csv("vgsales.csv")
output_path = "Practica_3/plots/"
os.makedirs(output_path, exist_ok=True)

regiones = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]

regiones_titulos = {
    "NA_Sales": "Norteamérica",
    "EU_Sales": "Europa",
    "JP_Sales": "Japón",
    "Other_Sales": "Otras Regiones"
}

# PIE CHARTS: Ventas por región
for region in regiones:
    ventas_region = df.groupby("Genre")[region].sum().sort_values(ascending=False)
    plt.figure(figsize=(6,6))
    plt.pie(ventas_region, labels=ventas_region.index, autopct='%1.1f%%', startangle=90)
    plt.title(f"Distribución de ventas por género en {regiones_titulos[region]}", fontsize=10)
    plt.savefig(f"{output_path}pie_{region}.png")
    plt.close()

# BOXPLOTS: Ventas por Género
plt.figure(figsize=(12,6))
sns.boxplot(x="Genre", y="Global_Sales", data=df)
plt.xticks(rotation=45)
plt.title("Distribución de Ventas Globales por Género")
plt.savefig(f"{output_path}boxplot_genero.png")
plt.close()

# LINE PLOTS: Ventas por Año
ventas_por_año = df.groupby("Year")["Global_Sales"].sum()
plt.figure(figsize=(12,6))
plt.plot(ventas_por_año.index, ventas_por_año.values, marker="o")
plt.title("Ventas Globales por Año")
plt.xlabel("Año")
plt.ylabel("Ventas Globales (millones)")
plt.savefig(f"{output_path}linea_ventas_año.png")
plt.close()

# SCATTER PLOTS: Comparación regiones vs globales
for region in regiones:
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=region, y="Global_Sales", data=df, alpha=1, hue="Genre", palette="Set2")
    plt.title(f"Relación {regiones_titulos[region]} vs Ventas Globales")
    plt.xlabel(f"Ventas en {regiones_titulos[region]} (millones)")
    plt.ylabel("Ventas Globales (millones)")
    plt.savefig(f"{output_path}scatter_{region}.png")
    plt.close()
