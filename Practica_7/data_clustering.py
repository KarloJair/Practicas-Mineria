import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

df = pd.read_csv("vgsales.csv")
output_path = "Practica_7/outputs"
os.makedirs(output_path, exist_ok=True)


cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

df_na_clean = df[cols].dropna()


scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_na_clean)


inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertias.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'o-', color='blue')
plt.title('Elbow Method')
plt.xlabel('Número de Clusters: k')
plt.ylabel('Inercia')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_path}/elbow_method.png")
plt.close()


optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df_na_clean["Cluster"] = kmeans.fit_predict(scaled_data)

print(f"RESULTADOS DEL CLUSTERING k={optimal_k}")

cluster_summary = df_na_clean.groupby("Cluster")[cols].mean()
print(cluster_summary)


sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df_na_clean["NA_Sales"],
    y=df_na_clean["EU_Sales"],
    hue=df_na_clean["Cluster"],
    palette="tab10"
)

plt.title("Clusters según ventas en NA vs EU")
plt.xlabel("Ventas en Norteamérica")
plt.ylabel("Ventas en Europa")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(f"{output_path}/clusters_ventas.png")
plt.close()


df_na_clean_final = df_na_clean.copy()
df_na_clean_final.loc[df_na_clean.index, "Cluster"] = df_na_clean["Cluster"]
df_na_clean_final.to_csv(f"{output_path}/cluster_data.csv", index=False)

