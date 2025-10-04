import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("vgsales.csv")
output_path = "Practica_6/plots/"
os.makedirs(output_path, exist_ok=True)

UMBRAL_EXITO = 1.0 # Se considera un juego exitoso si vende al menos 1 millón de copias
df['Exito'] = (df['Global_Sales'] >= UMBRAL_EXITO).astype(int)

print("\nDistribución de la Variable Objetivo 'Exito':")
print(df['Exito'].value_counts())
print("--------------------------------------------------------------------------------------------")

features = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Year', 'Genre', 'Platform']
X = df[features]
y = df['Exito']


X = pd.get_dummies(X, columns=['Genre', 'Platform'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


print(f"Tamaño del conjunto de entrenamiento: {len(X_train)}")
print(f"Tamaño del conjunto de prueba: {len(X_test)}")
print("--------------------------------------------------------------------------------------------")

scaler = StandardScaler()

cols_to_scale = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Year']

X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])


K = 5 
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud (Accuracy) del modelo KNN (K={K}): {accuracy:.4f}")
print("--------------------------------------------------------------------------------------------")

conf_mat = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Exitoso', 'Exitoso'], yticklabels=['No Exitoso', 'Exitoso'])
plt.title(f'Matriz de Confusión (KNN, K={K})')
plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.savefig(f"{output_path}confusion_matrix.png")
plt.close()


print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['No Exitoso (0)', 'Exitoso (1)']))
print("--------------------------------------------------------------------------------------------")

