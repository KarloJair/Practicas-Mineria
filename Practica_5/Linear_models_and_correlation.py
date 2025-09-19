import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

output_path = "Practica_5/plots/"
output_path_JP_plots = "Practica_5/plots/JP_plots/"
df = pd.read_csv("vgsales.csv")

sales_colums = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]

# Matriz de correlación
corr = df[sales_colums].corr()

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Matriz de correlación entre ventas")
plt.savefig(f"{output_path}heatmap_sales.png")
plt.close()
# El mapa ya nos dice que aun que las ventas entre los otros paises tiene cierta correlación
# con las ventas en Japón, la correlación es muy baja en comparación con la correlación
# entre las ventas en América del Norte y Europa.

# Regresión lineal ---------------------------------------
# Predicción de ventas en América del Norte (NA_Sales)
X = df[["EU_Sales", "JP_Sales", "Other_Sales"]]
y = df["NA_Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"(NA_Sales) R² score: {r2:.4f}")

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Ventas de America Reales")
plt.ylabel("Ventas de America Predichas")
plt.title("Regresión Lineal - Ventas de America")
plt.plot([0, max(y_test)], [0, max(y_test)], color='red')
plt.savefig(f"{output_path}linear_regression_NA_Sales.png")
plt.close()

# Predicción de ventas en Europa (EU_Sales)
X = df[["NA_Sales", "JP_Sales", "Other_Sales"]]
y = df["EU_Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"(EU_Sales) R² score: {r2:.4f}")

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Ventas de Europa Reales")
plt.ylabel("Ventas de Europa Predichas")
plt.title("Regresión Lineal - Ventas de Europa")
plt.plot([0, max(y_test)], [0, max(y_test)], color='red')
plt.savefig(f"{output_path}linear_regression_EU_Sales.png")
plt.close()

# Predicción de ventas en Japón (JP_Sales) en base a las ventas globales
X = df[["Global_Sales"]]
y = df["JP_Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"(JP_Sales-Global) R² score: {r2:.4f}")

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Ventas de Japon Reales")
plt.ylabel("Ventas de Japon Predichas")
plt.title("Regresión Lineal - Ventas de Japon")
plt.plot([0, max(y_test)], [0, max(y_test)], color='red')
plt.savefig(f"{output_path_JP_plots}linear_regression_JP_Sales_Global_sales.png")
plt.close()

# Predicción de ventas en Japón (JP_Sales) en base a las ventas globales
X = df[["NA_Sales", "EU_Sales", "Other_Sales"]]
y = df["JP_Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"(JP_Sales) R² score: {r2:.4f}")

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Ventas de Japon Reales")
plt.ylabel("Ventas de Japon Predichas")
plt.title("Regresión Lineal - Ventas de Japon")
plt.plot([0, max(y_test)], [0, max(y_test)], color='red')
plt.savefig(f"{output_path_JP_plots}linear_regression_JP_Sales.png")
plt.close()

# -------------------------------------------------------