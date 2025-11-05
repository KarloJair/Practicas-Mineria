import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

df = pd.read_csv('vgsales.csv')
output_path = "Practica_9/outputs"

os.makedirs(output_path, exist_ok=True)
 

sales_publisher = df.groupby('Publisher')['Global_Sales'].sum().to_dict()

wordcloud = WordCloud(width=800, height=400, background_color='white')
wordcloud.generate_from_frequencies(sales_publisher)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig(f"{output_path}/global_sales_publisher.png")
plt.title("Ventas globales por publisher")
plt.show()

sales_genre = df.groupby('Genre')['Global_Sales'].sum().to_dict()

wordcloud = WordCloud(width=800, height=400, background_color='white')
wordcloud.generate_from_frequencies(sales_genre)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig(f"{output_path}/global_sales_genre.png")
plt.title("Ventas globales por genero")
plt.show()