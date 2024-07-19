import pandas as pd

# Crear un DataFrame de pandas
df = pd.read_csv("Fase3\mapping_porcentage.csv")

# Calcular los porcentajes de acierto de cada modelo
gpt_accuracy = df["gpt_percentage"].mean() * 100
llama_accuracy = df["llama_percentage"].mean() * 100

print(f"Porcentaje de acierto de GPT: {gpt_accuracy:.2f}%")
print(f"Porcentaje de acierto de Llama: {llama_accuracy:.2f}%")
