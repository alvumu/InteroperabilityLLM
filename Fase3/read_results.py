
import pandas as pd
# Leer los datos en un DataFrame
df = pd.read_csv("Fase3/mapping_porcentage.csv")

# Número total de atributos
total_attributes = df.shape[0]

# Contar el número de atributos efectivos para cada modelo
gpt_effective = df['gpt_percentage'].sum()
llama_effective = df['llama_percentage'].sum()

# Calcular los porcentajes de efectividad
gpt_percentage = (gpt_effective / total_attributes) * 100
llama_percentage = (llama_effective / total_attributes) * 100

print("Porcentaje de efectividad de GPT: ", gpt_percentage)
print("Porcentaje de efectividad de Llama: ", llama_percentage)
