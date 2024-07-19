import os
import pandas as pd
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Se define la funcion que se encargara de generar el texto de salida del LLM
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

def get_completion_from_messages(messages, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

# Cargar los datos
admissions_data = pd.read_csv('Fase3/admissions_mock.csv')
transfers_data = pd.read_csv('Fase3/transfers_mock.csv')
hcpsevents_data = pd.read_csv('Fase3/hcpsevents_mock.csv')
pharmacy_data = pd.read_csv('Fase3/pharmacy_mock.csv')
prescriptions_data = pd.read_csv('Fase3/prescriptions_mock.csv')
emar_data = pd.read_csv('Fase3/emar_mock.csv')


# Read resource data from datasetRecursos.ndjson

def uploadResourceInfo():
    data = []
    with open('Fase3/datasetRecursos.ndjson', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            json_line = json.loads(line)
            data.append(json_line)
    return data

# Convertir DataFrames a JSON
resource_data_json = uploadResourceInfo()
admissions_data_json = admissions_data.to_json(orient='records')
transfers_data_json = transfers_data.to_json(orient='records')
hcpsevents_data_json = hcpsevents_data.to_json(orient='records')
pharmacy_data_json = pharmacy_data.to_json(orient='records')
prescriptions_data_json = prescriptions_data.to_json(orient='records')
emar_data_json = emar_data.to_json(orient='records')

# Crear los mensajes
messages = [
    {'role': 'system', 'content': "You are a mapper tool assistant. You have information about FHIR resources and the tabular data to map. Your task is to tell the user to which FHIR resource each column of the table belongs, a column can belongs to multiple FHIR resources."},
    {'role': 'system', 'content': "You have to provide the user with a table that shows the mapping between the columns of the tabular data and the FHIR resources."},
    {'role': 'system', 'content': "The output format you have to provide is a JSON object with the following structure: {'column_name': 'resource_name'}. After that, you have to tell the user which resource should be generated with the provided data,based on how many times a FHIR resource appears in the column mapping  "},
    {'role': 'system', 'content': f"Resource information: {resource_data_json}"},
    {'role': 'user', 'content': f"The tabular data I would like to know the columns mapping are : {hcpsevents_data_json}. Tell me to which FHIR resource item each column belongs."},
]

# Obtener la respuesta del modelo
response = get_completion_from_messages(messages)

print(response)
