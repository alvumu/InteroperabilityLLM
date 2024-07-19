# Importar las librerías necesarias
import os
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from together import Together
from transformers import BertModel, BertTokenizer
import nltk
from nltk.tokenize import word_tokenize
import tensorflow_hub as hub
from rank_bm25 import BM25Okapi
import pandas as pd
from gensim.models import Word2Vec

# Descargar datos necesarios para NLTK
nltk.download('punkt')


def load_dataset(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        texts = []
        for line in lines:
            json_line = json.loads(line)  # Convertir cada línea a un objeto JSON
            texts.append(json_line)       # Añadir el objeto JSON a la lista de textos
    return texts

    
def create_corpus(texts):
    corpus = []
    for text in texts:
        # Concatenar el recurso, descripción y estructura en un solo string y convertirlo a minúsculas
        content = text["resource"].lower() + " " + text["description"].lower() + " " + str(text["structure"]).lower()
        corpus.append(content)  # Añadir el contenido procesado al corpus
    return corpus


def tfidf_faiss_similarity(corpus, query):
    vectorizer = TfidfVectorizer()  # Crear el vectorizador TF-IDF
    tfidf_matrix = vectorizer.fit_transform(corpus).toarray().astype(np.float32)  # Vectorizar el corpus

    query_vec = vectorizer.transform([query]).toarray().astype(np.float32)  # Vectorizar la consulta

    similarity_scores = cosine_similarity(query_vec, tfidf_matrix)[0]  # Calcular la similitud del coseno
    similarity_dict = {i: score for i, score in enumerate(similarity_scores)}  # Crear un diccionario con los puntajes de similitud
    print(similarity_dict)
    return similarity_dict


def load_univ_sent_encoder_model():
    
    model_url = "https://tfhub.dev/google/universal-sentence-encoder"
    model = hub.load(model_url)  # Cargar el modelo desde TensorFlow Hub
    return model

def univ_sent_encoder(corpus, query):
    model = load_univ_sent_encoder_model()  # Cargar el modelo
    query_embedding = model([query])  # Obtener el embedding de la consulta

    similarity_dict = {}
    for idx, doc in enumerate(corpus):
        doc_embedding = model([doc])  # Obtener el embedding del documento
        similarity_scores = cosine_similarity(query_embedding, doc_embedding)  # Calcular la similitud del coseno
        similarity_dict[idx] = similarity_scores[0][0]  # Guardar la similitud en el diccionario
    print(similarity_dict)
    return similarity_dict

def word2vec_similarity(corpus, query):

    # Preprocesar el corpus y la consulta
    processed_query = query.split()

    # Entrenar el modelo Word2Vec
    model = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=1, workers=4)

    # Obtener el vector de la consulta
    query_vec = np.mean([model.wv[word] for word in processed_query if word in model.wv], axis=0).reshape(1, -1)

    # Obtener los vectores de cada documento en el corpus
    corpus_vecs = []
    for doc in corpus:
        doc_vec = np.mean([model.wv[word] for word in doc if word in model.wv], axis=0)
        corpus_vecs.append(doc_vec)
    corpus_vecs = np.array(corpus_vecs)

    # Calcular la similitud del coseno
    similarity_scores = cosine_similarity(query_vec, corpus_vecs)

    return similarity_scores

def bm25_similarity(corpus, query):
    tokenized_corpus = [doc.split(" ") for doc in corpus]  # Tokenizar el corpus
    tokenized_query = query.split(" ")  # Tokenizar la consulta
    bm25_index = BM25Okapi(tokenized_corpus, k1=1.75, b=0.75)  # Crear el índice BM25

    bm25_scores = bm25_index.get_scores(tokenized_query)  # Obtener los puntajes BM25 para la consulta
    similarity_dict = {i: score for i, score in enumerate(bm25_scores)}  # Crear un diccionario con los puntajes de similitud
    print(similarity_dict)

    return similarity_dict

def reciprocal_rank_fusion(rank_lists, k=60):
    rrf_scores = defaultdict(float)  # Crear un diccionario para almacenar los puntajes RRF

    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list):
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)  # Calcular la recíproca del rango y sumar al puntaje del documento

    sorted_rrf_scores = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)  # Ordenar los puntajes de mayor a menor
    return sorted_rrf_scores

def transformers_similarity(corpus, query):

    model = SentenceTransformer("avsolatorio/GIST-Embedding-v0") # Cargar el modelo de Transformers

    # Obtener el embedding de la consulta
    query_vec = model.encode([query])[0]
    # Obtener los embeddings del corpus
    corpus_vecs = model.encode(corpus)

    # Calcular la similitud del coseno entre el embedding de la consulta y los embeddings del corpus
    similarity_scores = cosine_similarity([query_vec], corpus_vecs)[0]

    # Crear un diccionario con los puntajes de similitud
    similarity_dict = {i: score for i, score in enumerate(similarity_scores)}

    return similarity_dict

def fasttext_similarity(corpus, query):
    
        # Preprocesar el corpus y la consulta
        processed_corpus = [word_tokenize(doc) for doc in corpus]
        processed_query = word_tokenize(query)
    
        # Entrenar el modelo FastText
        model = Word2Vec(sentences=processed_corpus, vector_size=128, window=8, min_count=1, workers=4)
    
        # Obtener el vector de la consulta
        query_vec = np.mean([model.wv[word] for word in processed_query if word in model.wv], axis=0).reshape(1, -1)
    
        # Obtener los vectores de cada documento en el corpus
        corpus_vecs = []
        for doc in processed_corpus:
            doc_vec = np.mean([model.wv[word] for word in doc if word in model.wv], axis=0)
            corpus_vecs.append(doc_vec)
        corpus_vecs = np.array(corpus_vecs)
    
        # Calcular la similitud del coseno
        similarity_scores = cosine_similarity(query_vec, corpus_vecs)[0]
        similarity_dict = {i: score for i, score in enumerate(similarity_scores)}  # Crear un diccionario con los puntajes de similitud
        return similarity_dict

def encode_bert(corpus, query):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Inicializar el tokenizador BERT
    model = BertModel.from_pretrained('bert-base-uncased')  # Inicializar el modelo BERT

    query_inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)  # Vectorizar la consulta
    query_outputs = model(**query_inputs)  # Obtener los outputs del modelo
    query_vec = query_outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Obtener el vector de la consulta

    corpus_vecs = []
    for doc in corpus:
        doc_inputs = tokenizer(doc, return_tensors="pt", truncation=True, padding=True)  # Vectorizar cada documento
        doc_outputs = model(**doc_inputs)  # Obtener los outputs del modelo
        doc_vec = doc_outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Obtener el vector del documento
        corpus_vecs.append(doc_vec)

    corpus_vecs = np.array(corpus_vecs).squeeze()  # Convertir la lista de vectores a un array numpy

    similarity_scores = cosine_similarity(query_vec, corpus_vecs)[0]  # Calcular la similitud del coseno
    similarity_dict = {i: score for i, score in enumerate(similarity_scores)}  # Crear un diccionario con los puntajes de similitud
    return similarity_dict



def get_top_texts(vec_similarity_dict, bm25_similarity_dict, univ_sent_encoder_similarity_dic, fasttext_similarities, corpus, top_k=5):
    # Obtener los índices de los textos más similares usando TF-IDF y ordenar de mayor a menor similitud
    vec_similarity_dict_sorted = dict(sorted(vec_similarity_dict.items(), key=lambda item: item[1], reverse=True))
    vec_top_similarity = {k: vec_similarity_dict_sorted[k] for k in list(vec_similarity_dict_sorted)}
    vec_indexes = list(vec_top_similarity.keys())
    print("Faiss: ", vec_indexes)

    # Obtener los índices de los textos más similares usando BM25 y ordenar de mayor a menor similitud
    bm25_similarity_dict_sorted = dict(sorted(bm25_similarity_dict.items(), key=lambda item: item[1], reverse=True))
    bm25_top_similarity = {k: bm25_similarity_dict_sorted[k] for k in list(bm25_similarity_dict_sorted)}
    bm25_indexes = list(bm25_top_similarity.keys())
    print("BM25: ", bm25_indexes)

    # Obtener los índices de los textos más similares usando Universal Sentence Encoder y ordenar de mayor a menor similitud
    univ_sent_encoder_similarity_dic_similarity_dict_sorted = dict(sorted(univ_sent_encoder_similarity_dic.items(), key=lambda item: item[1], reverse=True))
    univ_sent_encoder_similarity_dic_top_similarity = {k: univ_sent_encoder_similarity_dic_similarity_dict_sorted[k] for k in list(univ_sent_encoder_similarity_dic_similarity_dict_sorted)}
    univ_sent_encoder_similarity_dic_indexes = list(univ_sent_encoder_similarity_dic_top_similarity.keys())
    print("Universal Sentence Encoder: ", univ_sent_encoder_similarity_dic_indexes)

    # transformers_similarity_dic_sorted = dict(sorted(transformers_similarity_dic.items(), key=lambda item: item[1], reverse=True))
    # transformers_similarity_dic_top_similarity = {k: transformers_similarity_dic_sorted[k] for k in list(transformers_similarity_dic_sorted)}
    # transformers_dic_indexes = list(transformers_similarity_dic_top_similarity.keys())
    # print("Transformers: ", transformers_dic_indexes)

    fasttext_similarities_sorted = dict(sorted(fasttext_similarities.items(), key=lambda item: item[1], reverse=True))
    fasttext_top_similarity = {k: fasttext_similarities_sorted[k] for k in list(fasttext_similarities_sorted)}
    fasttext_indexes = list(fasttext_top_similarity.keys())
    print("FastText: ", fasttext_indexes)

    # Se combinan los rankings de los tres métodos utilizando la fusión de rango recíproco (RRF)
    combined_rankings = reciprocal_rank_fusion([vec_indexes, bm25_indexes, univ_sent_encoder_similarity_dic_indexes])

    # Se procesan los índices, obteniendo únicamente los k mejores
    final_indexes = [index for index, _ in combined_rankings[:top_k]]
    print("Final Indexes: ", final_indexes)

    # Recuperar los textos relevantes del corpus utilizando los índices obtenidos
    relevant_texts = [corpus[i] for i in final_indexes]

    # Concatenar los textos relevantes en un solo string
    context = "\n".join(relevant_texts)

    return context


def generate_response(query, modelName, context):

    if modelName == "GPT":
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages = [
                {'role': 'system', 'content': "You are a mapper tool assistant. You have information about FHIR resources and the tabular data to map. Your task is to tell the user to which FHIR resource each column of the table belongs, a column can belongs to multiple FHIR resources. Take into account also the column values"},
                {'role': 'system', 'content': "You have to provide the user with a table that shows the mapping between the columns of the tabular data and the FHIR resources."},
                {'role': 'system', 'content': "The output format you have to provide is a JSON object with the following structure: {'column_name': 'resource_name'}. After that, you have to tell the user which resource should be generated with the provided data,based on how many times a FHIR resource appears in the column mapping  "},
                {'role': 'system', 'content': f"Context information: {context}"},
                {'role': 'user', 'content': f"{query}"}],
            temperature=0, 
        )
    elif modelName == "Llama":
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        response = client.chat.completions.create(
                    model="meta-llama/Llama-3-8b-chat-hf",
                    messages = [
                        {'role': 'system', 'content': "You are a mapper tool assistant. You have information about FHIR resources and the tabular data to map. Your task is to tell the user to which FHIR resource each column of the table belongs, a column can belongs to multiple FHIR resources."},
                        {'role': 'system', 'content': "You have to provide the user with a table that shows the mapping between the columns of the tabular data and the FHIR resources."},
                        {'role': 'system', 'content': "The output format you have to provide is a JSON object with the following structure: {'column_name': 'resource_name'}. After that, you have to tell the user which resource should be generated with the provided data,based on how many times a FHIR resource appears in the column mapping"},
                        {'role': 'system', 'content': f"Context information: {context}"},
                        {'role': 'user', 'content': f"{query}"}],
                    temperature=0, 
                )


    return response.choices[0].message.content


#------------------------------------------------------------
def get_description(table):

    admissions_description = """ 
    The admissions table gives information regarding a patient’s admission to the hospital. Since each unique hospital visit for a patient is assigned a unique hadm_id, the admissions table can be considered as a definition table for hadm_id. Information available includes timing information for admission and discharge, demographic information, the source of the admission, and so on.
    subject_id
    subject_id is a unique identifier which specifies an individual patient. Any rows associated with a single subject_id pertain to the same individual. As subject_id is the primary key for the table, it is unique for each row.

    hadm_id
    hadm_id is a unique identifier which specifies an individual hospitalization. Any rows associated with a single hadm_id pertain to the same hospitalization.

    admittime, dischtime, deathtime
    admittime provides the date and time the patient was admitted to the hospital, while dischtime provides the date and time the patient was discharged from the hospital. If applicable, deathtime provides the time of in-hospital death for the patient. Note that deathtime is only present if the patient died in-hospital, and is almost always the same as the patient’s dischtime. However, there can be some discrepancies due to typographical errors.

    admission_type
    admission_type is useful for classifying the urgency of the admission. There are 9 possibilities: ‘AMBULATORY OBSERVATION’, ‘DIRECT EMER.’, ‘DIRECT OBSERVATION’, ‘ELECTIVE’, ‘EU OBSERVATION’, ‘EW EMER.’, ‘OBSERVATION ADMIT’, ‘SURGICAL SAME DAY ADMISSION’, ‘URGENT’.

    admit_provider_id
    admit_provider_id provides an anonymous identifier for the provider who admitted the patient. Provider identifiers follow a consistent pattern: the letter “P”, followed by either three numbers, followed by two letters or two numbers. For example, “P003AB”, “P00102”, “P1248B”, etc. Provider identifiers are randomly generated and do not have any inherent meaning aside from uniquely identifying the same provider across the database.

    admission_location, discharge_location
    admission_location provides information about the location of the patient prior to arriving at the hospital. Note that as the emergency room is technically a clinic, patients who are admitted via the emergency room usually have it as their admission location.

    Similarly, discharge_location is the disposition of the patient after they are discharged from the hospital.

    Association with UB-04 billing codes
    admission_location and discharge_location are associated with internal hospital ibax codes which aren’t provided in MIMIC-IV. These internal codes tend to align with UB-04 billing codes.

    In some cases more than one internal code is associated with a given admission_location and discharge_location. This can either be do to; 1) multiple codes being used by the hospital for the same admission_location or discharge_location, or 2) during de-identification multiple internal codes may be combined into a single admission_location or discharge_location.

    insurance, language, marital_status, ethnicity
    The insurance, language, marital_status, and ethnicity columns provide information about patient demographics for the given hospitalization. Note that as this data is documented for each hospital admission, they may change from stay to stay.

    edregtime, edouttime
    The date and time at which the patient was registered and discharged from the emergency department.

    hospital_expire_flag
    This is a binary flag which indicates whether the patient died within the given hospitalization. 1 indicates death in the hospital, and 0 indicates survival to hospital discharge.
    """
    transfers_description = """Physical locations for patients throughout their hospital stay. subject_id, hadm_id, transfer_id
subject_id
subject_id is a unique identifier which specifies an individual patient. Any rows associated with a single subject_id pertain to the same individual. As subject_id is the primary key for the table, it is unique for each row.

hadm_id
hadm_id is a unique identifier which specifies an individual hospitalization. Any rows associated with a single hadm_id pertain to the same hospitalization.
    
    transfer_id
    transfer_id is unique to a patient physical location. Represented as unique identifier for each transfer event.

Note that stay_id present in the icustays and edstays tables is derived from transfer_id. For example, three contiguous ICU stays will have three separate transfer_id for each distinct physical location (e.g. a patient could move from one bed to another). The entire stay will have a single stay_id, whih will be equal to the transfer_id of the first physical location.

eventtype
eventtype describes what transfer event occurred: ‘ed’ for an emergency department stay, ‘admit’ for an admission to the hospital, ‘transfer’ for an intra-hospital transfer and ‘discharge’ for a discharge from the hospital.

careunit
The type of unit or ward in which the patient is physically located. Examples of care units include medical ICUs, surgical ICUs, medical wards, new baby nurseries, and so on.

intime, outtime
intime provides the date and time the patient was transferred into the current care unit (careunit) from the previous care unit. outtime provides the date and time the patient was transferred out of the current physical location."""
    hcpsevents_description =   """
The hcpsevents table records specific events that occur during a patient's hospitalization, particularly focusing on those that are billable. Each event is documented using the Healthcare Common Procedure Coding System (HCPCS) codes, which provide a standardized method for reporting medical, surgical, and diagnostic services. The table includes information about the event, the patient, and a brief description of the service provided.
subject_id: A unique identifier for the patient. This field links the event to a specific patient within the hospital system.
hadm_id: The hospital admission ID. This field associates the event with a particular hospital admission instance, indicating when the patient was hospitalized.
hcpcs_cd: The HCPCS code for the event. This code specifies the exact procedure or service performed, facilitating standardized billing and documentation.
seq_num: The sequence number of the event. This field indicates the order in which events occurred during the hospital stay.
short_description: A brief description of the service or procedure provided. This description offers a quick reference to the type of event recorded."""
    services_description = """The Services table represents the type of patient admissions, detailing the specific service under which patients were admitted during their hospital stay. This table captures transitions between different services, providing a clear record of the patient's care journey within the hospital.

Data Fields:
subject_id: A unique identifier for the patient. This field links the service record to a specific patient within the hospital system.
hadm_id: The hospital admission ID. This field associates the service record with a particular hospital admission instance.
transfertime: The exact date and time when the patient was transferred to the current service. This timestamp helps track when transitions between services occurred.
prev_service: The service the patient was under before the transfer. This field is empty if the patient was not transferred from another service.
curr_service: The current service under which the patient is admitted. This field indicates the type of care or specialty (e.g., MED for medical, TRAUM for trauma)."""
    inputevents_description = """     
The inputevents table stores data related to the direct application of prescribed medications. This can include administration by injection, inhalation, ingestion, or other means. The table captures detailed records of the administration process, including the timing, dosage, and method of delivery. This ensures accurate tracking of medication administration for patient care and monitoring purposes.

Data Stored in the Table:
subject_id: Unique identifier for the patient.
hadm_id: Hospital admission ID, linking to a specific hospital stay.
stay_id: Unique identifier for the ICU stay.
starttime: Start time of the medication administration.
endtime: End time of the medication administration.
storetime: Time when the administration event was recorded.
itemid: Identifier for the medication or item administered.
amount: Amount of medication administered.
amountuom: Unit of measure for the amount (e.g., ml, mg).
rate: Rate at which the medication is administered.
rateuom: Unit of measure for the rate (e.g., ml/hr).
orderid: Identifier for the medication order.
linkorderid: Identifier linking to the related medication order.
ordercategoryname: Category name of the order.
secondaryordercategoryname: Secondary category name for more detailed classification.
ordercomponenttypedescription: Description of the order component type.
ordercategorydescription: Description of the order category.
patientweight: Patient's weight, potentially used to calculate dosage.
totalamount: Total amount of medication administered.
totalamountuom: Unit of measure for the total amount.
isopenbag: Indicator if the medication was administered from an open bag.
continueinnextdept: Indicator if the medication administration continues in the next department.
cancelreason: Reason for cancellation if the administration was canceled.
statusdescription: Description of the administration status.
originalamount: Original amount of medication ordered.
originalrate: Original rate at which the medication was to be administered. """
    diagnoses_icd_description = """
The diagnoses_icd table stores data related to the condition, problem, diagnosis, or other event using their respective ICD (International Classification of Diseases) codes. The data is used for billing purposes, clinical documentation, and health statistics. The diagnoses are assigned by trained medical professionals who review clinical notes and determine the appropriate ICD codes. The table helps in tracking the medical conditions treated, which is essential for billing, medical record-keeping, and statistical analysis.
Data Stored in the Table:
subject_id: Unique identifier for the patient.
hadm_id: Hospital admission ID, linking to a specific hospital stay.
seq_num: Sequence number indicating the order of diagnoses for the given hospital admission.
icd_code: ICD code representing the diagnosis.
icd_version: Version of the ICD code (e.g., 9 for ICD-9, 10 for ICD-10)."""
    prescriptions_description = """
The prescriptions table is designed to manage and document comprehensive medication orders for patients. It facilitates precise documentation and communication of medication-related interventions across various clinical settings, ensuring continuity and quality of care.

Data Fields:

subject_id: A unique identifier for the patient receiving the medication.
hadm_id: The hospital admission ID associated with the medication order.
icustay_id: The ICU stay ID, if applicable, linking the prescription to a specific ICU stay.
startdate: The date and time when the medication order starts.
enddate: The date and time when the medication order ends.
drug_type: The type or category of the medication ordered.
drug: The name of the medication as specified in the order.
drug_name_poe: The name of the medication as entered in the physician order entry system.
drug_name_generic: The generic name of the medication.
formulary_drug_cd: The formulary drug code for the medication.
gsn: The Generic Sequence Number, a unique identifier for the drug.
ndc: The National Drug Code, a unique identifier for the medication product.
prod_strength: The strength of the medication product.
dose_val_rx: The prescribed dose value of the medication.
dose_unit_rx: The unit of measurement for the prescribed dose.
form_val_disp: The dispensed form value of the medication.
form_unit_disp: The unit of measurement for the dispensed form.
route: The route of administration for the medication (e.g., oral, intravenous)."""
    outputevents_description = """  
subject_id: Each row in the table corresponds to a unique individual patient identified by subject_id. This identifier ensures that all records associated with a specific subject_id pertain to the same individual.

hadm_id: An identifier that uniquely specifies a hospitalization. Rows sharing the same hadm_id relate to the same hospital admission, providing contextual information about where and when the observations were made.

stay_id: This identifier groups reasonably contiguous episodes of care within the hospitalization. It helps organize related observations that occur during a patient's stay, ensuring coherence in the data captured over the course of care.

charttime: Indicates the time of an observation event, capturing when specific data points were recorded. This temporal aspect is crucial for understanding the context and timing of each observation within the patient's healthcare journey.

storetime: Records the time when an observation was manually input or validated by clinical staff. This timestamp provides metadata about the data entry or validation process, offering insights into the handling and verification of clinical observations.

itemid: An identifier for a specific measurement type or observation in the database. Each itemid corresponds to a distinct measurement or observation recorded, such as heart rate or blood glucose level, ensuring clarity and categorization of clinical data.

value, valueuom: Describes the actual measurement or observation value at the charttime, along with its unit of measurement (valueuom). These fields provide quantitative or qualitative data points, such as numeric results (value) and the corresponding units, offering precise details about each recorded observation.
"""
    pharmacy_description = """ 

The pharmacy table serves as a comprehensive repository for data related to medication dispensation to patients within a healthcare facility. It meticulously records details about each medication filled, including dosage, frequency, route of administration, and the duration prescribed. This table captures the entire dispensing process from the initial prescription through to the final dispensation, ensuring thorough documentation and traceability of medication administration within the hospital system.

The pharmacy table's focus lies in the operational details of medication dispensation such as start and stop times, medication statuses, and specific procedural types related to medication administration. It links each record to patient identifiers, hospital admission IDs, and unique pharmacy IDs, providing a structured view of medication management from the perspective of dispensing logistics and patient-specific medication details.

This table supports healthcare providers in managing complex medication regimens, ensuring that medications are dispensed accurately and efficiently. It includes tracking and documentation of each step involved in the medication dispensing process, from preparation and packaging to authorization and verification, aligning closely with regulatory standards and supporting patient safety and effective treatment outcomes.

Key Data Fields:

subject_id: A unique identifier for the patient. This field links the pharmacy record to a specific patient within the hospital system.
hadm_id: The hospital admission ID. This field associates the pharmacy record with a particular hospital admission instance.
pharmacy_id: A unique identifier for the pharmacy record.
poe_id: The physician order entry ID, linking to the specific order entry instance for the medication.
starttime: The start time for the medication administration.
stoptime: The stop time for the medication administration.
medication: The name of the medication dispensed.
proc_type: The type of procedure related to the medication (e.g., IV Large Volume, Unit Dose).
status: The status of the medication (e.g., Discontinued, Active).
entertime: The time when the medication was entered into the system.
verifiedtime: The time when the medication was verified.
route: The route of administration for the medication (e.g., IV, IM).
frequency: The frequency of dosing (e.g., Q8H, ASDIR).
disp_sched: The dispensation schedule, if applicable.
infusion_type: The type of infusion, if applicable.
sliding_scale: Indicates if a sliding scale dosing was used.
lockout_interval: The lockout interval for medication administration, if applicable.
basal_rate: The basal rate of administration, if applicable.
one_hr_max: The maximum dose allowed within one hour, if applicable.
doses_per_24_hrs: The total number of doses allowed within 24 hours, if applicable.
duration: The duration for which the medication is prescribed.
duration_interval: The unit of time for the duration (e.g., Hours, Days).
expiration_value: The expiration value of the medication.
expiration_unit: The unit of time for the expiration (e.g., Hours, Days).
expirationdate: The expiration date of the medication.
dispensation: Information about the dispensation of the medication.
fill_quantity: The quantity of the medication filled.


"""

    d_items_description = """
The D_ITEMS table stores metadata related to various medical observations and measurements recorded in the database. Each record in the table provides detailed information about a specific measurement or observation type, facilitating the accurate and consistent recording of clinical data across different events and patient encounters.

The table includes columns that describe each measurement type, including its label, abbreviation, and category, as well as the unit of measurement and normal value ranges where applicable. This structured metadata ensures that medical observations are systematically categorized and can be accurately interpreted and utilized for clinical review, research, and analysis.

Data Stored in the Table:
itemid: A unique identifier for each type of measurement or observation. Each itemid is greater than 220000.
label: The full name or description of the measurement (e.g., "Heart Rate").
abbreviation: A shorter form or acronym of the measurement name (e.g., "HR").
linksto: Indicates the table where the actual measurement data is stored (e.g., "chartevents").
category: The general category to which the measurement belongs (e.g., "Routine Vital Signs").
unitname: The unit of measurement for the observation (e.g., "bpm" for beats per minute).
param_type: The type of parameter, such as Numeric or Categorical, indicating the data format of the measurement.
lownormalvalue: The lower bound of the normal range for the measurement, if applicable.
highnormalvalue: The upper bound of the normal range for the measurement, if applicable.
"""
    emar_description = """ 
The emar table stores data related to medication management, ensuring the safe and effective use of medications through accurate identification and comprehensive drug information. This table captures detailed records of medication administration, including the timing, dosage, and method of delivery. The data is recorded in real-time by nursing staff through barcode scanning, which helps in verifying the correct medication is administered to the right patient.

Data Stored in the Table:
subject_id: Unique identifier for the patient.
hadm_id: Hospital admission ID, linking to a specific hospital stay.
stay_id: Unique identifier for the ICU stay.
starttime: Start time of the medication administration.
endtime: End time of the medication administration.
storetime: Time when the administration event was recorded.
itemid: Identifier for the medication or item administered.
amount: Amount of medication administered.
amountuom: Unit of measure for the amount (e.g., ml, mg).
rate: Rate at which the medication is administered (if applicable).
rateuom: Unit of measure for the rate (e.g., ml/hr).
orderid: Identifier for the medication order.
linkorderid: Identifier linking to the related medication order.
ordercategoryname: Category name of the order.
secondaryordercategoryname: Secondary category name for more detailed classification.
ordercomponenttypedescription: Description of the order component type.
ordercategorydescription: Description of the order category.
patientweight: Patient's weight, potentially used to calculate dosage.
totalamount: Total amount of medication administered.
totalamountuom: Unit of measure for the total amount.
isopenbag: Indicator if the medication was administered from an open bag.
continueinnextdept: Indicator if the medication administration continues in the next department.
cancelreason: Reason for cancellation if the administration was canceled.
statusdescription: Description of the administration status.
originalamount: Original amount of medication ordered.
originalrate: Original rate at which the medication was to be administered.
"""
    datetimeevents_description = """

The datetimeevents table captures all date and time measurements documented for patients in the ICU, representing observations in datetime form. For instance, the date of the last dialysis session would be recorded in the datetimeevents table, while a measurement such as systolic blood pressure would not. Each record in this table corresponds to a specific datetime event observed and documented during a patient's ICU stay. This includes events such as the start or end of a medical procedure, the timing of specific medical observations, or other significant datetime-related information pertinent to patient care.

The datetimeevents table provides detailed temporal records that can be used for clinical review, research, and analysis. These records support the monitoring of patient progress, the determination of baselines and patterns, and the overall management of patient care. The table plays a crucial role in maintaining a precise and comprehensive timeline of a patient's ICU stay, enabling healthcare professionals to track and analyze the timing of critical events accurately. This information is essential for ensuring high-quality patient care, facilitating effective clinical decision-making, and supporting ongoing medical research and quality improvement initiatives. 

Data Stored in the Table:
subject_id: Unique identifier for the patient.
hadm_id: Hospital admission ID, linking to a specific hospital stay.
stay_id: ICU stay ID, linking to a specific ICU admission.
charttime: The time when the observation was charted.
storetime: The time when the observation was stored in the database.
itemid: Identifier for the type of observation or event.
value: The datetime value of the observed event.
valueuom: Unit of measurement for the value (typically "Date" for datetime events).
warning: Indicator for any warnings or flags associated with the observation.
 """
    microbiologyevents_description = """
The "microbiologyevents" table is instrumental in healthcare for tracking and analyzing microbiological data from cultures and antibiotic sensitivity tests performed during hospital admissions. It allows healthcare professionals to monitor infection trends, assess the efficacy of antibiotic therapies, and make informed treatment decisions based on microbial findings. Researchers utilize this data to investigate patterns of microbial resistance and conduct epidemiological studies, contributing to the development of effective infection control strategies. 
Data Fields:

subject_id: Identifier of the subject (patient) associated with the microbiological observation event.
hadm_id: Identifier of the hospital admission during which the microbiological observation event occurred.
chartdate: Date of charting the microbiological observation event.
charttime: Time of charting the microbiological observation event.
spec_itemid: Identifier of the specimen item tested.
spec_type_desc: Description of the type of specimen tested (e.g., blood culture, urine).
org_itemid: Identifier of the organism item detected.
org_name: Name of the organism detected during the microbiological observation test.
isolate_num: Numerical identifier of the isolate if multiple isolates are identified.
ab_itemid: Identifier of the antibiotic item tested against.
ab_name: Name of the antibiotic tested.
dilution_text: Text representation of dilution result.
dilution_comparison: Comparison operator used in dilution result (e.g., <=, =).
dilution_value: Numeric value of the dilution result.
interpretation: Interpretation of the microbiological  test result (e.g., Sensitive, Resistant)."""
    procedures_icd_description = """
    
The procedures_icd table stores data representing the various procedures performed on patients during their hospital stay. Each procedure is documented using standardized ICD (International Classification of Diseases) codes, specifically ICD-9 and ICD-10, which are widely used for coding and billing medical procedures.

subject_id: A unique identifier for the patient. This field links the procedure record to a specific patient within the hospital system.
hadm_id: The hospital admission ID. This field associates the procedure record with a particular hospital admission instance.
seq_num: The sequence number of the procedure. This field indicates the order in which procedures were performed during the hospital stay.
icd_code: The ICD code for the procedure. This field contains the specific code that identifies the medical procedure performed.
icd_version: The version of the ICD code (9 or 10) used to document the procedure. This field specifies whether the code is from the ICD-9 or ICD-10 ontology. """
    icustays_description =""" Detailed Description
The Icustays table contains detailed records of patient admissions to the Intensive Care Unit (ICU). Each record represents a unique ICU stay, providing comprehensive information about the patient's ICU admission and discharge times. This table is essential for tracking patient movements within the ICU, the duration of their stay, and the specific care units involved.

subject_id: A unique identifier for the patient. This field links the ICU stay to a specific patient within the hospital system.
hadm_id: The hospital admission ID. This field associates the ICU stay with a particular hospital admission instance.
stay_id: A unique identifier for the ICU stay. This field distinguishes each ICU admission event for a patient.
first_careunit: The initial ICU care unit where the patient was admitted (e.g., Coronary Care Unit (CCU), Trauma SICU (TSICU)).
last_careunit: The last ICU care unit where the patient was treated before discharge.
intime: The exact date and time when the patient was admitted to the ICU.
outtime: The exact date and time when the patient was discharged from the ICU.
los: Length of stay in the ICU, measured in days. This field helps in understanding the duration of care provided in the ICU."""
    patient_description= """ subject_id
subject_id is a unique identifier which specifies an individual patient. Any rows associated with a single subject_id pertain to the same individual. As subject_id is the primary key for the table, it is unique for each row.

gender
gender is the genotypical sex of the patient.

anchor_age, anchor_year, anchor_year_group
These columns provide information regarding the actual patient year for the patient admission, and the patient’s age at that time.

anchor_year is a shifted year for the patient.
anchor_year_group is a range of years - the patient’s anchor_year occurred during this range.
anchor_age is the patient’s age in the anchor_year. If a patient’s anchor_age is over 89 in the anchor_year then their anchor_age is set to 91, regardless of how old they actually were.
Example: a patient has an anchor_year of 2153, anchor_year_group of 2008 - 2010, and an anchor_age of 60.
The year 2153 for the patient corresponds to 2008, 2009, or 2010.
The patient was 60 in the shifted year of 2153, i.e. they were 60 in 2008, 2009, or 2010.
A patient admission in 2154 will occur in 2009-2011, an admission in 2155 will occur in 2010-2012, and so on.
dod
The de-identified date of death for the patient. Date of death is extracted from two sources: the hospital information system and the Massachusetts State Registry of Vital Records and Statistics. Individual patient records from MIMIC were matched to the vital records using a custom algorithm based on identifiers including name, social security number, and date of birth.

As a result of the linkage, out of hospital mortality is available for MIMIC-IV patients up to one year post-hospital discharge. All patient deaths occurring more than one year after hospital discharge are censored. Survival studies should incorporate this into their design."""
    procedureevents_description = """ 
The procedureevents table stores data representing various procedure events that a patient undergoes during their ICU stay. Each procedure in the table provides detailed information, including timing, type, and related specifics.

Data Fields:
subject_id: A unique identifier for the patient. This field links the procedural event to a specific patient within the hospital system.
hadm_id: The hospital admission ID. This field associates the procedural event with a particular hospital admission instance.
stay_id: A unique identifier for the ICU stay. This field distinguishes each ICU admission event for a patient.
starttime: The date and time when the procedure started. This timestamp helps track the initiation of the procedure.
endtime: The date and time when the procedure ended. This timestamp helps track the completion of the procedure.
storetime: The date and time when the procedure was recorded in the system.
itemid: A unique identifier for the specific procedure performed.
value: The numerical value associated with the procedure, if applicable.
valueuom: The unit of measurement for the value, if applicable.
location: The specific location where the procedure took place.
locationcategory: The category of the location where the procedure was performed.
orderid: A unique identifier for the order associated with the procedure.
linkorderid: A unique identifier linking related orders.
ordercategoryname: The name of the category to which the order belongs.
secondaryordercategoryname: The name of the secondary category to which the order belongs.
ordercategorydescription: A description of the order category.
patientweight: The weight of the patient at the time of the procedure.
totalamount: The total amount related to the procedure, if applicable.
totalamountuom: The unit of measurement for the total amount, if applicable.
isopenbag: Indicates whether an open bag was used for the procedure.
continueinnextdept: Indicates if the procedure will continue in the next department.
cancelreason: The reason for canceling the procedure, if applicable.
statusdescription: A description of the status of the procedure.
comments_date: The date of any comments related to the procedure.
originalamount: The original amount associated with the procedure.
originalrate: The original rate associated with the procedure."""

    if table == "admissions":
        return admissions_description
    elif table == "transfers":
        return transfers_description
    elif table == "hcpsevents":
        return hcpsevents_description
    elif table == "services":
        return services_description
    elif table == "inputevents":
        return inputevents_description
    elif table == "diagnoses_icd":
        return diagnoses_icd_description
    elif table == "prescriptions":
        return prescriptions_description
    elif table == "outputevents":
        return outputevents_description
    elif table == "pharmacy":
        return pharmacy_description
    elif table == "emar":
        return emar_description
    elif table == "d_items":
        return d_items_description
    elif table == "datetimeevents":
        return datetimeevents_description
    elif table == "microbiologyevents":
        return microbiologyevents_description
    elif table == "procedureevents":
        return procedureevents_description
    elif table == "procedures_icd":
        return procedures_icd_description
    elif table == "icustays":
        return icustays_description
    elif table == "patients":
        return patient_description
    


def load_data(path):
    #Cargar los datos tabulares desde un archivo CSV
    data = pd.read_csv(path)
    #Se convierte la data a formato JSON
    data_json = data.to_json(orient='records')
    return data_json
#------------------------------------------------------------

# Se crea la consulta
filename = "diagnoses_icd"
data = load_data("Fase3/" + filename + "_mock.csv")  # Cargar los datos tabulares desde un archivo CSV

desc = get_description(filename).lower()  # Obtener y convertir la descripción a minúsculas
# Crear la consulta con los datos tabulares y la descripción
query = f"The tabular data I would like to know the columns mapping is : {data}, this table is described as {desc}. Tell me to which FHIR resource item each column belongs."

#------------------------------------------------------------

# Se crea el índice y el corpus.
texts = load_dataset('Fase3/datasetRecursos.ndjson')  # Cargar el conjunto de datos desde un archivo NDJSON
corpus = create_corpus(texts)  # Crear el corpus a partir del conjunto de datos

# Calcular las similitudes entre el corpus y la consulta usando diferentes técnicas
vec_dict_similarity = tfidf_faiss_similarity(corpus, query)  # Similitud con TF-IDF y FAISS
bm25_similarity_dict = bm25_similarity(corpus, query)  # Similitud con BM25
# print("Tr")
# transformers_similarity_dict = transformers_similarity(corpus, query)  # Similitud con Transformers
# print("Univ")
univ_sent_encoder_similarity = univ_sent_encoder(corpus, query)  # Similitud con Universal Sentence Encoder
fasttext_similarities = fasttext_similarity(corpus, query)  # Similitud con FastText

#------------------------------------------------------------
top_k = 1  # Número de textos más relevantes a obtener
context = get_top_texts(vec_dict_similarity, bm25_similarity_dict, univ_sent_encoder_similarity, fasttext_similarities, corpus, top_k)  # Obtener los textos más relevantes

# Se obtiene el contexto específico para la consulta
for i in range(5):  # Realizar el proceso de generación de respuesta 5 veces
    print("---------------------------------", i)
    print("------------------------LLAMA RESPONSE------------------------------------")
    # Se crea la respuesta usando el modelo Llama
    response = generate_response(query, "Llama", context)
    print(response)  # Imprimir la respuesta generada por Llama
    print("------------------------GPT RESPONSE---------------------------------------")
    # Se crea la respuesta usando el modelo GPT
    response = generate_response(query, "GPT", context)
    print(response)  # Imprimir la respuesta generada por GPT
    #print("---------------------------------------------------------------------------")