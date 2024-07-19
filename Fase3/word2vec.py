import json
import tensorflow as tf
import tensorflow_hub as hub
from scipy.spatial.distance import cosine
import pandas as pd

# Cargar el modelo de Universal Sentence Encoder
def load_univ_sent_encoder_model():
    model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(model_url)
    return model

# Función para obtener el embedding de un texto
def embed_text(text):
    return model(text)

def get_description(table):

    admissions_description = """ 
    The admissions table gives information regarding a patient’s admission to the hospital. Since each unique hospital visit for a patient is assigned a unique hadm_id, the admissions table can be considered as a definition table for hadm_id. Information available includes timing information for admission and discharge, demographic information, the source of the admission, and so on.
    subject_id, hadm_id
    Each row of this table contains a unique hadm_id, which represents a single patient’s admission to the hospital. hadm_id ranges from 2000000 - 2999999. It is possible for this table to have duplicate subject_id, indicating that a single patient had multiple admissions to the hospital. The ADMISSIONS table can be linked to the PATIENTS table using subject_id.

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
Identifiers which specify the patient: subject_id is unique to a patient, hadm_id is unique to a patient hospital stay, and transfer_id is unique to a patient physical location.

Note that stay_id present in the icustays and edstays tables is derived from transfer_id. For example, three contiguous ICU stays will have three separate transfer_id for each distinct physical location (e.g. a patient could move from one bed to another). The entire stay will have a single stay_id, whih will be equal to the transfer_id of the first physical location.

eventtype
eventtype describes what transfer event occurred: ‘ed’ for an emergency department stay, ‘admit’ for an admission to the hospital, ‘transfer’ for an intra-hospital transfer and ‘discharge’ for a discharge from the hospital.

careunit
The type of unit or ward in which the patient is physically located. Examples of care units include medical ICUs, surgical ICUs, medical wards, new baby nurseries, and so on.

intime, outtime
intime provides the date and time the patient was transferred into the current care unit (careunit) from the previous care unit. outtime provides the date and time the patient was transferred out of the current physical location."""
    hcpsevents_description = """
    Table information: Billed events occurring during the hospitalization. Includes CPT codes. 
    Columns information:
    subject_id
    subject_id is a unique identifier which specifies an individual patient. Any rows associated with a single subject_id pertain to the same individual.

    hadm_id
    hadm_id is an integer identifier which is unique for each patient hospitalization.

    chartdate
    The date associated with the coded event.

    hcpcs_cd
    A five character code which uniquely represents the event. Link this to code in d_hcpcs for a longer description of the code.

    seq_num
    An assigned order to HCPCS codes for an individual hospitalization. This order sometimes conveys meaning, e.g. sometimes higher priority, but this is not guaranteed across all codes.

    short_description
    A short textual descriptions of the hcpcs_cd listed for the given row. """
    services_description = """The services table describes the service that a patient was admitted under. While a patient can be physicially located at a given ICU type (say MICU), they are not necessarily being cared for by the team which staffs the MICU. This can happen due to a number of reasons, including bed shortage. The services table should be used if interested in identifying the type of service a patient is receiving in the hospital. For example, if interested in identifying surgical patients, the recommended method is searching for patients admitted under a surgical service. 
    subject_id 
    subject_id is a unique identifier which specifies an individual patient. Any rows associated with a single subject_id pertain to the same individual.

    hadm_id
    hadm_id is an integer identifier which is unique for each patient hospitalization.

    transfertime
    transfertime is the time at which the patient moved from the prev_service (if present) to the curr_service.

    prev_service, curr_service
    prev_service and curr_service are the previous and current service that the patient resides under."""
    inputevents_description = """ Advanced description: subject_id, hadm_id, stay_id
    Identifiers which specify the patient: subject_id is unique to a patient, hadm_id is unique to a patient hospital stay and stay_id is unique to a patient ICU stay.

    caregiver_id
    caregiver_id uniquely identifies a single caregiver who documented data in the ICU information system.

    starttime, endtime
    starttime and endtime record the start and end time of an input/output event.

    storetime
    storetime records the time at which an observation was manually input or manually validated by a member of the clinical staff.

    itemid
    Identifier for a single measurement type in the database. Each row associated with one itemid which corresponds to an instantiation of the same measurement (e.g. norepinephrine).

    amount, amountuom
    amount and amountuom list the amount of a drug or substance administered to the patient either between the starttime and endtime.

    rate, rateuom
    rate and rateuom list the rate at which the drug or substance was administered to the patient either between the starttime and endtime.

    orderid, linkorderid
    orderid links multiple items contained in the same solution together. For example, when a solution of noradrenaline and normal saline is administered both noradrenaline and normal saline occur on distinct rows but will have the same orderid.

    linkorderid links the same order across multiple instantiations: for example, if the rate of delivery for the solution with noradrenaline and normal saline is changed, two new rows which share the same new orderid will be generated, but the linkorderid will be the same.

    ordercategoryname, secondaryordercategoryname, ordercomponenttypedescription, ordercategorydescription
    These columns provide higher level information about the order the medication/solution is a part of. Categories represent the type of administration, while the ordercomponenttypedescription describes the role of the substance in the solution (i.e. main order parameter, additive, or mixed solution)

    patientweight
    The patient weight in kilograms.

    totalamount, totalamountuom
    Intravenous administrations are usually given by hanging a bag of fluid at the bedside for continuous infusion over a certain period of time. These columns list the total amount of the fluid in the bag containing the solution.

    isopenbag
    Whether the order was from an open bag.

    continueinnextdept
    If the order ended on patient transfer, this field indicates if it continued into the next department (e.g. a floor).

    statusdescription
    statusdescription states the ultimate status of the item, or more specifically, row. It is used to indicate why the delivery of the compound has ended. There are only six possible statuses:

    Changed - The current delivery has ended as some aspect of it has changed (most frequently, the rate has been changed)
    Paused - The current delivery has been paused
    FinishedRunning - The delivery of the item has finished (most frequently, the bag containing the compound is empty)
    Stopped - The delivery of the item been terminated by the caregiver
    Flushed - A line was flushed.
    originalamount
    Drugs are usually mixed within a solution and delivered continuously from the same bag. This column represents the amount of the drug contained in the bag at starttime. For the first infusion of a new bag, originalamount: totalamount. Later on, if the rate is changed, then the amount of the drug in the bag will be lower (as some has been administered to the patient). As a result, originalamount < totalamount, and originalamount will be the amount of drug leftover in the bag at that starttime.

    originalrate
    This is the rate that was input by the care provider. Note that this may differ from rate because of various reasons: originalrate was the original planned rate, while the rate column will be the true rate delivered. For example, if a a bag is about to run out and the care giver decides to push the rest of the fluid, then rate > originalrate. However, these two columns are usually the same, but have minor non-clinically significant differences due to rounding error."""
    diagnoses_icd_description = """
    During routine hospital care, patients are billed by the hospital for diagnoses associated with their hospital stay. This table contains a record of all diagnoses a patient was billed for during their hospital stay using the ICD-9 and ICD-10 ontologies. Diagnoses are billed on hospital discharge, and are determined by trained persons who read signed clinical notes. 
    Detailed Description
    subject_id
    subject_id is a unique identifier which specifies an individual patient. Any rows associated with a single subject_id pertain to the same individual.

    hadm_id
    hadm_id is an integer identifier which is unique for each patient hospitalization.

    seq_num
    The priority assigned to the diagnoses. The priority can be interpreted as a ranking of which diagnoses are “important”, but many caveats to this broad statement exist. For example, patients who are diagnosed with sepsis must have sepsis as their 2nd billed condition. The 1st billed condition must be the infectious agent. There’s also less importance placed on ranking low priority diagnoses “correctly” (as there may be no correct ordering of the priority of the 5th - 10th diagnosis codes, for example).

    icd_code, icd_version
    icd_code is the International Coding Definitions (ICD) code.

    There are two versions for this coding system: version 9 (ICD-9) and version 10 (ICD-10). These can be differentiated using the icd_version column. In general, ICD-10 codes are more detailed, though code mappings (or “cross-walks”) exist which convert ICD-9 codes to ICD-10 codes.

    Both ICD-9 and ICD-10 codes are often presented with a decimal. This decimal is not required for interpretation of an ICD code; i.e. the icd_code of ‘0010’ is equivalent to ‘001.0’."""
    prescriptions_description = """ The prescriptions table provides information about prescribed medications. Information includes the name of the drug, coded identifiers including the Generic Sequence Number (GSN) and National Drug Code (NDC), the product strength, the formulary dose, and the route of administration. subject_id
subject_id is a unique identifier which specifies an individual patient. Any rows associated with a single subject_id pertain to the same individual.

hadm_id
hadm_id is an integer identifier which is unique for each patient hospitalization.

pharmacy_id
An identifier which links administrations in emar to pharmacy information in the pharmacy table.

poe_id, poe_seq
These columns allow linking prescriptions to associated orders in the poe table.

order_provider_id
order_provider_id provides an anonymous identifier for the provider who initiated the order. Provider identifiers follow a consistent pattern: the letter “P”, followed by either three numbers, followed by two letters or two numbers. For example, “P003AB”, “P00102”, “P1248B”, etc. Provider identifiers are randomly generated and do not have any inherent meaning aside from uniquely identifying the same provider across the database.

starttime, stoptime
The prescribed start and stop time for the medication.

drug_type
The component of the prescription which the drug occupies. Can be one of ‘MAIN’, ‘BASE’, or ‘ADDITIVE’.

drug
A free-text description of the medication administered.

formulary_drug_cd
A hospital specific ontology used to order drugs from the formulary.

gsn
The Generic Sequence Number (GSN), a coded identifier used for medications.

ndc
The National Drug Code (NDC), a coded identifier which uniquely identifiers medications.

prod_strength
A free-text description of the composition of the prescribed medication (e.g. ‘12 mg / 0.8 mL Oral Syringe’, ‘12.5mg Tablet’, etc).

form_rx
The container in which the formulary dose is delivered (e.g. ‘TABLET’, ‘VIAL’, etc).

dose_val_rx
The prescribed dose for the patient intended to be administered over the given time period.

dose_unit_rx
The unit of measurement for the dose.

form_val_disp
The amount of the medication which is contained in a single formulary dose.

form_unit_disp
The unit of measurement used for the formulary dosage.

doses_per_24_hrs
The number of doses per 24 hours for which the medication is to be given. A daily dose would result in doses_per_24_hrs: 1, bidaily (BID) would be 2, and so on.

route 
The route of administration for the medication."""
    outputevents_description = """ Detailed Description
subject_id, hadm_id, stay_id
Identifiers which specify the patient: subject_id is unique to a patient, hadm_id is unique to a patient hospital stay and stay_id is unique to a patient ICU stay.

caregiver_id
caregiver_id uniquely identifies a single caregiver who documented data in the ICU information system.

charttime
charttime is the time of an output event.

storetime
storetime records the time at which an observation was manually input or manually validated by a member of the clinical staff.

itemid
Identifier for a single measurement type in the database. Each row associated with one itemid (e.g. 212) corresponds to an instantiation of the same measurement (e.g. heart rate).

value, valueuom
value and valueuom list the amount of a substance at the charttime (when the exact start time is unknown, but usually up to an hour before)."""
    pharmacy_description = """ The pharmacy table provides detailed information regarding filled medications which were prescribed to the patient. Pharmacy information includes the dose of the drug, the number of formulary doses, the frequency of dosing, the medication route, and the duration of the prescription.
subject_id
subject_id is a unique identifier which specifies an individual patient. Any rows associated with a single subject_id pertain to the same individual.

hadm_id
hadm_id is an integer identifier which is unique for each patient hospitalization.

pharmacy_id
A unique identifier for the given pharmacy entry. Each row of the pharmacy table has a unique pharmacy_id. This identifier can be used to link the pharmacy information to the provider order (in poe or prescriptions) or to the administration of the medication (in emar).

poe_id
A foreign key which links to the provider order entry order in the prescriptions table associated with this pharmacy record.

starttime, stoptime
The start and stop times for the given prescribed medication.

medication
The name of the medication provided.

proc_type
The type of order: “IV Piggyback”, “Non-formulary”, “Unit Dose”, and so on.

status
Whether the prescription is active, inactive, or discontinued.

entertime
The date and time at which the prescription was entered into the pharmacy system.

verifiedtime
The date and time at which the prescription was verified by a physician.

route
The intended route of administration for the prescription.

frequency
The frequency at which the medication should be administered to the patient. Many commonly used short hands are used in the frequency column. Q# indicates every # hours; e.g. “Q6” or “Q6H” is every 6 hours.

disp_sched
The hours of the day at which the medication should be administered, e.g. “08, 20” would indicate the medication should be administered at 8:00 AM and 8:00 PM, respectively.

infusion_type
A coded letter describing the type of infusion: ‘B’, ‘C’, ‘N’, ‘N1’, ‘O’, or ‘R’.

sliding_scale
Indicates whether the medication should be given on a sliding scale: either ‘Y’ or ‘N’.

lockout_interval
The time the patient must wait until providing themselves with another dose; often used with patient controlled analgesia.

basal_rate
The rate at which the medication is given over 24 hours.

one_hr_max
The maximum dose that may be given in a single hour.

doses_per_24_hrs
The number of expected doses per 24 hours. Note that this column can be misleading for continuously infused medications as they are usually only “dosed” once per day, despite continuous administration.

duration, duration_interval
duration is the numeric duration of the given dose, while duration_interval can be considered as the unit of measurement for the given duration. For example, often duration is 1 and duration_interval is “Doses”. Alternatively, duration could be 8 and the duration_interval could be “Weeks”.

expiration_value, expiration_unit, expirationdate
If the drug has a relevant expiry date, these columns detail when this occurs. expiration_value and expiration_unit provide a length of time until the drug expires, e.g. 30 days, 72 hours, and so on. expirationdate provides the deidentified date of expiry.

dispensation
The source of dispensation for the medication.

fill_quantity
What proportion of the formulary to fill.
"""
    d_items_description = """ Detailed Description
The D_ITEMS table defines itemid, which represents measurements in the database. Measurements of the same type (e.g. heart rate) will have the same itemid (e.g. 220045). Values in the itemid column are unique to each row. All itemids will have a value > 220000.

itemid
As an alternate primary key to the table, itemid is unique to each row.

label, abbreviation
The label column describes the concept which is represented by the itemid. The abbreviation column, only available in Metavision, lists a common abbreviation for the label.

linksto
linksto provides the table name which the data links to. For example, a value of ‘chartevents’ indicates that the itemid of the given row is contained in CHARTEVENTS. A single itemid is only used in one event table, that is, if an itemid is contained in CHARTEVENTS it will not be contained in any other event table (e.g. IOEVENTS, CHARTEVENTS, etc).

category
category provides some information of the type of data the itemid corresponds to. Examples include ‘ABG’, which indicates the measurement is sourced from an arterial blood gas, ‘IV Medication’, which indicates that the medication is administered through an intravenous line, and so on.

unitname
unitname specifies the unit of measurement used for the itemid. This column is not always available, and this may be because the unit of measurement varies, a unit of measurement does not make sense for the given data type, or the unit of measurement is simply missing. Note that there is sometimes additional information on the unit of measurement in the associated event table, e.g. the valueuom column in CHARTEVENTS.

param_type
param_type describes the type of data which is recorded: a date, a number or a text field.

lownormalvalue, highnormalvalue
These columns store reference ranges for the measurement. Note that a reference range encompasses the expected value of a measurement: values outside of this may still be physiologically plausible, but are considered unusual."""
    emar_description = """ The EMAR table is used to record administrations of a given medicine to an individual patient. Records in this table are populated by bedside nursing staff scanning barcodes associated with the medicine and the patient. subject_id
subject_id is a unique identifier which specifies an individual patient. Any rows associated with a single subject_id pertain to the same individual.

hadm_id
hadm_id is an integer identifier which is unique for each patient hospitalization.

emar_id, emar_seq
Identifiers for the eMAR table. emar_id is a unique identifier for each order made in eMAR. emar_seq is a consecutive integer which numbers eMAR orders chronologically. emar_id is composed of subject_id and emar_seq in the following pattern: ‘subject_id-emar_seq’.

poe_id
An identifier which links administrations in emar to orders in poe and prescriptions.

pharmacy_id
An identifier which links administrations in emar to pharmacy information in the pharmacy table.

enter_provider_id
enter_provider_id provides an anonymous identifier for the provider who entered the information into the eMAR system. Provider identifiers follow a consistent pattern: the letter “P”, followed by either three numbers, followed by two letters or two numbers. For example, “P003AB”, “P00102”, “P1248B”, etc. Provider identifiers are randomly generated and do not have any inherent meaning aside from uniquely identifying the same provider across the database.

charttime
The time at which the medication was administered.

medication
The name of the medication which was administered.

event_txt
Information about the administration. Most frequently event_txt is ‘Administered’, but other possible values are ‘Applied’, ‘Confirmed’, ‘Delayed’, ‘Not Given’, and so on.

scheduletime
If present, the time at which the administration was scheduled.

storetime
The time at which the administration was documented in the eMAR table."""
    datetimeevents_description = """Detailed Description
datetimeevents contains all date measurements about a patient in the ICU. For example, the date of last dialysis would be in the datetimeevents table, but the systolic blood pressure would not be in this table. As all dates in MIMIC are anonymized to protect patient confidentiality, all dates in this table have been shifted. Note that the chronology for an individual patient has been unaffected however, and quantities such as the difference between two dates remain true to reality.

subject_id, hadm_id, stay_id
Identifiers which specify the patient: subject_id is unique to a patient, hadm_id is unique to a patient hospital stay and stay_id is unique to a patient ward stay.

caregiver_id
caregiver_id uniquely identifies a single caregiver who documented data in the ICU information system.

charttime, storetime
charttime records the time at which an observation was charted, and is usually the closest proxy to the time the data was actually measured. storetime records the time at which an observation was manually input or manually validated by a member of the clinical staff.

itemid
Identifier for a single measurement type in the database. Each row associated with one itemid (e.g. 212) corresponds to an instantiation of the same measurement (e.g. heart rate).

value
The documented date - this is the value that corresponds to the concept referred to by itemid. For example, if querying for itemid: 225755 (“18 Gauge Insertion Date”), then the value column indicates the date the line was inserted.

valueuom
The unit of measurement for the value - almost always the text string “Date”.

warning
warning specifies if a warning for this observation was manually documented by the care provider. """
    microbiologyevents_description = """microbiologyevents
Microbiology tests are a common procedure to check for infectious growth and to assess which antibiotic treatments are most effective.

The table is best explained with a demonstrative example. If a blood culture is requested for a patient, then a blood sample will be taken and sent to the microbiology lab. The time at which this blood sample is taken is the charttime. The spec_type_desc will indicate that this is a blood sample. Bacteria will be cultured on the blood sample, and the remaining columns depend on the outcome of this growth:

If no growth is found, the remaining columns will be NULL
If bacteria is found, then each organism of bacteria will be present in org_name, resulting in multiple rows for the single specimen (i.e. multiple rows for the given spec_type_desc).
If antibiotics are tested on a given bacterial organism, then each antibiotic tested will be present in the ab_name column (i.e. multiple rows for the given org_name associated with the given spec_type_desc). Antibiotic parameters and sensitivities are present in the remaining columns (dilution_text, dilution_comparison, dilution_value, interpretation). 
microevent_id
A unique integer denoting the row.

subject_id
subject_id is a unique identifier which specifies an individual patient. Any rows associated with a single subject_id pertain to the same individual.

hadm_id
hadm_id is an integer identifier which is unique for each patient hospitalization.

micro_specimen_id
Uniquely denoted the specimen from which the microbiology measurement was made. Most microbiology measurements are made on patient derived samples (specimens) such as blood, urine, and so on. Often multiple measurements are made on the same sample. The micro_specimen_id will group measurements made on the same sample, e.g. organisms which grew from the same blood sample.

order_provider_id
order_provider_id provides an anonymous identifier for the provider who ordered the microbiology test. Provider identifiers follow a consistent pattern: the letter “P”, followed by either three numbers, followed by two letters or two numbers. For example, “P003AB”, “P00102”, “P1248B”, etc. Provider identifiers are randomly generated and do not have any inherent meaning aside from uniquely identifying the same provider across the database.

chartdate, charttime
charttime records the time at which an observation was charted, and is usually the closest proxy to the time the data was actually measured. chartdate is the same as charttime, except there is no time available.

chartdate was included as time information is not always available for microbiology measurements: in order to be clear about when this occurs, charttime is null, and chartdate contains the date of the measurement.

In the cases where both charttime and chartdate exists, chartdate is equal to a truncated version of charttime (i.e. charttime without the timing information). Not all observations have a charttime, but all observations have a chartdate.

spec_itemid, spec_type_desc
The specimen which is tested for bacterial growth. The specimen is a sample derived from a patient; e.g. blood, urine, sputum, etc.

test_seq
If multiple samples are drawn, the test_seq will delineate them. For example, if an aerobic and anerobic culture bottle are used for the same specimen, they will have distinct test_seq values (likely 1 and 2).

storedate, storetime
The date (storedate) or date and time (storetime) of when the microbiology result was available. While many interim results are made available during the process of assessing a microbiology culture, the times here are the time of the last known update.

test_itemid, test_name
The test performed on the given specimen.

org_itemid, org_name
The organism, if any, which grew when tested. If NULL, no organism grew (i.e. a negative culture).

isolate_num
For testing antibiotics, the isolated colony (integer; starts at 1).

ab_itemid, ab_name
If an antibiotic was tested against the given organism for sensitivity, the antibiotic is listed here.

dilution_text, dilution_comparison, dilution_value
Dilution values when testing antibiotic sensitivity.

interpretation
interpretation of the antibiotic sensitivity, and indicates the results of the test. “S” is sensitive, “R” is resistant, “I” is intermediate, and “P” is pending.

comments
Deidentified free-text comments associated with the microbiology measurement. Usually these provide information about the sample, whether any notifications were made to care providers regarding the results, considerations for interpretation, or in some cases the comments contain the result of the measurement itself. Comments which have been fully deidentified (i.e. no information content retained) are present as three underscores: ___. A NULL comment indicates no comment was made for the row."""
    procedureevents_description = """ Detailed Description
subject_id, hadm_id, stay_id
Identifiers which specify the patient: subject_id is unique to a patient, hadm_id is unique to a patient hospital stay and stay_id is unique to a patient ICU stay.

caregiver_id
caregiver_id uniquely identifies a single caregiver who documented data in the ICU information system.

starttime, endtime
starttime and endtime record the start and end time of an event.

storetime
storetime specifies the time when the event was recorded in the system.

itemid
Identifier for a single measurement type in the database. Each row associated with one itemid (e.g. 212) corresponds to a type of measurement (e.g. heart rate). The d_items table may be joined on this field. For any itemid appearing in the procedureevents table, d_items linksto column will have the value ‘procedureevents’.

value
In the procedureevents table, this identifies the duration of the procedure (if applicable). For example, if querying for itemid 225794 (“Non-invasive Ventilation”), then the value column indicates the duration of ventilation therapy.

valueuom
The unit of measurement for the value. Most frequently “None” (no value recorded); otherwise one of “day”, “hour”, or “min”. A query for itemiid 225794 (“Non-invasive Ventilation”) returning a value of 461 and valueuom of ‘min’ would correspond to non-invasive ventilation provided for 461 minutes; this value is expected to match the difference between the starttime and endtime fields for the record. A procedure with valueuom equal to “None” corresponds to a procedure which is instantaneous (e.g. intubation, patient transfer) or whose duration is not relevant (e.g. imaging procedures). For these records, there will be a difference of one second between starttime and endtime values.

location , locationcategory
location and locationcategory provide information about where on the patient’s body the procedure is taking place. For example, the location might be ‘Left Upper Arm’ and the locationcategory might be ‘Invasive Venous’.

orderid, linkorderid
These columns link procedures to specific physician orders. Unlike in the mimic_icu.inputevents table, most procedures in procedureevents are ordered independently.

There are a limited number of records for which the same procedure was performed again at a later date under the same original order. When a procedure was repeated under the same original order, the linkorderid field of the record for the later procedure will be set to the orderid field of the earlier record. In all other cases, orderid = linkorderid.

ordercategoryname, ordercategorydescription
These columns provide higher level information about the medication/solution order. Categories represent the type of administration.

patientweight
The patient weight in kilograms.

isopenbag
Whether the order was from an open bag.

continueinnextdept
If the order ended on patient transfer, this field indicates if it continued into the next department (e.g. a floor).

statusdescription
statusdescription states the ultimate status of the procedure referred to in the row. The statuses appearing on the procedureevents table are:

Paused - The current delivery has been paused.
FinishedRunning - The delivery of the item has finished (most frequently, the bag containing the compound is empty).
Stopped - The delivery of the item been terminated by the caregiver.
Nearly all procedures recorded in procedureevents have a status of FinishedRunning.

originalamount, originalrate
These fields are present in the table and never null, but have no clear meaning. In particular, “originalrate” is either 0 or 1 for all records. """
    procedures_icd_description = """ During routine hospital care, patients are billed by the hospital for procedures they undergo. This table contains a record of all procedures a patient was billed for during their hospital stay using the ICD-9 and ICD-10 ontologies.
     subject_id
subject_id is a unique identifier which specifies an individual patient. Any rows associated with a single subject_id pertain to the same individual.

hadm_id
hadm_id is an integer identifier which is unique for each patient hospitalization.

seq_num
An assigned priority for procedures which occurred within the hospital stay.

chartdate
The date of the associated procedures. Date does not strictly correlate with seq_num.

icd_code, icd_version
icd_code is the International Coding Definitions (ICD) code.

There are two versions for this coding system: version 9 (ICD-9) and version 10 (ICD-10). These can be differentiated using the icd_version column. In general, ICD-10 codes are more detailed, though code mappings (or “cross-walks”) exist which convert ICD-9 codes to ICD-10 codes.

Both ICD-9 and ICD-10 codes are often presented with a decimal. This decimal is not required for interpretation of an ICD code; i.e. the icd_code of ‘0010’ is equivalent to ‘001.0’. """
    icustays_description =""" Detailed Description
subject_id, hadm_id, stay_id
Identifiers which specify the patient: subject_id is unique to a patient, hadm_id is unique to a patient hospital stay and stay_id is unique to a patient ward stay.

FIRST_CAREUNIT, LAST_CAREUNIT
FIRST_CAREUNIT and LAST_CAREUNIT contain, respectively, the first and last ICU type in which the patient was cared for. As an stay_id groups all ICU admissions within 24 hours of each other, it is possible for a patient to be transferred from one type of ICU to another and have the same stay_id.

Care units are derived from the TRANSFERS table, and definition for the abbreviations can be found in the documentation for TRANSFERS.

INTIME, OUTTIME
INTIME provides the date and time the patient was transferred into the ICU. OUTTIME provides the date and time the patient was transferred out of the ICU.

LOS
LOS is the length of stay for the patient for the given ICU stay, which may include one or more ICU units. The length of stay is measured in fractional days"""
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
    data = pd.read_csv(path)
    data_json = data.to_json(orient='records')
    return data_json
#------------------------------------------------------------

def load_dataset(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        texts = []
        for line in lines:
            json_line = json.loads(line)
            texts.append(json_line)
    return texts
    


def create_corpus(texts):
    corpus = []
    for text in texts:
        content = text["resource"].lower()+ " "+ text["description"].lower()
        corpus.append(content)
    return corpus


texts = load_dataset("Fase3/datasetRecursos.ndjson")
corpus = create_corpus(texts)
# Se crea la consulta
filename = "transfers"
data = load_data("Fase3/"+filename+"_mock.csv")
desc = get_description(filename).lower()
query = f"The tabular data I would like to know the columns mapping is : {data}, this table is described as {desc}. Tell me to which FHIR resource item each column belongs. Explain me why you selected that resource for each column."


def univ_sent_encoder(query,corpus):

    model = load_univ_sent_encoder_model()
    # Obtener el embedding de la frase concreta
    query_embedding = model([query])[0]

    # Calcular la similitud entre la frase y cada documento
    similarities_dict = {}
    contador = 0
    for c in corpus:
        doc_embedding = model([c])[0]
        similarity = 1 - cosine(query_embedding, doc_embedding)
        similarities_dict[contador] = similarity
        contador+=1

    similarities_dict

# Mostrar los resultados
for doc_id, similarity in sorted_similarities:
    print(f"Documento: {doc_id}, Similitud: {similarity:.4f}")
