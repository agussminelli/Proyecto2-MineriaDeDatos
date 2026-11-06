import pandas as pd

def relabel_categories(df, target_col="gs_text34"):
    """
    Groups specific death cause labels into broader category chapters.
    """
    
    category_groups = {
        "[Chapter I] Certain infectious and Parasitic Diseases": [
            "Diarrhea/Dysentery", "Other infectious diseases", "Other Infectious Diseases", "AIDS", "Sepsis", 
            "Meningitis", "Meningitis/Sepsis", "Malaria", "Encephalitis", 
            "Measles", "Hemorrhagic Fever", "Hemorrhagic fever","TB"
        ],
        "[Chapter II] Neoplasms": [
            "Leukemia/Lymphomas", "Colorectal Cancer", "Lung Cancer", 
            "Cervical Cancer", "Breast Cancer", "Stomach Cancer", 
            "Prostate Cancer", "Esophageal Cancer", "Other Cancers"
        ],
        "[Chapter IV] Endocrine, Nutritional and Metabolic Diseases": [
            "Diabetes"
        ],
        "[Chapter VI] Diseases of the Nervous System": [
            "Epilepsy"
        ],
        "[Chapter IX] Diseases of the circulatory system": [
            "Stroke", "Acute Myocardial Infarction", "Other Cardiovascular Diseases"
        ],
        "[Chapter X] Diseases of Respiratory System": [
            "Pneumonia", "Asthma", "COPD"
        ],
        "[Chapter XI] Diseases of the Digestive System": [
            "Cirrhosis", "Other Digestive Diseases"
        ],
        "[Chapter XIV] Diseases of the Genitourinary System": [
            "Renal Failure"
        ],
        "[Chapter XV] Pregnancy, childbirth and the puerperium": [
            "Preterm Delivery", "Stillbirth", "Maternal", "Birth Asphyxia", "Birth asphyxia"
        ],
        "[Chapter XVII] Congenital Malformations": [
            "Congenital Malformations", "Congenital malformations", "Congenital malformation",
        ],
        "[Chapter XIX] Injury, Poisoning and External Causes": [
            "Bite of Venomous Animal", "Poisonings"
        ],
        "[Chapter XX] External Causes of Morbidity and Mortality": [
            "Road Traffic", "Falls", "Homicide", "Fires", "Drowning", 
            "Suicide", "Violent Death", "Other injuries", "Other Injuries"
        ]
    }

    # Reverse the dictionary to map specific labels to the broad category
    mapping_dict = {}
    for broad_category, specific_labels in category_groups.items():
        for label in specific_labels:
            mapping_dict[label] = broad_category

    # Strip any hidden whitespaces in the dataset to ensure exact matches
    df[target_col] = df[target_col].str.strip()
    
    # Any label not in the mapping_dict will remain unchanged
    df[target_col] = df[target_col].replace(mapping_dict)
    
    return df