import pandas as pd
import spacy

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Load dataset
data = pd.read_csv("../Data/Raw/cross_domain_nlp_dataset_5000.csv")  # change filename if needed

# Function to extract relations (subject, relation, object) from text
def extract_relations(text):
    doc = nlp(str(text))  # ensure text is string
    triples = []

    for sent in doc.sents:
        subject = None
        relation = None
        object_ = None

        for token in sent:
            # Detect subject
            if "subj" in token.dep_:
                subject = token.text
            # Detect relation (verb)
            if token.pos_ == "VERB":
                relation = token.lemma_  # base form
            # Detect object
            if "obj" in token.dep_:
                object_ = token.text

        if subject and relation and object_:
            triples.append((subject, relation, object_))

    return triples

# Apply RE to each sentence
data["relations"] = data["sentence"].apply(extract_relations)

# Display first few rows to check
print(data.head())

# Save to new CSV
data.to_csv("../Data/Processed/relations_extracted_dataset.csv", index=False)
print("✅ Relations saved to relations_extracted_dataset.csv")
