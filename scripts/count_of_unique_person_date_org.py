import pandas as pd
import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load dataset
data = pd.read_csv("../Data/Raw/cross_domain_nlp_dataset_5000.csv")

# Function to extract specific entities
def extract_selected_entities(text):
    doc = nlp(str(text))
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "DATE"]:
            entities.append((ent.text.strip(), ent.label_))
    return entities

# Apply function to each sentence
data["entities"] = data["sentence"].apply(extract_selected_entities)

# Flatten all entities into one list
all_entities = [ent for sublist in data["entities"] for ent in sublist]

# Convert to DataFrame for easy counting
entities_df = pd.DataFrame(all_entities, columns=["Entity", "Label"])

# Count unique entities by type
unique_counts = entities_df.groupby("Label")["Entity"].nunique()

# Print results
print("✅ Unique entity counts:")
print(unique_counts)

# Optional: Save all extracted entities to CSV
entities_df.to_csv("../Data/Processed/selected_entities_extracted.csv", index=False)
print("\nAll PERSON, ORG, and DATE entities saved to selected_entities_extracted.csv")
