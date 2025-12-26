import pandas as pd
import ast
import os

def process_triples(data):
    """
    Converts a DataFrame with a 'relations' column (list of tuples or stringified list)
    into a DataFrame of triples (subject, relation, object), preserving all original columns.
    
    This function aligns with the Web App's upload feature by:
    1. Handling both stringified lists (from CSV) and actual lists (from memory).
    2. Preserving ALL original columns (dynamic handling).
    """
    triples_list = []

    # Ensure 'relations' column exists
    if "relations" not in data.columns:
        print("⚠ Warning: 'relations' column missing in dataset.")
        return pd.DataFrame()

    for _, row in data.iterrows():
        relations_raw = row["relations"]
        
        # 1. Handle String (from CSV) vs List (from App/Memory)
        if isinstance(relations_raw, str):
            try:
                # robust evaluation
                rels = ast.literal_eval(relations_raw)
            except (ValueError, SyntaxError):
                rels = []
        elif isinstance(relations_raw, list):
            rels = relations_raw
        else:
            rels = []

        # 2. Extract triples and preserve row context
        if isinstance(rels, list):
            for r in rels:
                # Ensure we have exactly 3 elements (subject, relation, object)
                if isinstance(r, (list, tuple)) and len(r) == 3:
                    subject, relation, obj = r
                    
                    # Create a dictionary of the original row (preserved columns)
                    # We drop 'relations' because we are exploding it
                    new_row = row.drop("relations").to_dict()
                    
                    # Add the new triple columns
                    new_row["subject"] = subject
                    new_row["relation"] = relation
                    new_row["object"] = obj
                    
                    triples_list.append(new_row)

    # Convert list to DataFrame
    if not triples_list:
         # Return empty DF with expected columns if no triples found
        expected_cols = list(data.columns.drop("relations")) + ["subject", "relation", "object"]
        return pd.DataFrame(columns=expected_cols)

    return pd.DataFrame(triples_list)

# Standalone execution (Legacy Mode)
if __name__ == "__main__":
    # Define paths relative to this script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_PATH = os.path.join(BASE_DIR, "../Data/Processed/relations_extracted_dataset.csv")
    OUTPUT_PATH = os.path.join(BASE_DIR, "../Data/Processed/triples.csv")

    try:
        print(f"📂 Loading data from: {INPUT_PATH}")
        if not os.path.exists(INPUT_PATH):
             print(f"❌ Error: File not found at {INPUT_PATH}")
             exit(1)

        data = pd.read_csv(INPUT_PATH)
        
        # Process the data using the aligned function
        triples_df = process_triples(data)
        
        # Save output
        triples_df.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ Success! Generated {len(triples_df)} triples.")
        print(f"💾 Saved to: {OUTPUT_PATH}")
        print("\nSnapshot:")
        print(triples_df.head())

    except Exception as e:
        print(f"❌ An error occurred: {e}")
