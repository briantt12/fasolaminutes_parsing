import pandas as pd
import ollama

# Load the CSV file
df = pd.read_csv("spacy_names.csv")  # Update with your actual file name

# Check if 'Name Words' column exists
if 'Name Words' not in df.columns:
    raise ValueError("The CSV file must contain a column named 'Name Words'.")

# **Batch size:** Adjust based on performance
BATCH_SIZE = 10  

# Function to classify multiple names in one query
def classify_names_batch(names):
    names_list = " : ".join(names)  # Use '-' as a separator
    prompt = f"""For each of the following names, return 'True' if it's a real person's name and 'False' otherwise.
    Only return 'True' or 'False' for each name, in the same order, one per line:
    {names_list}
    """
    
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])

    # Extract response and split by the separator
    predictions = response['message']['content'].strip().split("\n")
    
    # Ensure the number of predictions matches the input size
    if len(predictions) != len(names):
        print(f"⚠️ Warning: Mismatch in batch size (Expected {len(names)}, Got {len(predictions)})")
        print(f"Raw Response: {response['message']['content']}")
        return ["ERROR"] * len(names)  # Fill errors if mismatch

    return predictions

# Split data into batches
total_entries = len(df)
df["Prediction"] = ""

for i in range(0, total_entries, BATCH_SIZE):
    batch = df["Name Words"].astype(str)[i:i + BATCH_SIZE].tolist()
    
    try:
        batch_predictions = classify_names_batch(batch)

        # Ensure batch size matches response size
        if len(batch_predictions) == len(batch):
            df.loc[i:i + BATCH_SIZE - 1, "Prediction"] = batch_predictions
        else:
            print(f"⚠️ Warning: Mismatch in batch {i}-{i+BATCH_SIZE-1} (Expected {len(batch)}, Got {len(batch_predictions)})")
            df.loc[i:i + BATCH_SIZE - 1, "Prediction"] = ["ERROR"] * len(batch)

    except Exception as e:
        print(f"❌ Error processing batch {i}-{i+BATCH_SIZE-1}: {e}")
        df.loc[i:i + BATCH_SIZE - 1, "Prediction"] = ["ERROR"] * len(batch)

    # Print progress update
    percent_complete = min((i + BATCH_SIZE) / total_entries * 100, 100)
    print(f"Processed {min(i + BATCH_SIZE, total_entries)}/{total_entries} ({percent_complete:.2f}% completed)", end="\r")

# Count occurrences of 'True' and 'False'
true_count = (df["Prediction"] == "True").sum()
false_count = (df["Prediction"] == "False").sum()
error_count = (df["Prediction"] == "ERROR").sum()
total = len(df)

# Print results
print("\nProcessing complete!")
print(f"Total Entries: {total}")
print(f"True Count: {true_count}")
print(f"False Count: {false_count}")
print(f"Errors: {error_count}")
print(f"Accuracy (assuming Ollama is perfect): {true_count / total:.2%}")

# Save results to a new CSV file
df.to_csv("classified_names.csv", index=False)
print("Results saved to classified_names.csv")
