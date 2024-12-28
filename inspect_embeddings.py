import pickle

def inspect_embeddings(file_path):
    try:
        with open(file_path, "rb") as f:
            embeddings = pickle.load(f)
        print("Embeddings loaded successfully!")
        print(f"Number of shows: {len(embeddings)}")
        print("Sample keys (TV show titles):")
        for i, key in enumerate(embeddings.keys()):
            print(f"{i+1}. {key}")
            if i >= 9:  # Show only first 10 keys
                break
    except Exception as e:
        print(f"Error reading the file: {e}")

if __name__ == "__main__":
    inspect_embeddings("embeddings.pkl")
