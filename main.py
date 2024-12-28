import csv
import os
import pickle
import openai
from thefuzz import process
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_csv_data(file_path):
    """
    Load CSV data into a dictionary of show titles and descriptions.
    """
    data = {}
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data[row['Title']] = row['Description']
    return data

def generate_embeddings(show_descriptions, output_file):
    """
    Generate and save embeddings using OpenAI API.
    """
    if os.path.exists(output_file):
        print(f"Embeddings already exist in {output_file}.")
        return

    embeddings = {}
    for title, description in show_descriptions.items():
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Generate embeddings for the input text."},
                {"role": "user", "content": description}
            ]
        )
        embeddings[title] = eval(response['choices'][0]['message']['content'])

    with open(output_file, "wb") as f:
        pickle.dump(embeddings, f)

def recommend_shows(user_shows, embeddings_file):
    """
    Recommend shows based on input shows.
    """
    with open(embeddings_file, "rb") as f:
        embeddings = pickle.load(f)

    # Calculate average vector
    user_vectors = [embeddings[show] for show in user_shows if show in embeddings]
    if not user_vectors:
        return []
    average_vector = np.mean(user_vectors, axis=0).reshape(1, -1)

    # Calculate cosine similarity
    all_vectors = list(embeddings.values())
    all_titles = list(embeddings.keys())
    similarities = cosine_similarity(average_vector, all_vectors)[0]

    # Sort and return top recommendations
    ranked = sorted(
        zip(all_titles, similarities), key=lambda x: x[1], reverse=True
    )
    return [title for title, score in ranked if title not in user_shows][:5]
