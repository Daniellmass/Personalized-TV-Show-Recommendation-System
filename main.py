import openai
import pickle

def generate_embeddings(show_descriptions, output_file):
    """
    Generate embeddings for given show descriptions using OpenAI API and save them to a file.

    Args:
        show_descriptions (dict): A dictionary where keys are show titles and values are descriptions.
        output_file (str): Path to the file where embeddings will be saved.

    Returns:
        None
    """
    embeddings = {}
    for title, description in show_descriptions.items():
        response = openai.Embedding.create(
            input=description,
            model="text-embedding-ada-002"
        )
        embeddings[title] = response['data'][0]['embedding']

    # Save embeddings to a file
    with open(output_file, "wb") as f:
        pickle.dump(embeddings, f)
