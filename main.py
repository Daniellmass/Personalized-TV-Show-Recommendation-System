import openai
import os
import pickle
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment variable
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("API key is not set. Please set OPENAI_API_KEY in your environment variables.")

# Set the API key for OpenAI
openai.api_key = API_KEY

def generate_embeddings(show_descriptions, output_file):
    """
    Generate embeddings for given show descriptions using OpenAI's gpt-4o-mini and save them to a file.

    Args:
        show_descriptions (dict): A dictionary where keys are show titles and values are descriptions.
        output_file (str): Path to the file where embeddings will be saved.

    Returns:
        None
    """
    embeddings = {}
    for title, description in show_descriptions.items():
        try:
            # Generate embeddings using gpt-4o-mini
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Generate embeddings for the input text."},
                    {"role": "user", "content": description}
                ]
            )
            embeddings[title] = response['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error generating embedding for {title}: {e}")
            raise

    # Save embeddings to a file
    with open(output_file, "wb") as f:
        pickle.dump(embeddings, f)
