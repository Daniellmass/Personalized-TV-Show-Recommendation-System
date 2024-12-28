import os
import pickle
import openai
from annoy import AnnoyIndex
from thefuzz import process
import requests

# Load API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
lightx_api_key = os.getenv("LIGHTX_API_KEY")


def load_csv_data(file_path):
    """
    Load TV show data from a CSV file.
    """
    import csv
    data = {}
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data[row['Title']] = row['Description']
    return data

def generate_embeddings(data, output_file):
    """
    Generate embeddings for the given TV show descriptions and save them to a file.
    """
    embeddings = {}
    for title, description in data.items():
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": f"Generate an embedding vector for: {description}"}
            ]
        )
        embedding = eval(response['choices'][0]['message']['content'])
        embeddings[title] = embedding
    with open(output_file, "wb") as f:
        pickle.dump(embeddings, f)

def build_annoy_index(embeddings, num_dimensions=512, output_file="annoy_index.ann"):
    """
    Build an Annoy index for fast similarity search.
    """
    index = AnnoyIndex(num_dimensions, 'angular')
    mapping = {}
    for i, (title, vector) in enumerate(embeddings.items()):
        index.add_item(i, vector)
        mapping[i] = title
    index.build(10)  # Number of trees
    index.save(str(output_file))  # Convert to string explicitly
    return index, mapping
def recommend_shows(user_input, embeddings_file, annoy_index_file, mapping):
    """
    Recommend TV shows based on user input using embeddings and Annoy index.
    """
    # Load embeddings
    with open(embeddings_file, "rb") as f:
        embeddings = pickle.load(f)

    # Load Annoy index
    num_dimensions = len(next(iter(embeddings.values())))
    index = AnnoyIndex(num_dimensions, 'angular')
    index.load(annoy_index_file)

    # Fuzzy match user input to titles
    matched_titles = []
    for input_title in user_input:
        match, _ = process.extractOne(input_title, embeddings.keys())
        matched_titles.append(match)

    # Calculate recommendations
    average_vector = [0] * num_dimensions
    for title in matched_titles:
        average_vector = [sum(x) for x in zip(average_vector, embeddings[title])]
    average_vector = [x / len(matched_titles) for x in average_vector]

    # Get nearest neighbors
    nearest_neighbors = index.get_nns_by_vector(average_vector, 5, include_distances=True)
    recommendations = []
    for i, distance in zip(nearest_neighbors[0], nearest_neighbors[1]):
        show_title = mapping[i]
        if show_title not in matched_titles:
            percentage = int((1 - distance) * 100)
            recommendations.append(f"{show_title} ({percentage}%)")
    return recommendations


def generate_show_ad(show_names):
    """
    Generate custom show ads using LightX API.
    """
    ads = []
    for show in show_names:
        url = "https://api.lightxeditor.com/v1/generate-image"
        headers = {"Authorization": f"Bearer {lightx_api_key}"}
        payload = {
            "prompt": f"Artistic poster for the TV show: {show}.",
            "n": 1,
            "size": "256x256"
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            image_url = response.json().get("image_url")
            ads.append(image_url)
        else:
            ads.append("Failed to generate ad.")
    return ads


if __name__ == "__main__":
    # Example usage
    csv_file = "shows.csv"
    embeddings_file = "embeddings.pkl"
    annoy_index_file = "annoy_index.ann"

    # Load data and generate embeddings if needed
    if not os.path.exists(embeddings_file):
        print("Generating embeddings...")
        data = load_csv_data(csv_file)
        generate_embeddings(data, embeddings_file)

    # Build Annoy index if needed
    if not os.path.exists(annoy_index_file):
        print("Building Annoy index...")
        with open(embeddings_file, "rb") as f:
            embeddings = pickle.load(f)
        build_annoy_index(embeddings)

    # User interaction
    user_input = input("Which TV shows did you really like watching? Separate them by commas: ").split(", ")
    recommendations = recommend_shows(user_input, embeddings_file, annoy_index_file)
    print("\nHere are the TV shows I recommend:")
    print("\n".join(recommendations))

    # Generate and display show ads
    ad_shows = ["Breaking Bad", "Stranger Things"]  # Example
    ads = generate_show_ad(ad_shows)
    for url in ads:
        print(f"Generated ad: {url}")
