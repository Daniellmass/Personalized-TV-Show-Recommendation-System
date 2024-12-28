import os
import pickle
import csv
import openai
import requests
from thefuzz import process
from annoy import AnnoyIndex
import json
import re
import time


# Color constants
GREEN = "\033[92m"
YELLOW_ORANGE = "\033[93m"
RESET = "\033[0m"

openai.api_key = os.getenv("OPENAI_API_KEY")
lightx_api_key = os.getenv("LIGHTX_API_KEY")

CSV_FILE = "./imdb_tvshows - imdb_tvshows.csv"
EMBEDDINGS_FILE = "embeddings.pkl"
ANNOY_INDEX_FILE = "annoy_index.ann"
NUM_DIMENSIONS = 3  
ANNOY_NUM_TREES = 10


def load_csv_data(file_path):
    """
    Load TV show data from a CSV file and return a dict: {title: description}.
    """
    data = {}
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data[row['Title']] = row['Description']
    return data

def generate_embeddings_if_needed(csv_file, embeddings_file):
    """
    Generate an embedding for each TV show only once and store it in a pickle.
    If the embeddings file already exists, skip generation.
    """
    if os.path.exists(embeddings_file):
        print(f"{GREEN}>> Embeddings file already exists. Skipping generation.{RESET}")
        return

    print(f"{GREEN}>> Generating embeddings...{RESET}")
    data = load_csv_data(csv_file)
    embeddings = {}

    for title, description in data.items():
        try:
            # Call the ChatCompletion API
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Generate a list of three numerical values for embedding "
                            f"based on the following description:\n"
                            f"\"{description}\"\n"
                            f"Return the list in Python list format, e.g. [1.23, 4.56, 7.89]."
                        ),
                    }
                ],
                max_tokens=100,
                temperature=0.2
            )

            raw_text = response.choices[0].message["content"].strip()
            

            # Use a regular expression to extract the list
            match = re.search(r"\[([\d.,\s\-]+)\]", raw_text)
            if not match:
                raise ValueError(f"No valid embedding found in response for {title}: {raw_text}")

            # Convert the matched string to a list of floats
            embedding = json.loads(f"[{match.group(1)}]")
            if isinstance(embedding, list) and len(embedding) == NUM_DIMENSIONS:
                embeddings[title] = embedding
            else:
                raise ValueError(f"Invalid embedding format for {title}: {embedding}")

        except Exception as e:
            print(f"{GREEN}Error generating embedding for {title}: {e}{RESET}")

    with open(embeddings_file, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"{GREEN}>> Embeddings saved to {embeddings_file}{RESET}")
    
def build_annoy_index_if_needed(embeddings_file, annoy_index_file):
    """
    Build an Annoy index from the embeddings if needed.
    """
    if os.path.exists(annoy_index_file):
        print(f"{GREEN}>> Annoy index file already exists. Skipping build.{RESET}")
        return

    print(f"{GREEN}>> Building Annoy index...{RESET}")
    with open(embeddings_file, "rb") as f:
        embeddings = pickle.load(f)

    index = AnnoyIndex(NUM_DIMENSIONS, 'angular')
    mapping = {}

    i = 0
    for title, vector in embeddings.items():
        if len(vector) == NUM_DIMENSIONS:
            index.add_item(i, vector)
            mapping[i] = title
            i += 1

    index.build(ANNOY_NUM_TREES)
    index.save(annoy_index_file)

    # Save the mapping (title to Annoy index ID)
    with open("mapping.pkl", "wb") as mf:
        pickle.dump(mapping, mf)

    print(f"{GREEN}>> Annoy index built and saved to {annoy_index_file} + mapping.pkl{RESET}")

def load_annoy_index(annoy_index_file):
    """
    Load the Annoy index, the mapping, and the embeddings from disk.
    """
    with open(EMBEDDINGS_FILE, "rb") as ef:
        embeddings = pickle.load(ef)
    with open("mapping.pkl", "rb") as mf:
        mapping = pickle.load(mf)

    index = AnnoyIndex(NUM_DIMENSIONS, 'angular')
    index.load(annoy_index_file)

    return index, mapping, embeddings


def fuzzy_match_user_shows(user_input_list, possible_titles):
    """
    Perform fuzzy matching on the user's input show list against the CSV titles.
    """
    matched_titles = []
    for show_name in user_input_list:
        match, score = process.extractOne(show_name, possible_titles)
        matched_titles.append(match)
    return matched_titles

def generate_image_with_lightx(prompt):
    """
    Generate an image using the LightX API, returning the URL of the image or an error message.
    """
    create_url = "https://api.lightxeditor.com/external/api/v1/text2image"
    
    
    payload = {
        "textPrompt": prompt
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": lightx_api_key  
    }

    try:
        
        response = requests.post(create_url, headers=headers, json=payload)
        if response.status_code != 200:
            return f"Failed to create image. Status code: {response.status_code}, Response: {response.text}"

        
        response_data = response.json()
        body_data = response_data.get("body", {})
        order_id = body_data.get("orderId")
        if not order_id:
            return "Failed to retrieve order ID from response."

        
        status_url = "https://api.lightxeditor.com/external/api/v1/order-status"
        for _ in range(5):  
            time.sleep(3)  

            status_payload = {"orderId": order_id}
            status_response = requests.post(status_url, headers=headers, json=status_payload)
            if status_response.status_code != 200:
                continue

            status_data = status_response.json()
            status_body = status_data.get("body", {})
            status = status_body.get("status")

            if status == "active":
                
                output_url = status_body.get("output")
                if output_url:
                    return output_url
            elif status == "failed":
                return "Image generation failed."

        return "Image generation timed out after 5 attempts."

    except Exception as e:
        return f"An error occurred: {e}"
def recommend_shows(user_shows, embeddings, index, mapping, top_n=5):
    """
    1. Compute the average vector for the shows the user likes.
    2. Use Annoy to find the most similar shows.
    3. Return a list of strings like ["Sherlock (85%)", "Dark (81%)", ...].
    """
    avg_vector = [0.0] * NUM_DIMENSIONS
    for show in user_shows:
        vec = embeddings[show]
        avg_vector = [sum(x) for x in zip(avg_vector, vec)]
    avg_vector = [x / len(user_shows) for x in avg_vector]

    nn_indices, distances = index.get_nns_by_vector(
        avg_vector,
        top_n + len(user_shows),
        include_distances=True
    )

    recommendations = []
    for i, dist in zip(nn_indices, distances):
        show_title = mapping[i]
        if show_title not in user_shows:
            score_percent = max(0, min(100, (1 - dist) * 100))
            recommendations.append(f"{show_title} ({int(score_percent)}%)")
            if len(recommendations) == top_n:
                break

    return recommendations


def generate_two_new_shows(user_shows, recommended_shows):
    """
    Generate two new fictional shows for the user.
    """
    show1_name = " & ".join([s.split()[0] for s in user_shows]) + " Universe"
    show1_desc = (
        f"A thrilling new show inspired by your favorite inputs: {', '.join(user_shows)}. "
        f"It's full of unexpected twists and turns!"
    )

    rec_titles = [r.split("(")[0].strip() for r in recommended_shows]
    show2_name = " & ".join([t.split()[0] for t in rec_titles]) + " Chronicles"
    show2_desc = (
        f"A brand-new adventure based on the shows I recommended for you: "
        f"{', '.join(rec_titles)}. Prepare for epic storytelling!"
    )

    return (show1_name, show1_desc), (show2_name, show2_desc)


def generate_show_ad(show_names):
    """
    Generate custom show ads using LightX API.
    Returns a list of image URLs or failure messages.
    """
    ads = []
    for show in show_names:
        prompt = f"Artistic poster for the TV show: {show}"
        ad_url = generate_image_with_lightx(prompt)
        ads.append(ad_url)
    return ads


def main():
    generate_embeddings_if_needed(CSV_FILE, EMBEDDINGS_FILE)
    build_annoy_index_if_needed(EMBEDDINGS_FILE, ANNOY_INDEX_FILE)
    index, mapping, embeddings = load_annoy_index(ANNOY_INDEX_FILE)
    data = load_csv_data(CSV_FILE)
    possible_titles = list(data.keys())

    while True:
        prompt_text = (
            f"{GREEN}Which TV shows did you really like watching?\n"
            f"Separate them by a comma. Make sure to enter more than 1 show.{RESET}\n"
        )
        user_input_raw = input(prompt_text)


        raw_shows = [x.strip() for x in user_input_raw.split(",") if x.strip()]
        if len(raw_shows) < 2:
            print(f"{GREEN}Please enter more than 1 show\n{RESET}")
            continue

        matched = fuzzy_match_user_shows(raw_shows, possible_titles)

        print(
            f"{GREEN}\nMaking sure, do you mean {', '.join(matched)}? (y/n){RESET}"
        )
        confirm = input().lower().strip()
        if confirm == 'y':
            print(f"{GREEN}Great! Generating recommendations now…\n{RESET}")
            user_shows = matched
            break
        else:
            print(f"{GREEN}Sorry about that. Let's try again, please make sure to write the names of the tv shows correctly\n{RESET}")
            continue

    recs = recommend_shows(user_shows, embeddings, index, mapping, top_n=5)
    print(f"{GREEN}Here are the TV shows that I think you would love:{RESET}")
    for r in recs:
        print(f"{GREEN}{r}{RESET}")

    (show1_name, show1_desc), (show2_name, show2_desc) = generate_two_new_shows(user_shows, recs)

    print(f"\n{GREEN}I have also created two shows which I think you would love.{RESET}")
    print(f"{GREEN}Show #1 is based on the fact that you loved the input shows that you gave me.{RESET}")
    print(f"{GREEN}Its name is {show1_name} and it is about {show1_desc}{RESET}")
    print(f"{GREEN}Show #2 is based on the shows I recommended for you.{RESET}")
    print(f"{GREEN}Its name is {show2_name} and it is about {show2_desc}{RESET}")
    print(f"{GREEN}Here are also the 2 TV show ads. Hope you like them!\n{RESET}")

    ads = generate_show_ad([show1_name, show2_name])
    for i, ad_link in enumerate(ads, start=1):
        print(f"{GREEN}Ad for Show #{i}: {ad_link}{RESET}")


if __name__ == "__main__":
    main()
