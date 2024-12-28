import os
import pickle
import pytest
from unittest.mock import patch
from annoy import AnnoyIndex
from main import load_csv_data, generate_embeddings, recommend_shows, generate_show_ad
from thefuzz import process


@pytest.fixture
def dummy_data():
    """
    Fixture to provide dummy data for testing.
    """
    return {
        "Game of Thrones": "Nine noble families fight for control over Westeros.",
        "Breaking Bad": "A high school chemistry teacher turns to manufacturing methamphetamine.",
        "Stranger Things": "When a young boy disappears, supernatural forces must be confronted."
    }


@patch("main.openai.ChatCompletion.create")
def test_generate_embeddings_with_mock(mock_openai, dummy_data):
    """
    Test generate_embeddings with OpenAI API mocked.
    """
    mock_openai.return_value = {
        "choices": [{"message": {"content": "[0.1, 0.2, 0.3]"}}]
    }
    output_file = "test_embeddings.pkl"
    generate_embeddings(dummy_data, output_file)

    assert os.path.exists(output_file), "Embeddings file was not created."

    with open(output_file, "rb") as f:
        embeddings = pickle.load(f)

    assert isinstance(embeddings, dict), "Embeddings should be a dictionary."
    assert len(embeddings) == len(dummy_data), "Embeddings count mismatch."

    os.remove(output_file)


@patch("main.requests.post")
def test_generate_show_ad(mock_post):
    """
    Test the generate_show_ad function with LightX API mocked.
    """
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"image_url": "http://mockimage.com/image1.jpg"}
    show_names = ["Breaking Bad", "Stranger Things"]

    ads = generate_show_ad(show_names)

    assert len(ads) == 2, "Ads should be generated for all input shows."
    for ad in ads:
        assert ad.startswith("http://"), "Generated ad URL is invalid."


def test_recommend_shows(dummy_data, tmp_path):
    """
    Test recommendation logic with dummy embeddings and Annoy index.
    """
    embeddings_file = str(tmp_path / "test_embeddings.pkl")  # Convert to string
    annoy_index_file = str(tmp_path / "test_annoy_index.ann")  # Convert to string

    # Save dummy embeddings
    dummy_embeddings = {
        "Game of Thrones": [0.1, 0.2, 0.3],
        "Breaking Bad": [0.4, 0.5, 0.6],
        "Stranger Things": [0.7, 0.8, 0.9],
    }
    with open(embeddings_file, "wb") as f:
        pickle.dump(dummy_embeddings, f)

    # Build and save Annoy index
    num_dimensions = len(next(iter(dummy_embeddings.values())))
    index = AnnoyIndex(num_dimensions, 'angular')
    for i, (title, vector) in enumerate(dummy_embeddings.items()):
        index.add_item(i, vector)
    index.build(10)
    index.save(annoy_index_file)  # Already converted to str

    user_input = ["Game of Thrones"]
    recommendations = recommend_shows(user_input, embeddings_file, annoy_index_file)

    assert isinstance(recommendations, list), "Recommendations should be a list."
    assert len(recommendations) > 0, "Recommendations list should not be empty."
