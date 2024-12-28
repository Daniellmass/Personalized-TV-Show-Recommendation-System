import os
import pickle
import pytest
from unittest.mock import patch
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


@patch("main.openai.Embedding.create")
def test_generate_embeddings_with_mock(mock_openai, dummy_data):
    """
    Test generate_embeddings with OpenAI API mocked.
    """
    mock_openai.return_value = {
        "data": [{"embedding": [0.1, 0.2, 0.3]}]
    }
    output_file = "test_embeddings.pkl"
    generate_embeddings(dummy_data, output_file)

    assert os.path.exists(output_file), "Embeddings file was not created."

    with open(output_file, "rb") as f:
        embeddings = pickle.load(f)

    assert isinstance(embeddings, dict), "Embeddings should be a dictionary."
    assert len(embeddings) == len(dummy_data), "Embeddings count mismatch."

    os.remove(output_file)


@patch("main.LightXAPI.generate_image")
def test_generate_show_ad(mock_lightx):
    """
    Test the generate_show_ad function with LightX API mocked.
    """
    mock_lightx.return_value = {"image_url": "http://mockimage.com/image1.jpg"}
    show_names = ["Breaking Bad", "Stranger Things"]

    ads = generate_show_ad(show_names)

    assert len(ads) == 2, "Ads should be generated for all input shows."
    for ad in ads:
        assert ad.startswith("http://"), "Generated ad URL is invalid."


@patch("main.cosine_similarity")
def test_recommend_shows(mock_cosine):
    """
    Test recommendation logic.
    """
    embeddings = {
        "Game of Thrones": [0.1, 0.2, 0.3],
        "Breaking Bad": [0.4, 0.5, 0.6],
        "Stranger Things": [0.7, 0.8, 0.9],
    }
    user_input = ["Game of Thrones"]
    mock_cosine.return_value = [0.9, 0.8]

    recommendations = recommend_shows(user_input, embeddings)
    assert len(recommendations) == 2, "Incorrect number of recommendations."
