import os
import pickle
import pytest
from unittest.mock import patch
from main import load_csv_data, generate_embeddings, recommend_shows, build_annoy_index
from thefuzz import process
from annoy import AnnoyIndex

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

def test_load_csv_data(tmp_path):
    """
    Test the load_csv_data function.
    """
    csv_content = """Title,Description
    Game of Thrones,Nine noble families fight for control over Westeros.
    Breaking Bad,A high school chemistry teacher turns to manufacturing methamphetamine."""
    csv_file = tmp_path / "shows.csv"
    csv_file.write_text(csv_content)

    data = load_csv_data(csv_file)
    assert len(data) == 2, "CSV loading failed."

def test_fuzzy_matching():
    """
    Test fuzzy matching functionality.
    """
    shows = ["Game of Thrones", "Breaking Bad", "Stranger Things"]
    input_show = "gem of throns"
    match, score = process.extractOne(input_show, shows)
    assert match == "Game of Thrones", "Fuzzy matching failed."
    assert score > 80, "Fuzzy match score too low."

def test_build_annoy_index(dummy_data):
    """
    Test building an Annoy index for embeddings.
    """
    embeddings = {
        "Game of Thrones": [0.1, 0.2, 0.3],
        "Breaking Bad": [0.4, 0.5, 0.6],
        "Stranger Things": [0.7, 0.8, 0.9],
    }
    index, mapping = build_annoy_index(embeddings, num_dimensions=3)
    assert isinstance(index, AnnoyIndex), "Annoy index was not created properly."
    assert len(mapping) == len(embeddings), "Mapping size mismatch."

@patch("main.build_annoy_index")
def test_recommend_shows_with_annoy(mock_annoy, dummy_data):
    """
    Test the recommend_shows function using Annoy index.
    """
    embeddings = {
        "Game of Thrones": [0.1, 0.2, 0.3],
        "Breaking Bad": [0.4, 0.5, 0.6],
        "Stranger Things": [0.7, 0.8, 0.9],
    }

    user_input = ["Game of Thrones"]
    mock_annoy.return_value = AnnoyIndex(3, 'angular'), {
        0: "Game of Thrones",
        1: "Breaking Bad",
        2: "Stranger Things",
    }

    recommendations = recommend_shows(user_input, embeddings, "dummy_annoy_index.ann")
    assert len(recommendations) == 2, "Incorrect number of recommendations."
    assert "Breaking Bad" in recommendations, "Expected show not in recommendations."
    assert "Stranger Things" in recommendations, "Expected show not in recommendations."
