import os
import pickle
import pytest
from unittest.mock import patch
from main import generate_embeddings

@pytest.fixture
def dummy_data():
    """Fixture to provide dummy data for testing."""
    return {
        "Game of Thrones": "Nine noble families fight for control over the lands of Westeros.",
        "Breaking Bad": "A high school chemistry teacher turns to manufacturing methamphetamine.",
    }

@patch("main.openai.Embedding.create")
def test_generate_embeddings_with_mock(mock_openai, dummy_data):
    """
    Test the generate_embeddings function with OpenAI API mocked.
    """
    # Mock the API response
    mock_openai.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    
    output_file = "test_embeddings.pkl"
    generate_embeddings(dummy_data, output_file)

    # Check if the file exists
    assert os.path.exists(output_file), "Embeddings file was not created."

    # Verify content in the file
    with open(output_file, "rb") as f:
        embeddings = pickle.load(f)

    assert isinstance(embeddings, dict), "Embeddings should be a dictionary."
    assert len(embeddings) == len(dummy_data), "The number of embeddings does not match input data."

    for key, vector in embeddings.items():
        assert isinstance(vector, list), "Each embedding should be a list."
        assert len(vector) == 3, "Each embedding should have 3 dimensions."

    # Ensure OpenAI API was called the correct number of times
    assert mock_openai.call_count == len(dummy_data), "API should be called once per description."

    # Clean up
    os.remove(output_file)

def test_generate_embeddings_empty_data():
    """
    Test that the function handles empty input gracefully.
    """
    output_file = "test_embeddings.pkl"
    generate_embeddings({}, output_file)

    # Verify the file content
    with open(output_file, "rb") as f:
        embeddings = pickle.load(f)

    assert isinstance(embeddings, dict), "Embeddings should be a dictionary."
    assert len(embeddings) == 0, "Embeddings should be empty for empty input."

    # Clean up
    os.remove(output_file)
