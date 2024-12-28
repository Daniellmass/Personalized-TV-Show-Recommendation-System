import os
import pickle
from main import generate_embeddings

def test_generate_embeddings():
    """
    Test the generate_embeddings function to ensure embeddings are generated and saved correctly.
    """
    # Sample data for testing
    sample_data = {
        "Game of Thrones": "Nine noble families fight for control over the lands of Westeros.",
        "Breaking Bad": "A high school chemistry teacher turns to manufacturing methamphetamine.",
    }

    # Call the function
    generate_embeddings(sample_data, "test_embeddings.pkl")

    # Check if the file is created
    assert os.path.exists("test_embeddings.pkl"), "Embeddings file was not created."

    # Check if the file contains a dictionary
    with open("test_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    assert isinstance(embeddings, dict), "Embeddings should be a dictionary."
    assert len(embeddings) == len(sample_data), "The number of embeddings should match the input data."

    # Check the format of the embeddings
    for key, vector in embeddings.items():
        assert isinstance(vector, list), "Each embedding should be a list."
        assert len(vector) == 3, "Each embedding should have a length of 3 (example placeholder)."

    # Clean up the test file
    os.remove("test_embeddings.pkl")
