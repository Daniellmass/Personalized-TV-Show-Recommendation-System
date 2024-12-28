# Personalized TV Show Recommendation System

## Overview
This project builds a personalized recommendation system for TV shows using:
- Embedding generation with OpenAI's API.
- Efficient similarity search with Annoy.
- Fuzzy matching for user inputs.
- Creative poster generation using the LightX API.

The system suggests similar TV shows based on user preferences, creates fictional shows, and generates artistic posters for them.

## Features
1. **CSV Data Parsing**: Load TV shows and their descriptions from a CSV file.
2. **Embedding Generation**: Use OpenAI to generate embeddings for each show.
3. **Annoy Index**: Build and use Annoy for fast similarity-based recommendations.
4. **Fuzzy Matching**: Match user inputs to titles with typos or variations.
5. **Show Recommendations**: Suggest top shows based on user preferences.
6. **Fictional Shows**: Create new fictional TV shows inspired by user preferences and recommendations.
7. **Poster Generation**: Generate custom posters for fictional shows using LightX API.

## Requirements
- Python 3.7+
- Libraries:
  - `openai`
  - `thefuzz`
  - `annoy`
  - `requests`
  - `pickle`
  - `csv`
  - `re`
  - `json`
  - `time`
- **API Keys**:
  - OpenAI: Set as `OPENAI_API_KEY`.
  - LightX: Set as `LIGHTX_API_KEY`.

## Setup

1. **Install Required Libraries**:
   ```bash
   pip install openai thefuzz annoy requests
   ```

2. **Set Environment Variables**:
   ```bash
   export OPENAI_API_KEY="your_openai_key"
   export LIGHTX_API_KEY="your_lightx_key"
   ```

3. **Prepare CSV File**:
   - Ensure the CSV file (default: `imdb_tvshows.csv`) is formatted with columns:
     - `Title`
     - `Description`

4. **Run the Program**:
   ```bash
   python tv_recommendation.py
   ```

## Usage

1. **Start the Program**:
   - The program will load or generate embeddings and build the Annoy index.

2. **Enter Your Favorite Shows**:
   - Input a list of TV shows you like (comma-separated).
   - The program performs fuzzy matching to ensure accurate matching.

3. **Receive Recommendations**:
   - View personalized TV show recommendations based on your input.

4. **Explore Fictional Shows**:
   - The program generates two new fictional shows based on your preferences and recommendations.

5. **Custom Posters**:
   - Posters for the fictional shows are generated and displayed as image URLs.

## Example Interaction

1. **Input**:
   ```
   User: Breaking Bad, Game of Thrones
   ```
2. **Output**:
   - Recommendations: "Sherlock (85%)", "Dark (81%)", etc.
   - Fictional Shows:
     - Name: "Breaking Game Universe"
     - Description: "A thrilling new show inspired by Breaking Bad and Game of Thrones."
   - Poster URLs:
     - Ad for Show #1: `<URL>`
     - Ad for Show #2: `<URL>`

## Key Files
- **`tv_recommendation.py`**: Main script.
- **`imdb_tvshows.csv`**: Input data for TV shows.
- **`embeddings.pkl`**: Generated embeddings.
- **`annoy_index.ann`**: Annoy index file.
- **`mapping.pkl`**: Mapping of titles to Annoy index IDs.
