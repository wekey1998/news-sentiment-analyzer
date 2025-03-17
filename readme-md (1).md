# News Sentiment Analysis Application

This application extracts news articles about a given company, performs sentiment analysis, conducts a comparative analysis across articles, and generates a text-to-speech summary in Hindi.

## Features

- **News Extraction**: Extracts title, summary, and content from news articles related to a given company
- **Sentiment Analysis**: Performs sentiment analysis (positive, negative, neutral) on articles
- **Topic Extraction**: Identifies key topics discussed in each article
- **Comparative Analysis**: Compares sentiment and topics across articles
- **Text-to-Speech**: Converts summaries to Hindi speech
- **User Interface**: Simple Streamlit interface for user interaction
- **API Backend**: FastAPI backend for processing requests

## Architecture

The application uses a client-server architecture:

1. **Frontend**: Streamlit web interface for user input and result visualization
2. **Backend**: FastAPI server providing API endpoints for news analysis
3. **Processing Pipeline**:
   - News extraction using BeautifulSoup
   - Sentiment analysis using Hugging Face Transformers
   - Topic extraction using NLTK
   - Text-to-speech conversion using gTTS

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/news-sentiment-analysis.git
   cd news-sentiment-analysis
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Start the API server:
   ```
   uvicorn api:app --reload --host 0.0.0.0 --port 8000
   ```

4. In a separate terminal, start the Streamlit app:
   ```
   streamlit run app.py
   ```

5. Open your browser and navigate to http://localhost:8501 to access the application.

## API Documentation

The application provides the following API endpoints:

### Analyze Company News

**Endpoint**: `/analyze`

**Method**: POST

**Request Body**:
```json
{
    "company_name": "Tesla"
}
```

**Response**:
```json
{
    "company": "Tesla",
    "articles": [
        {
            "title": "Article Title",
            "summary": "Article Summary",
            "sentiment": {
                "label": "Positive",
                "score": 0.95
            },
            "topics": ["Electric Vehicles", "Stock Market"],
            "url": "https://example.com/article"
        }
    ],
    "comparative_sentiment_score": {
        "sentiment_distribution": {
            "Positive": 5,
            "Negative": 3,
            "Neutral": 2
        },
        "common_topics": ["Electric Vehicles"],
        "unique_topics": {
            "Unique Topics in Article 1": ["Innovation"]
        },
        "coverage_differences": [
            {
                "comparison": "Article comparison text",
                "impact": "Impact analysis"
            }
        ],
        "final_sentiment_analysis": "Overall sentiment summary"
    },
    "final_sentiment_analysis": "Tesla's latest news coverage is mostly positive.",
    "audio": "data/tesla_report.mp3"
}
```

### Get Audio File

**Endpoint**: `/audio/{filename}`

**Method**: GET

**Response**: Audio file (MP3)

## Models Used

### Sentiment Analysis

The application uses Hugging Face's pre-trained sentiment analysis model to classify article sentiments as positive, negative, or neutral.

### Topic Extraction

For topic extraction, the application uses NLTK for tokenization, stopword removal, and frequency analysis to identify key topics in each article.

### Text-to-Speech

The application uses Google's Text-to-Speech (gTTS) API to convert text summaries to Hindi speech.

## Assumptions & Limitations

- **News Sources**: The application currently scrapes news from Google News, which might have limitations on the number of requests.
- **Article Content Extraction**: The content extraction might not work perfectly for all news websites due to varying HTML structures.
- **Sentiment Analysis**: The sentiment model is trained on general text and might not be optimized for financial news.
- **Topic Extraction**: The current implementation uses frequency-based methods rather than more sophisticated topic modeling.
- **Hindi Translation**: The Hindi summary is currently generated using basic templates rather than actual translation models.
- **Browser Compatibility**: The application is tested on Chrome and Firefox; other browsers might have compatibility issues.

## Deployment

The application is deployed on Hugging Face Spaces at [https://huggingface.co/spaces/yourusername/news-sentiment-analysis](https://huggingface.co/spaces/yourusername/news-sentiment-analysis).

## Future Improvements

- Implement more sophisticated news extraction methods
- Use domain-specific sentiment analysis models
- Add more advanced topic modeling techniques
- Implement proper translation for multiple languages
- Add user authentication for personalized reports
- Implement caching to improve performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.
