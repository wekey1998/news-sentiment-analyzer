import requests
from bs4 import BeautifulSoup
import re
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import os
import numpy as np
import json
from gtts import gTTS
from typing import List, Dict, Any

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class NewsExtractor:
    """Class for extracting news articles related to a company."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def search_news(self, company_name: str, num_articles: int = 10) -> List[str]:
        """
        Search for news articles related to a company.
        
        Args:
            company_name: Name of the company to search for
            num_articles: Number of articles to retrieve (default: 10)
            
        Returns:
            List of URLs for news articles
        """
        # Using a search engine API would be ideal, but for this example,
        # we'll use a basic approach with a news search URL
        search_url = f"https://news.google.com/search?q={company_name}&hl=en-US&gl=US&ceid=US%3Aen"
        
        try:
            response = requests.get(search_url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract article links
            article_links = []
            articles = soup.find_all('a', class_='VDXfz')
            
            for article in articles[:num_articles]:
                # Get the href attribute
                href = article.get('href')
                if href and href.startswith('./articles/'):
                    # Convert relative URL to absolute URL
                    full_url = "https://news.google.com/" + href[2:]
                    article_links.append(full_url)
            
            return article_links[:num_articles]
        
        except Exception as e:
            print(f"Error searching for news: {str(e)}")
            return []
    
    def extract_article_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a news article.
        
        Args:
            url: URL of the news article
            
        Returns:
            Dictionary containing article title, summary, and content
        """
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.find('title').text if soup.find('title') else "No title found"
            
            # Extract main content
            # This is a simplified approach; actual implementation would need to be adjusted
            # based on the structure of the news websites
            paragraphs = soup.find_all('p')
            content = ' '.join([p.text for p in paragraphs if len(p.text) > 100])
            
            # Create a summary (first 2 sentences or 200 characters)
            summary = ' '.join(content.split('. ')[:2]) if content else "No summary available"
            if len(summary) > 200:
                summary = summary[:197] + "..."
            
            return {
                "title": title,
                "summary": summary,
                "content": content
            }
        
        except Exception as e:
            print(f"Error extracting article content: {str(e)}")
            return {
                "title": "Error retrieving article",
                "summary": "Could not extract content from this URL",
                "content": ""
            }

class SentimentAnalyzer:
    """Class for performing sentiment analysis on news articles."""
    
    def __init__(self):
        # Load sentiment analysis model
        self.sentiment_model = pipeline("sentiment-analysis")
        
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing sentiment label and score
        """
        # Truncate text if it's too long (model limit is typically around 512 tokens)
        truncated_text = text[:1000] if len(text) > 1000 else text
        
        if not truncated_text:
            return {"label": "Neutral", "score": 0.5}
        
        try:
            result = self.sentiment_model(truncated_text)[0]
            
            # Map the model's output to Positive, Negative, or Neutral
            label = result['label']
            score = result['score']
            
            if label == 'POSITIVE':
                sentiment = 'Positive'
            elif label == 'NEGATIVE':
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
            
            return {"label": sentiment, "score": score}
        
        except Exception as e:
            print(f"Error analyzing sentiment: {str(e)}")
            return {"label": "Neutral", "score": 0.5}
    
    def extract_topics(self, text: str, num_topics: int = 3) -> List[str]:
        """
        Extract key topics from a text.
        
        Args:
            text: Text to analyze
            num_topics: Number of topics to extract
            
        Returns:
            List of key topics
        """
        # Tokenize the text
        words = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
        
        # Count word frequencies
        word_freq = Counter(filtered_words)
        
        # Get the most common words as topics
        topics = [word for word, _ in word_freq.most_common(num_topics)]
        
        # Capitalize topics for better presentation
        topics = [topic.capitalize() for topic in topics]
        
        return topics

class ComparativeAnalyzer:
    """Class for performing comparative analysis on news articles."""
    
    def compare_articles(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comparative analysis on a list of articles.
        
        Args:
            articles: List of article dictionaries with sentiment and topics
            
        Returns:
            Dictionary containing comparative analysis results
        """
        # Count sentiment distribution
        sentiment_counts = {
            "Positive": 0,
            "Negative": 0,
            "Neutral": 0
        }
        
        for article in articles:
            sentiment = article.get("sentiment", {}).get("label", "Neutral")
            sentiment_counts[sentiment] += 1
        
        # Find topic overlap
        all_topics = [set(article.get("topics", [])) for article in articles]
        common_topics = set.intersection(*all_topics) if all_topics else set()
        
        # Generate comparisons
        comparisons = []
        if len(articles) > 1:
            for i in range(len(articles) - 1):
                for j in range(i + 1, len(articles)):
                    article1 = articles[i]
                    article2 = articles[j]
                    
                    # Skip if articles don't have required fields
                    if not all(key in article1 and key in article2 for key in ["title", "sentiment", "topics"]):
                        continue
                    
                    sentiment1 = article1["sentiment"]["label"]
                    sentiment2 = article2["sentiment"]["label"]
                    
                    if sentiment1 != sentiment2:
                        comparison = {
                            "comparison": f"Article '{article1['title']}' is {sentiment1.lower()}, while article '{article2['title']}' is {sentiment2.lower()}.",
                            "impact": f"This contrast highlights the mixed reception of the company's activities in different contexts."
                        }
                        comparisons.append(comparison)
        
        # Determine the final sentiment
        max_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
        final_sentiment = f"The overall sentiment across the articles is predominantly {max_sentiment.lower()}."
        
        if sentiment_counts["Positive"] == sentiment_counts["Negative"]:
            final_sentiment = "The overall sentiment across the articles is mixed."
        
        # Create unique topics for each article
        unique_topics = {}
        for i, article in enumerate(articles):
            article_topics = set(article.get("topics", []))
            other_topics = set()
            for j, other_article in enumerate(articles):
                if i != j:
                    other_topics.update(set(other_article.get("topics", [])))
            
            unique_topics[f"Unique Topics in Article {i+1}"] = list(article_topics - other_topics)
        
        return {
            "sentiment_distribution": sentiment_counts,
            "common_topics": list(common_topics),
            "unique_topics": unique_topics,
            "coverage_differences": comparisons[:5],  # Limit to top 5 comparisons
            "final_sentiment_analysis": final_sentiment
        }

class TextToSpeechConverter:
    """Class for converting text to speech in Hindi."""
    
    def text_to_speech(self, text: str, company_name: str) -> str:
        """
        Convert text to Hindi speech.
        
        Args:
            text: Text to convert to speech
            company_name: Name of the company for filename
            
        Returns:
            Path to the saved audio file
        """
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Generate a clean filename
        clean_name = re.sub(r'[^\w\s]', '', company_name).lower().replace(' ', '_')
        filename = f"data/{clean_name}_report.mp3"
        
        try:
            # Convert text to Hindi speech
            tts = gTTS(text=text, lang='hi', slow=False)
            tts.save(filename)
            return filename
        
        except Exception as e:
            print(f"Error converting text to speech: {str(e)}")
            return ""
    
    def generate_hindi_summary(self, company_name: str, sentiment_analysis: Dict[str, Any]) -> str:
        """
        Generate a Hindi summary of the sentiment analysis.
        
        Args:
            company_name: Name of the company
            sentiment_analysis: Sentiment analysis results
            
        Returns:
            Hindi summary text
        """
        # This is a basic template - in a real application, you might want to use
        # a proper translation service or more sophisticated text generation
        
        # Simple mapping of sentiment to Hindi
        sentiment_hindi = {
            "Positive": "सकारात्मक",
            "Negative": "नकारात्मक",
            "Neutral": "तटस्थ"
        }
        
        # Get the predominant sentiment
        sentiment_counts = sentiment_analysis.get("sentiment_distribution", {})
        max_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
        hindi_sentiment = sentiment_hindi.get(max_sentiment, "तटस्थ")
        
        # Generate Hindi summary
        hindi_summary = f"{company_name} के बारे में समाचार विश्लेषण। अधिकतर समाचार {hindi_sentiment} हैं। "
        
        positive_count = sentiment_counts.get("Positive", 0)
        negative_count = sentiment_counts.get("Negative", 0)
        neutral_count = sentiment_counts.get("Neutral", 0)
        
        hindi_summary += f"कुल {positive_count} सकारात्मक, {negative_count} नकारात्मक, और {neutral_count} तटस्थ समाचार मिले।"
        
        return hindi_summary

def process_company_news(company_name: str, num_articles: int = 10) -> Dict[str, Any]:
    """
    Process news articles for a given company.
    
    Args:
        company_name: Name of the company
        num_articles: Number of articles to retrieve
        
    Returns:
        Dictionary containing processed news data
    """
    # Initialize components
    news_extractor = NewsExtractor()
    sentiment_analyzer = SentimentAnalyzer()
    comparative_analyzer = ComparativeAnalyzer()
    tts_converter = TextToSpeechConverter()
    
    # Search for news articles
    article_urls = news_extractor.search_news(company_name, num_articles)
    
    # Process each article
    articles = []
    for url in article_urls:
        # Extract article content
        article_data = news_extractor.extract_article_content(url)
        
        # Skip if no content was extracted
        if not article_data["content"]:
            continue
        
        # Analyze sentiment
        sentiment = sentiment_analyzer.analyze_sentiment(article_data["content"])
        
        # Extract topics
        topics = sentiment_analyzer.extract_topics(article_data["content"])
        
        # Create article object
        article = {
            "title": article_data["title"],
            "summary": article_data["summary"],
            "sentiment": sentiment,
            "topics": topics,
            "url": url
        }
        
        articles.append(article)
    
    # Perform comparative analysis
    comparative_analysis = comparative_analyzer.compare_articles(articles)
    
    # Generate Hindi summary
    hindi_summary = tts_converter.generate_hindi_summary(company_name, comparative_analysis)
    
    # Convert summary to speech
    audio_file = tts_converter.text_to_speech(hindi_summary, company_name)
    
    # Prepare response
    response = {
        "company": company_name,
        "articles": articles,
        "comparative_sentiment_score": comparative_analysis,
        "final_sentiment_analysis": comparative_analysis["final_sentiment_analysis"],
        "audio": audio_file
    }
    
    return response
