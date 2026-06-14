import re
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Optional


def clean_text(text: str) -> str:
    """
    Remove HTML tags and clean whitespace from text.
    
    Args:
        text: Raw text with potential HTML
        
    Returns:
        Clean plain text
    """
    text = re.sub(r'<br\s*/?>', ' ', str(text))
    text = re.sub(r'<[^>]+>', '', text)
    return ' '.join(text.split())


def chunk_review(text: str, max_sentences: int = 3, min_length: int = 30) -> List[str]:
    """
    Split a review into chunks of sentences.
    
    Args:
        text: Review text
        max_sentences: Maximum sentences per chunk
        min_length: Minimum character length for a chunk
        
    Returns:
        List of text chunks
    """
    from nltk.tokenize import sent_tokenize
    
    if not text or len(str(text).strip()) < min_length:
        return []
    
    sentences = sent_tokenize(str(text))
    chunks = []
    current = []
    
    for sent in sentences:
        current.append(sent)
        if len(current) >= max_sentences:
            chunk = ' '.join(current).strip()
            if len(chunk) >= min_length:
                chunks.append(chunk)
            current = []
    
    if current:
        chunk = ' '.join(current).strip()
        if len(chunk) >= min_length:
            chunks.append(chunk)
    
    return chunks


def detect_aspect(text: str, aspect_patterns: Dict[str, re.Pattern]) -> str:
    """
    Detect which audio aspect a chunk discusses.
    
    Args:
        text: Chunk text
        aspect_patterns: Dictionary of aspect name → compiled regex
        
    Returns:
        Aspect name or 'general'
    """
    scores = {}
    for aspect, regex in aspect_patterns.items():
        matches = regex.findall(text)
        if matches:
            scores[aspect] = len(matches)
    
    if scores:
        return max(scores, key=scores.get)
    return 'general'


def get_sentiment(text: str, rating: float) -> str:
    """
    Determine sentiment from text and rating.
    
    Args:
        text: Review text
        rating: Star rating (1-5)
        
    Returns:
        'positive', 'negative', or 'neutral'
    """
    negative_words = [
        'harsh', 'sibilant', 'uncomfortable', 'broke', 'cheap',
        'muddy', 'veiled', 'distorted', 'flimsy', 'poor',
        'terrible', 'awful', 'worst', 'return', 'refund'
    ]
    positive_words = [
        'amazing', 'excellent', 'great', 'perfect', 'love',
        'best', 'outstanding', 'impressive', 'fantastic'
    ]
    
    text_lower = text.lower()
    neg_count = sum(1 for w in negative_words if w in text_lower)
    pos_count = sum(1 for w in positive_words if w in text_lower)
    
    if rating >= 4 and pos_count > neg_count:
        return 'positive'
    elif rating <= 2 and neg_count > pos_count:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    elif pos_count > neg_count:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    return 'neutral'


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Min-max normalize a list of scores to 0-1 range.
    
    Args:
        scores: List of raw scores
        
    Returns:
        Normalized scores
    """
    scores = np.array(scores)
    min_val, max_val = scores.min(), scores.max()
    if max_val == min_val:
        return list(np.ones_like(scores))
    return list((scores - min_val) / (max_val - min_val))


def compute_popularity_score(rating_number: int, max_ratings: int) -> float:
    """
    Compute popularity score using log scale.
    
    Args:
        rating_number: Number of ratings for the product
        max_ratings: Maximum ratings across all products
        
    Returns:
        Popularity score between 0 and 1
    """
    if max_ratings <= 1:
        return 0.0
    return np.log1p(rating_number) / np.log1p(max_ratings)


def format_price(price) -> str:
    """
    Format price for display.
    
    Args:
        price: Raw price value
        
    Returns:
        Formatted price string
    """
    if price is None or price == 'None' or price == '':
        return 'N/A'
    try:
        return f"${float(price):.2f}"
    except (ValueError, TypeError):
        return str(price)


def summarize_product(chunks: List[str], max_length: int = 200) -> str:
    """
    Create a brief summary from product review chunks.
    
    Args:
        chunks: List of review chunks for a product
        max_length: Maximum summary length
        
    Returns:
        Summary string
    """
    if not chunks:
        return "No reviews available."
    
    # Take the most informative chunk
    best_chunk = max(chunks, key=len)
    
    if len(best_chunk) <= max_length:
        return best_chunk
    
    # Truncate at sentence boundary
    truncated = best_chunk[:max_length]
    last_period = truncated.rfind('.')
    if last_period > max_length // 2:
        return truncated[:last_period + 1]
    return truncated + '...'


if __name__ == "__main__":
    # Test utilities
    test_text = "The bass is amazing! <br>Really punchy and deep.<br>Great for jazz."
    print(f"Original: {test_text}")
    print(f"Cleaned:  {clean_text(test_text)}")
    
    chunks = chunk_review("This is sentence one. This is sentence two. This is sentence three. This is sentence four.")
    print(f"\nChunks: {chunks}")
    
    print(f"\nSentiment test:")
    print(f"  'amazing bass, love it' + 5★ = {get_sentiment('amazing bass, love it', 5)}")
    print(f"  'harsh treble, returned' + 2★ = {get_sentiment('harsh treble, returned', 2)}")
    
    print(f"\nPopularity test:")
    print(f"  10 ratings:  {compute_popularity_score(10, 50000):.2f}")
    print(f"  1000 ratings: {compute_popularity_score(1000, 50000):.2f}")
    print(f"  10000 ratings: {compute_popularity_score(10000, 50000):.2f}")