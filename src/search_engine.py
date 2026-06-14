import pandas as pd
import numpy as np
import re
import faiss
from sentence_transformers import SentenceTransformer
from collections import defaultdict


class AudioSearchEngine:
    """Review-based audio product recommendation engine."""
    
    def __init__(self, embeddings_path, index_path, metadata_path, product_mapping_path=None):
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.index = faiss.read_index(index_path)
        self.audio_df = pd.read_csv(metadata_path)
        self.product_mapping = None
        
        if product_mapping_path:
            self.product_mapping = pd.read_csv(product_mapping_path)
            self._merge_product_names()
        
        self.exclude_words = [
            'case', 'pad', 'replacement', 'cable', 'adapter', 
            'cover', 'pouch', 'charger', 'mount', 'stand', 
            'holder', 'kids', 'children', 'toy'
        ]
    
    def _merge_product_names(self):
        """Merge real product names from mapping file."""
        if self.product_mapping is None:
            return
        
        self.audio_df = self.audio_df.merge(
            self.product_mapping[['parent_asin', 'title', 'price', 'average_rating', 'rating_number']],
            on='parent_asin',
            how='left',
            suffixes=('_old', '')
        )
        self.audio_df['title'] = self.audio_df['title'].fillna(self.audio_df['title_old'])
    
    def _clean_text(self, text):
        """Remove HTML tags and clean whitespace."""
        text = re.sub(r'<br\s*/?>', ' ', str(text))
        text = re.sub(r'<[^>]+>', '', text)
        return ' '.join(text.split())
    
    def _parse_query(self, query):
        """Extract constraints from natural language query."""
        constraints = {}
        q = query.lower()
        
        # Price
        m = re.search(r'(?:under|below|less than|budget|max|upto)\s*\$?(\d+)', q)
        if m:
            constraints['max_price'] = int(m.group(1))
        
        # Product type
        if any(w in q for w in ['iem', 'in-ear', 'in ear']):
            constraints['type'] = 'IEM'
        elif any(w in q for w in ['headphone', 'headset', 'over-ear']):
            constraints['type'] = 'headphone'
        elif any(w in q for w in ['earbud', 'tws', 'true wireless']):
            constraints['type'] = 'earbud'
        
        # Sound signature
        for sig in ['warm', 'neutral', 'bright', 'bassy', 'v-shaped', 'dark', 'analytical']:
            if sig in q:
                constraints['sound_signature'] = sig
        
        # Use case
        for use in ['gaming', 'studio', 'commuting', 'gym', 'workout', 'office', 'travel', 'jazz']:
            if use in q:
                constraints['use_case'] = use
        
        # Features
        if any(w in q for w in ['noise cancel', 'anc', 'isolating']):
            constraints['noise_cancel'] = True
        if any(w in q for w in ['wireless', 'bluetooth']):
            constraints['wireless'] = True
        
        return constraints
    
    def _get_valid_products(self, constraints):
        """Filter products based on constraints."""
        valid = set(self.audio_df['parent_asin'].unique())
        
        # Exclude accessories
        for word in self.exclude_words:
            bad = self.audio_df[
                self.audio_df['title'].str.lower().str.contains(word, na=False)
            ]['parent_asin'].unique()
            valid -= set(bad)
        
        # Type filter
        if 'type' in constraints:
            kw = {
                'IEM': 'iem|in-ear',
                'headphone': 'headphone|headset|over-ear',
                'earbud': 'earbud|tws|true wireless'
            }
            matching = self.audio_df[
                self.audio_df['chunk_text'].str.lower().str.contains(
                    kw[constraints['type']], na=False, regex=True
                )
            ]['parent_asin'].unique()
            valid &= set(matching)
        
        # Price filter
        if 'max_price' in constraints:
            try:
                affordable = self.audio_df[
                    (self.audio_df['price'].notna()) &
                    (self.audio_df['price'] != 'None') &
                    (self.audio_df['price'].astype(float) <= constraints['max_price'])
                ]['parent_asin'].unique()
                if len(affordable) > 0:
                    valid &= set(affordable)
            except:
                pass
        
        # Wireless filter
        if constraints.get('wireless'):
            matching = self.audio_df[
                self.audio_df['chunk_text'].str.lower().str.contains(
                    'wireless|bluetooth', na=False
                )
            ]['parent_asin'].unique()
            valid &= set(matching)
        
        return valid
    
    def search(self, query, k=5):
        """
        Search for audio products matching the query.
        
        Args:
            query: Natural language search query
            k: Number of results to return
            
        Returns:
            List of dicts with title, price, score, aspects, evidence
        """
        constraints = self._parse_query(query)
        valid_products = self._get_valid_products(constraints)
        
        # Semantic search
        q_vec = self.model.encode([query])
        faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, 300)
        
        # Group results by product
        products = defaultdict(lambda: {
            'chunks': [], 'scores': [], 'aspects': [], 'ratings': []
        })
        
        for score, idx in zip(scores[0], indices[0]):
            row = self.audio_df.iloc[idx]
            pid = row['parent_asin']
            
            if pid not in valid_products:
                continue
            
            # Sentiment adjustment
            if row.get('sentiment') == 'positive':
                score *= 1.1
            elif row.get('sentiment') == 'negative':
                score *= 0.5
            
            # Skip low ratings
            if row.get('rating', 5) < 3:
                continue
            
            products[pid]['chunks'].append(self._clean_text(row['chunk_text']))
            products[pid]['scores'].append(score)
            products[pid]['aspects'].append(row['aspect'])
            products[pid]['ratings'].append(row.get('rating', 0))
            products[pid]['title'] = row.get('title', 'Unknown Product')
            products[pid]['price'] = row.get('price', 'N/A')
        
        # Rank products
        ranked = []
        for pid, data in products.items():
            if len(data['chunks']) < 2:
                continue
            
            avg_score = np.mean(data['scores'])
            avg_rating = np.mean(data['ratings'])
            diversity = len(set(data['aspects'])) * 0.01
            rating_boost = (avg_rating - 3.5) * 0.02
            
            ranked.append({
                'product_id': pid,
                'title': data['title'],
                'price': data['price'],
                'score': round(avg_score + diversity + rating_boost, 3),
                'aspects': list(set(data['aspects'])),
                'num_chunks': len(data['chunks']),
                'avg_rating': round(avg_rating, 1),
                'evidence': data['chunks'][:3]
            })
        
        ranked.sort(key=lambda x: x['score'], reverse=True)
        return ranked[:k], constraints


if __name__ == "__main__":
    # Example usage
    engine = AudioSearchEngine(
        embeddings_path="embeddings/embedding_25.npy",
        index_path="embeddings/faiss_25.bin",
        metadata_path="embeddings/chunk_metadata_25.csv",
        product_mapping_path="data/product_mapping.csv"
    )
    
    queries = [
        "warm IEMs under $100 for jazz",
        "comfortable headphones for office under $200",
        "best budget earbuds under $50",
    ]
    
    for q in queries:
        print(f"\n{'='*60}")
        print(f"🔍 {q}")
        results, constraints = engine.search(q, k=3)
        print(f"Constraints: {constraints}")
        print(f"Found: {len(results)} results\n")
        for i, r in enumerate(results):
            print(f"  #{i+1} | Score: {r['score']} | 💰 ${r['price']}")
            print(f"  {r['title'][:90]}")
            for ev in r['evidence'][:2]:
                print(f"    └─ {ev[:120]}...")
            print()