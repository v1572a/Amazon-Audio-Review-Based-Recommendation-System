import re
from typing import Dict, Optional


class QueryParser:
    """Parse natural language queries into structured constraints."""
    
    SOUND_SIGNATURES = [
        'warm', 'neutral', 'bright', 'dark', 'bassy',
        'v-shaped', 'u-shaped', 'analytical', 'fun', 'flat'
    ]
    
    USE_CASES = [
        'gaming', 'studio', 'commuting', 'gym', 'workout',
        'office', 'travel', 'jazz', 'rock', 'classical',
        'monitoring', 'mixing', 'mastering', 'running'
    ]
    
    PRODUCT_TYPES = {
        'IEM': ['iem', 'in-ear', 'in ear', 'iems'],
        'headphone': ['headphone', 'headset', 'over-ear', 'over ear', 'on-ear', 'headphones'],
        'earbud': ['earbud', 'earbuds', 'tws', 'true wireless', 'bluetooth ear'],
    }
    
    PRICE_PATTERNS = [
        r'(?:under|below|less than|budget|max|upto|up to)\s*\$?(\d+)',
        r'(?:under|below|less than|budget|max|upto|up to)\s*(\d+)\s*(?:dollars|bucks|usd)',
        r'\$(\d+)\s*(?:budget|max|limit)',
    ]
    
    FEATURE_PATTERNS = {
        'noise_cancel': ['noise cancel', 'anc', 'noise isolating', 'noise isolation'],
        'wireless': ['wireless', 'bluetooth', 'bt', 'bluetooth'],
        'wired': ['wired', 'with wire', 'corded'],
        'open_back': ['open back', 'open-back', 'open ear'],
        'closed_back': ['closed back', 'closed-back', 'isolating'],
    }
    
    def parse(self, query: str) -> Dict:
        """
        Parse a natural language query into constraints.
        
        Args:
            query: User's search query
            
        Returns:
            Dictionary of extracted constraints
        """
        constraints = {}
        q = query.lower()
        
        # Price
        constraints['max_price'] = self._extract_price(q)
        
        # Product type
        constraints['type'] = self._extract_type(q)
        
        # Sound signature
        constraints['sound_signature'] = self._extract_sound(q)
        
        # Use case
        constraints['use_case'] = self._extract_use_case(q)
        
        # Features
        constraints.update(self._extract_features(q))
        
        return {k: v for k, v in constraints.items() if v is not None}
    
    def _extract_price(self, q: str) -> Optional[int]:
        """Extract maximum price from query."""
        for pattern in self.PRICE_PATTERNS:
            match = re.search(pattern, q)
            if match:
                return int(match.group(1))
        return None
    
    def _extract_type(self, q: str) -> Optional[str]:
        """Extract product type from query."""
        for ptype, keywords in self.PRODUCT_TYPES.items():
            if any(kw in q for kw in keywords):
                return ptype
        return None
    
    def _extract_sound(self, q: str) -> Optional[str]:
        """Extract sound signature preference."""
        for sig in self.SOUND_SIGNATURES:
            if sig in q:
                return sig
        return None
    
    def _extract_use_case(self, q: str) -> Optional[str]:
        """Extract use case from query."""
        for use in self.USE_CASES:
            if use in q:
                return use
        return None
    
    def _extract_features(self, q: str) -> Dict:
        """Extract feature flags from query."""
        features = {}
        for feature, patterns in self.FEATURE_PATTERNS.items():
            if any(p in q for p in patterns):
                features[feature] = True
        return features


if __name__ == "__main__":
    parser = QueryParser()
    
    test_queries = [
        "warm IEMs under $100 for jazz",
        "noise canceling headphones under 200 dollars for office",
        "best budget earbuds under 50",
        "bassy IEMs for gym workout",
        "neutral open back headphones for studio monitoring",
    ]
    
    print("Query Parser Test\n")
    for q in test_queries:
        print(f"Query: {q}")
        print(f"Constraints: {parser.parse(q)}\n")