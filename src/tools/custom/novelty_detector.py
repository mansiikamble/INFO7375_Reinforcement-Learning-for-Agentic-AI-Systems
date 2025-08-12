class NoveltyDetector:
    """Custom tool to detect research novelty"""
    
    def __init__(self):
        self.encoder = SentenceTransformer('allenai-specter')
        self.similarity_threshold = 0.85
        
    def detect_novelty(self, research_idea: str, 
                      existing_literature: List[Dict]) -> Dict[str, Any]:
        """Detect how novel a research idea is"""
        # Encode research idea
        idea_embedding = self.encoder.encode(research_idea)
        
        # Encode existing literature
        lit_texts = [f"{p['title']} {p['abstract']}" for p in existing_literature]
        lit_embeddings = self.encoder.encode(lit_texts)
        
        # Calculate similarities
        similarities = cosine_similarity([idea_embedding], lit_embeddings)[0]
        
        # Find most similar works
        top_k = 5
        most_similar_indices = np.argsort(similarities)[-top_k:][::-1]
        most_similar = [
            {
                'paper': existing_literature[i],
                'similarity': similarities[i]
            }
            for i in most_similar_indices
        ]
        
        # Calculate novelty score
        max_similarity = np.max(similarities)
        novelty_score = 1 - max_similarity
        
        # Extract novel aspects
        novel_aspects = self.extract_novel_aspects(
            research_idea, most_similar, idea_embedding, lit_embeddings
        )
        
        return {
            'novelty_score': novelty_score,
            'is_novel': max_similarity < self.similarity_threshold,
            'most_similar_works': most_similar,
            'novel_aspects': novel_aspects,
            'research_gap': self.identify_research_gap(research_idea, most_similar)
        }