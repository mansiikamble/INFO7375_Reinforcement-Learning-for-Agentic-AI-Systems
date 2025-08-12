# src/utils/writing_helpers.py
import re
from typing import Dict, Any
import numpy as np

class StyleOptimizer:
    """Optimize text style based on parameters"""
    
    def apply_style(self, text: str, style_params: Dict[str, float]) -> str:
        """Apply style transformations to text"""
        # For now, return original text
        # In a real implementation, this would use NLP models
        optimized = text
        
        # Simple transformations based on parameters
        if style_params.get('conciseness', 0) > 0.7:
            # Remove redundant words
            optimized = self.make_concise(optimized)
            
        if style_params.get('formality', 0) > 0.8:
            # Increase formality
            optimized = self.make_formal(optimized)
            
        return optimized
    
    def make_concise(self, text: str) -> str:
        """Make text more concise"""
        # Simple rule-based approach
        replacements = {
            'in order to': 'to',
            'due to the fact that': 'because',
            'at this point in time': 'now',
            'in the event that': 'if'
        }
        
        for verbose, concise in replacements.items():
            text = text.replace(verbose, concise)
            
        return text
    
    def make_formal(self, text: str) -> str:
        """Make text more formal"""
        replacements = {
            "can't": "cannot",
            "won't": "will not",
            "it's": "it is",
            "we'll": "we will"
        }
        
        for informal, formal in replacements.items():
            text = text.replace(informal, formal)
            
        return text


class ReadabilityChecker:
    """Check text readability metrics"""
    
    def flesch_kincaid_score(self, text: str) -> float:
        """Calculate Flesch-Kincaid readability score"""
        # Simplified implementation
        sentences = text.split('.')
        words = text.split()
        syllables = sum([self.count_syllables(word) for word in words])
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
            
        score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))
        return max(0, min(100, score))
    
    def count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
            
        if word.endswith('e'):
            syllable_count -= 1
            
        return max(1, syllable_count)
    
    def academic_word_percentage(self, text: str) -> float:
        """Calculate percentage of academic words"""
        academic_words = {
            'analyze', 'analysis', 'approach', 'concept', 'data', 'derive',
            'establish', 'evaluate', 'evidence', 'framework', 'hypothesis',
            'indicate', 'interpret', 'method', 'occur', 'percent', 'principle',
            'process', 'require', 'research', 'significant', 'theory', 'variable'
        }
        
        words = text.lower().split()
        if not words:
            return 0.0
            
        academic_count = sum(1 for word in words if word in academic_words)
        return (academic_count / len(words)) * 100
    
    def overall_score(self, text: str) -> float:
        """Calculate overall readability score"""
        fk_score = self.flesch_kincaid_score(text) / 100
        academic_score = min(self.academic_word_percentage(text) / 20, 1.0)
        
        return (fk_score + academic_score) / 2


class AcademicFormatter:
    """Format text for academic writing"""
    
    def evaluate_style(self, text: str) -> float:
        """Evaluate academic style adherence"""
        score = 0.0
        
        # Check for passive voice (preferred in academic writing)
        if self.has_passive_voice(text):
            score += 0.2
            
        # Check for third person
        if not self.has_first_person(text):
            score += 0.3
            
        # Check for citations
        if self.has_citations(text):
            score += 0.3
            
        # Check for formal language
        if self.is_formal(text):
            score += 0.2
            
        return score
    
    def has_passive_voice(self, text: str) -> bool:
        """Check if text contains passive voice"""
        passive_indicators = ['was', 'were', 'been', 'being', 'is conducted', 'are presented']
        return any(indicator in text.lower() for indicator in passive_indicators)
    
    def has_first_person(self, text: str) -> bool:
        """Check if text contains first person pronouns"""
        first_person = ['i ', 'we ', 'our ', 'my ', 'us ']
        text_lower = ' ' + text.lower() + ' '
        return any(pronoun in text_lower for pronoun in first_person)
    
    def has_citations(self, text: str) -> bool:
        """Check if text contains citations"""
        # Look for patterns like (Author, Year) or [1]
        citation_patterns = [
            r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s+\d{4}\)',
            r'\[\d+\]',
            r'\([A-Z][a-z]+\s+&\s+[A-Z][a-z]+,\s+\d{4}\)'
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def is_formal(self, text: str) -> bool:
        """Check if text uses formal language"""
        informal_words = ['really', 'very', 'just', 'thing', 'stuff', 'basically']
        text_lower = text.lower()
        
        informal_count = sum(1 for word in informal_words if word in text_lower)
        return informal_count < 2