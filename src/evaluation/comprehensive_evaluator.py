# src/evaluation/comprehensive_evaluator.py
import numpy as np
from typing import Dict, Any, List, Tuple
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import re
from datetime import datetime
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class PaperQualityEvaluator:
    """Evaluates various aspects of paper quality"""
    
    def __init__(self):
        self.quality_weights = {
            'structure': 0.20,
            'content': 0.25, 
            'coherence': 0.20,
            'citations': 0.15,
            'writing': 0.15,
            'novelty': 0.05
        }
        
        # Expected structure for research papers
        self.required_sections = [
            'abstract', 'introduction', 'literature_review', 
            'methodology', 'results', 'discussion', 'conclusion'
        ]
    
    def evaluate_structure_completeness(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate completeness of paper structure"""
        sections = paper.get('sections', {})
        
        # Check presence of required sections
        present_sections = 0
        section_analysis = {}
        
        for section in self.required_sections[1:]:  # Skip abstract as it's separate
            content = sections.get(section, '')
            content_length = len(str(content))
            is_present = content_length > 50
            
            section_analysis[section] = {
                'present': is_present,
                'length': content_length,
                'quality': min(1.0, content_length / 500) if is_present else 0.0
            }
            
            if is_present:
                present_sections += 1
        
        # Check abstract
        abstract = paper.get('abstract', '')
        abstract_present = len(abstract) > 100
        if abstract_present:
            present_sections += 1
            
        section_analysis['abstract'] = {
            'present': abstract_present,
            'length': len(abstract),
            'quality': min(1.0, len(abstract) / 300) if abstract_present else 0.0
        }
        
        structure_score = present_sections / len(self.required_sections)
        
        return {
            'score': structure_score,
            'present_sections': present_sections,
            'total_sections': len(self.required_sections),
            'missing_sections': [s for s in self.required_sections 
                               if s not in sections or len(str(sections.get(s, ''))) <= 50],
            'section_analysis': section_analysis,
            'completeness_ratio': structure_score
        }
    
    def evaluate_content_coherence(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate coherence between different sections"""
        sections = paper.get('sections', {})
        abstract = paper.get('abstract', '')
        
        if len(sections) < 2:
            return {'score': 0.0, 'reason': 'Insufficient sections for coherence analysis'}
        
        # Extract key terms from each section
        section_terms = {}
        all_text = abstract + ' '
        
        for section_name, content in sections.items():
            text = str(content).lower()
            # Extract meaningful terms (words longer than 4 characters, excluding common words)
            stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'more', 'very', 'when', 'come', 'than', 'only', 'over', 'also', 'back', 'after', 'first', 'well', 'work', 'life', 'year', 'years', 'other', 'many', 'most', 'some', 'such', 'take', 'make', 'good', 'much', 'even', 'give', 'still', 'right', 'same', 'great', 'little', 'down', 'through', 'between', 'before', 'against', 'under', 'while', 'should', 'could', 'would', 'being', 'doing', 'having', 'getting', 'going', 'looking', 'thinking', 'seeing'}
            
            terms = [word.strip('.,!?;:()[]"') for word in text.split() 
                    if len(word) > 4 and word.isalpha() and word.lower() not in stop_words]
            section_terms[section_name] = set(terms)
            all_text += text + ' '
        
        # Calculate pairwise coherence using Jaccard similarity
        coherence_scores = []
        section_names = list(section_terms.keys())
        coherence_matrix = {}
        
        for i in range(len(section_names)):
            coherence_matrix[section_names[i]] = {}
            for j in range(len(section_names)):
                if i != j:
                    terms1 = section_terms[section_names[i]]
                    terms2 = section_terms[section_names[j]]
                    
                    if terms1 and terms2:
                        intersection = len(terms1.intersection(terms2))
                        union = len(terms1.union(terms2))
                        jaccard_similarity = intersection / union if union > 0 else 0
                        coherence_matrix[section_names[i]][section_names[j]] = jaccard_similarity
                        
                        if i < j:  # Avoid double counting
                            coherence_scores.append(jaccard_similarity)
                    else:
                        coherence_matrix[section_names[i]][section_names[j]] = 0.0
        
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        
        # Check for consistent terminology across sections
        term_frequency = {}
        for word in all_text.split():
            clean_word = word.strip('.,!?;:()[]"').lower()
            if len(clean_word) > 4 and clean_word.isalpha():
                term_frequency[clean_word] = term_frequency.get(clean_word, 0) + 1
        
        # Key technical terms that appear consistently
        consistent_terms = [term for term, freq in term_frequency.items() if freq >= 3]
        terminology_score = min(1.0, len(consistent_terms) / 15)  # Normalize to [0,1]
        
        # Flow coherence (check for transition words)
        transition_words = ['however', 'furthermore', 'moreover', 'therefore', 'consequently', 
                          'nevertheless', 'thus', 'hence', 'subsequently', 'similarly', 'likewise']
        transition_count = sum(1 for word in transition_words if word in all_text.lower())
        flow_score = min(1.0, transition_count / 10)
        
        final_coherence = (avg_coherence * 0.5 + terminology_score * 0.3 + flow_score * 0.2)
        
        return {
            'score': final_coherence,
            'pairwise_coherence': avg_coherence,
            'terminology_consistency': terminology_score,
            'flow_score': flow_score,
            'consistent_terms': consistent_terms[:15],  # Top 15
            'coherence_matrix': coherence_matrix,
            'section_similarities': coherence_scores,
            'total_unique_terms': len(term_frequency),
            'transition_indicators': transition_count
        }
    
    def evaluate_citation_quality(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate quality and appropriateness of citations"""
        references = paper.get('references', [])
        sections = paper.get('sections', {})
        abstract = paper.get('abstract', '')
        
        if not references:
            return {
                'score': 0.0,
                'reason': 'No references found',
                'reference_count': 0,
                'citations_in_text': 0,
                'integration_score': 0.0
            }
        
        # Count citations in text
        all_text = str(sections) + ' ' + abstract
        
        # Citation patterns
        citation_patterns = [
            r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s+\d{4}\)',  # (Author, Year)
            r'\[[0-9]+\]',  # [1], [2], etc.
            r'\([A-Z][a-z]+\s+&\s+[A-Z][a-z]+,\s+\d{4}\)',  # (Author & Author, Year)
            r'\([A-Z][a-z]+\s+et\s+al\.,\s+\d{4}\)'  # (Author et al., Year)
        ]
        
        citations_in_text = 0
        citation_details = {}
        for pattern in citation_patterns:
            matches = re.findall(pattern, all_text)
            citations_in_text += len(matches)
            citation_details[pattern] = len(matches)
        
        # Reference quality analysis
        quality_indicators = {
            'recent_papers': 0,    # Papers from 2020+
            'diverse_authors': 0,  # Different first authors
            'venue_diversity': 0,  # Different publication venues
            'appropriate_count': 0  # Reasonable number of references
        }
        
        # Extract detailed information from references
        years = []
        authors = set()
        venues = set()
        
        for ref in references:
            ref_str = str(ref)
            
            # Extract year
            year_matches = re.findall(r'\((\d{4})\)', ref_str)
            if year_matches:
                year = int(year_matches[0])
                years.append(year)
                if year >= 2020:
                    quality_indicators['recent_papers'] += 1
            
            # Extract first author
            author_match = re.match(r'^([A-Z][a-z]+)', ref_str)
            if author_match:
                authors.add(author_match.group(1))
            
            # Extract potential venue information
            if 'Proceedings' in ref_str:
                venues.add('Conference')
            elif 'Journal' in ref_str:
                venues.add('Journal')
            elif 'IEEE' in ref_str or 'ACM' in ref_str:
                venues.add('Technical Publication')
        
        quality_indicators['diverse_authors'] = len(authors)
        quality_indicators['venue_diversity'] = len(venues)
        
        # Score calculation with multiple criteria
        ref_count = len(references)
        
        # 1. Recency score (favor recent papers)
        recent_ratio = quality_indicators['recent_papers'] / ref_count if ref_count > 0 else 0
        
        # 2. Diversity score (different authors)
        diversity_ratio = len(authors) / ref_count if ref_count > 0 else 0
        
        # 3. Optimal count score (10-25 references is good)
        if 10 <= ref_count <= 25:
            count_score = 1.0
        elif ref_count < 10:
            count_score = ref_count / 10
        else:
            count_score = max(0.0, 1.0 - (ref_count - 25) / 25)
        
        # 4. Citation integration (references should be cited in text)
        expected_citations = max(1, ref_count * 0.6)  # Expect 60% of refs to be cited
        integration_score = min(1.0, citations_in_text / expected_citations)
        
        # 5. Venue diversity score
        venue_score = min(1.0, len(venues) / 3)  # Good to have multiple types
        
        # Weighted final score
        final_score = (recent_ratio * 0.25 + diversity_ratio * 0.25 + 
                      count_score * 0.20 + integration_score * 0.20 + venue_score * 0.10)
        
        return {
            'score': final_score,
            'reference_count': ref_count,
            'citations_in_text': citations_in_text,
            'recent_papers': quality_indicators['recent_papers'],
            'diverse_authors': len(authors),
            'venue_types': len(venues),
            'recent_ratio': recent_ratio,
            'diversity_ratio': diversity_ratio,
            'count_score': count_score,
            'integration_score': integration_score,
            'venue_score': venue_score,
            'years_range': [min(years), max(years)] if years else [0, 0],
            'citation_patterns': citation_details,
            'quality_breakdown': {
                'recency': recent_ratio * 0.25,
                'diversity': diversity_ratio * 0.25,
                'count': count_score * 0.20,
                'integration': integration_score * 0.20,
                'venues': venue_score * 0.10
            }
        }
    
    def evaluate_writing_quality(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate writing quality and style"""
        all_text = str(paper.get('abstract', ''))
        for content in paper.get('sections', {}).values():
            all_text += ' ' + str(content)
        
        if len(all_text) < 100:
            return {
                'score': 0.0,
                'reason': 'Insufficient text for analysis',
                'text_length': len(all_text)
            }
        
        # Basic readability metrics
        sentences = [s.strip() for s in all_text.split('.') if s.strip()]
        words = [w.strip('.,!?;:()[]"') for w in all_text.split() if w.strip()]
        
        # 1. Sentence length analysis (15-20 words optimal for academic writing)
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        sentence_length_score = 1.0 - min(1.0, abs(avg_sentence_length - 17.5) / 17.5)
        
        # 2. Vocabulary diversity and sophistication
        unique_words = len(set(word.lower() for word in words if word.isalpha()))
        diversity_ratio = unique_words / len(words) if words else 0
        
        # 3. Academic writing indicators
        academic_indicators = [
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'nevertheless', 'thus', 'hence', 'accordingly', 'subsequently',
            'demonstrate', 'analyze', 'evaluate', 'investigate', 'propose',
            'significant', 'substantial', 'comprehensive', 'systematic',
            'empirical', 'theoretical', 'methodology', 'hypothesis', 'evidence'
        ]
        
        academic_count = sum(1 for indicator in academic_indicators 
                           if indicator in all_text.lower())
        academic_score = min(1.0, academic_count / 15)  # Normalize
        
        # 4. Technical terminology appropriateness
        technical_terms = [
            'algorithm', 'model', 'framework', 'approach', 'method',
            'system', 'analysis', 'evaluation', 'performance', 'optimization',
            'learning', 'training', 'testing', 'validation', 'experimental'
        ]
        
        technical_count = sum(1 for term in technical_terms if term in all_text.lower())
        technical_score = min(1.0, technical_count / 10)
        
        # 5. Avoid informal language
        informal_words = ['really', 'very', 'quite', 'pretty', 'kind of', 'sort of', 
                         'basically', 'actually', 'literally', 'totally', 'absolutely']
        informal_count = sum(1 for word in informal_words if word in all_text.lower())
        informal_penalty = min(0.4, informal_count * 0.1)
        
        # 6. Grammar and style checks
        grammar_score = 1.0
        grammar_issues = []
        
        # Check for common issues
        if '  ' in all_text:  # Double spaces
            grammar_score -= 0.05
            grammar_issues.append('Double spaces detected')
            
        if all_text.count('(') != all_text.count(')'):  # Unmatched parentheses
            grammar_score -= 0.1
            grammar_issues.append('Unmatched parentheses')
            
        if all_text.count('"') % 2 != 0:  # Unmatched quotes
            grammar_score -= 0.05
            grammar_issues.append('Unmatched quotation marks')
        
        # Check for proper capitalization after periods
        sentences_check = all_text.split('. ')
        capitalization_errors = sum(1 for sent in sentences_check[1:] 
                                  if sent and sent[0].islower())
        if capitalization_errors > 0:
            grammar_score -= min(0.2, capitalization_errors * 0.02)
            grammar_issues.append(f'{capitalization_errors} capitalization errors')
        
        # 7. Passive voice analysis (some is good in academic writing)
        passive_indicators = ['was', 'were', 'been', 'being', 'is conducted', 'are presented', 'were analyzed']
        passive_count = sum(1 for indicator in passive_indicators if indicator in all_text.lower())
        passive_ratio = passive_count / len(words) if words else 0
        
        # Optimal passive voice ratio for academic writing is 20-40%
        if 0.2 <= passive_ratio <= 0.4:
            passive_score = 1.0
        else:
            passive_score = 1.0 - abs(passive_ratio - 0.3) / 0.3
        
        # 8. Readability score (Flesch-Kincaid approximation)
        def count_syllables(word):
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
        
        total_syllables = sum(count_syllables(word) for word in words if word.isalpha())
        
        if sentences and words:
            flesch_kincaid = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (total_syllables / len(words))
            # Normalize to 0-1 scale (academic papers typically score 30-50)
            readability_score = min(1.0, max(0.0, (flesch_kincaid - 20) / 40))
        else:
            readability_score = 0.0
        
        # Combine all scores
        final_score = (sentence_length_score * 0.15 + 
                      diversity_ratio * 0.15 + 
                      academic_score * 0.20 + 
                      technical_score * 0.15 + 
                      grammar_score * 0.15 + 
                      passive_score * 0.10 + 
                      readability_score * 0.10 - 
                      informal_penalty)
        
        return {
            'score': max(0.0, min(1.0, final_score)),
            'metrics': {
                'avg_sentence_length': avg_sentence_length,
                'vocabulary_diversity': diversity_ratio,
                'academic_indicators': academic_count,
                'technical_terms': technical_count,
                'informal_count': informal_count,
                'grammar_score': grammar_score,
                'passive_ratio': passive_ratio,
                'readability_score': readability_score
            },
            'statistics': {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'unique_words': unique_words,
                'syllable_count': total_syllables
            },
            'issues': grammar_issues,
            'score_breakdown': {
                'sentence_length': sentence_length_score * 0.15,
                'vocabulary': diversity_ratio * 0.15,
                'academic_style': academic_score * 0.20,
                'technical_content': technical_score * 0.15,
                'grammar': grammar_score * 0.15,
                'passive_voice': passive_score * 0.10,
                'readability': readability_score * 0.10,
                'informal_penalty': -informal_penalty
            }
        }
    
    def evaluate_novelty(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate novelty and contribution claims"""
        all_text = str(paper.get('abstract', ''))
        for content in paper.get('sections', {}).values():
            all_text += ' ' + str(content)
        
        text_lower = all_text.lower()
        
        # Novelty indicators with different strengths
        novelty_terms = {
            'strong': ['novel', 'new', 'first', 'unprecedented', 'breakthrough', 'innovative', 
                      'revolutionary', 'pioneering', 'groundbreaking', 'original'],
            'moderate': ['improved', 'enhanced', 'better', 'advanced', 'state-of-the-art', 
                        'superior', 'optimized', 'refined', 'updated', 'modernized'],
            'weak': ['different', 'alternative', 'another', 'modified', 'adapted', 
                    'variant', 'version', 'approach', 'method', 'technique']
        }
        
        novelty_score = 0.0
        term_counts = {'strong': 0, 'moderate': 0, 'weak': 0}
        term_details = {'strong': [], 'moderate': [], 'weak': []}
        
        for category, terms in novelty_terms.items():
            for term in terms:
                count = text_lower.count(term)
                term_counts[category] += count
                if count > 0:
                    term_details[category].append((term, count))
                
                # Weight different types of novelty claims
                if category == 'strong':
                    novelty_score += count * 0.4
                elif category == 'moderate':
                    novelty_score += count * 0.25
                else:  # weak
                    novelty_score += count * 0.1
        
        # Normalize novelty score
        novelty_score = min(1.0, novelty_score / 8)
        
        # Check for explicit contribution claims
        contribution_phrases = [
            'our contribution', 'we propose', 'our approach', 'our method',
            'we introduce', 'our work', 'our solution', 'we present',
            'this work contributes', 'our key insight', 'we demonstrate',
            'our findings', 'we show', 'we establish'
        ]
        
        contribution_count = sum(1 for phrase in contribution_phrases 
                               if phrase in text_lower)
        contribution_score = min(1.0, contribution_count / 5)
        
        # Check for problem identification (good research identifies problems)
        problem_phrases = [
            'challenge', 'problem', 'limitation', 'difficulty', 'issue',
            'bottleneck', 'constraint', 'obstacle', 'shortcoming', 'gap'
        ]
        
        problem_count = sum(1 for phrase in problem_phrases if phrase in text_lower)
        problem_awareness_score = min(1.0, problem_count / 5)
        
        # Check for solution claims
        solution_phrases = [
            'solution', 'solve', 'address', 'overcome', 'resolve',
            'tackle', 'handle', 'deal with', 'mitigate', 'improve'
        ]
        
        solution_count = sum(1 for phrase in solution_phrases if phrase in text_lower)
        solution_score = min(1.0, solution_count / 5)
        
        # Final novelty calculation
        final_novelty = (novelty_score * 0.4 + 
                        contribution_score * 0.25 + 
                        problem_awareness_score * 0.20 + 
                        solution_score * 0.15)
        
        return {
            'score': final_novelty,
            'novelty_terms': term_counts,
            'novelty_details': term_details,
            'contribution_claims': contribution_count,
            'problem_awareness': problem_count,
            'solution_claims': solution_count,
            'strong_novelty_indicators': term_counts['strong'],
            'total_novelty_terms': sum(term_counts.values()),
            'score_breakdown': {
                'novelty_terms': novelty_score * 0.4,
                'contributions': contribution_score * 0.25,
                'problem_awareness': problem_awareness_score * 0.20,
                'solutions': solution_score * 0.15
            }
        }
    
    def evaluate_overall_quality(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive quality evaluation"""
        evaluations = {
            'structure': self.evaluate_structure_completeness(paper),
            'coherence': self.evaluate_content_coherence(paper),
            'citations': self.evaluate_citation_quality(paper),
            'writing': self.evaluate_writing_quality(paper),
            'novelty': self.evaluate_novelty(paper)
        }
        
        # Calculate weighted overall score
        overall_score = 0.0
        for aspect, weight in self.quality_weights.items():
            if aspect in evaluations:
                aspect_score = evaluations[aspect]['score']
                overall_score += aspect_score * weight
        
        # Add content score (combination of structure and coherence)
        content_score = (evaluations['structure']['score'] * 0.5 + 
                        evaluations['coherence']['score'] * 0.5)
        
        # Calculate quality grade
        def get_quality_grade(score):
            if score >= 0.9:
                return 'A+'
            elif score >= 0.8:
                return 'A'
            elif score >= 0.7:
                return 'B+'
            elif score >= 0.6:
                return 'B'
            elif score >= 0.5:
                return 'C+'
            elif score >= 0.4:
                return 'C'
            else:
                return 'D'
        
        return {
            'overall_score': overall_score,
            'quality_grade': get_quality_grade(overall_score),
            'aspect_scores': {aspect: eval_data['score'] for aspect, eval_data in evaluations.items()},
            'detailed_evaluations': evaluations,
            'content_score': content_score,
            'weights_used': self.quality_weights,
            'strengths': self._identify_strengths(evaluations),
            'weaknesses': self._identify_weaknesses(evaluations),
            'improvement_suggestions': self._generate_improvement_suggestions(evaluations)
        }
    
    def _identify_strengths(self, evaluations: Dict[str, Any]) -> List[str]:
        """Identify strengths based on evaluation scores"""
        strengths = []
        
        for aspect, eval_data in evaluations.items():
            score = eval_data['score']
            if score >= 0.8:
                strengths.append(f"Strong {aspect}: {score:.3f}")
            elif score >= 0.7:
                strengths.append(f"Good {aspect}: {score:.3f}")
        
        return strengths
    
    def _identify_weaknesses(self, evaluations: Dict[str, Any]) -> List[str]:
        """Identify weaknesses based on evaluation scores"""
        weaknesses = []
        
        for aspect, eval_data in evaluations.items():
            score = eval_data['score']
            if score < 0.5:
                weaknesses.append(f"Poor {aspect}: {score:.3f}")
            elif score < 0.7:
                weaknesses.append(f"Needs improvement in {aspect}: {score:.3f}")
        
        return weaknesses
    
    def _generate_improvement_suggestions(self, evaluations: Dict[str, Any]) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        # Structure improvements
        structure_eval = evaluations.get('structure', {})
        if structure_eval.get('score', 0) < 0.8:
            missing = structure_eval.get('missing_sections', [])
            if missing:
                suggestions.append(f"Add missing sections: {', '.join(missing)}")
        
        # Citation improvements
        citation_eval = evaluations.get('citations', {})
        if citation_eval.get('score', 0) < 0.7:
            if citation_eval.get('reference_count', 0) < 10:
                suggestions.append("Increase number of references (aim for 10-25)")
            if citation_eval.get('integration_score', 0) < 0.5:
                suggestions.append("Better integrate citations into the text")
            if citation_eval.get('recent_ratio', 0) < 0.3:
                suggestions.append("Include more recent papers (2020+)")
        
        # Writing improvements
        writing_eval = evaluations.get('writing', {})
        if writing_eval.get('score', 0) < 0.7:
            metrics = writing_eval.get('metrics', {})
            if metrics.get('academic_indicators', 0) < 10:
                suggestions.append("Use more academic language and formal expressions")
            if metrics.get('informal_count', 0) > 3:
                suggestions.append("Reduce informal language and expressions")
            if len(writing_eval.get('issues', [])) > 0:
                suggestions.append("Fix grammar and formatting issues")
        
        # Coherence improvements
        coherence_eval = evaluations.get('coherence', {})
        if coherence_eval.get('score', 0) < 0.6:
            if coherence_eval.get('terminology_consistency', 0) < 0.5:
                suggestions.append("Use more consistent terminology across sections")
            if coherence_eval.get('flow_score', 0) < 0.5:
                suggestions.append("Add more transition words and improve section flow")
        
        # Novelty improvements
        novelty_eval = evaluations.get('novelty', {})
        if novelty_eval.get('score', 0) < 0.5:
            if novelty_eval.get('contribution_claims', 0) < 2:
                suggestions.append("Clearly state contributions and novel aspects")
            if novelty_eval.get('problem_awareness', 0) < 2:
                suggestions.append("Better identify and articulate the problems being addressed")
        
        return suggestions

class ComprehensiveEvaluator:
    """Comprehensive evaluation system for multi-agent paper generation"""
    
    def __init__(self, results_dir: Path = None):
        self.results_dir = results_dir or Path('experiments/evaluation')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.quality_evaluator = PaperQualityEvaluator()
        
        # Benchmark test cases for consistent evaluation
        self.benchmark_cases = [
            {
                'id': 'ml_survey_easy',
                'title': 'Survey of Machine Learning Methods',
                'research_question': 'What are the main machine learning approaches?',
                'paper_type': 'survey',
                'venue': 'Conference',
                'difficulty': 'easy',
                'expected_sections': 5,
                'expected_quality': 0.7,
                'max_pages': 8
            },
            {
                'id': 'rl_research_medium',
                'title': 'Reinforcement Learning for Autonomous Systems',
                'research_question': 'How can RL improve autonomous decision making?',
                'paper_type': 'research',
                'venue': 'Journal',
                'difficulty': 'medium',
                'expected_sections': 6,
                'expected_quality': 0.8,
                'max_pages': 12
            },
            {
                'id': 'meta_learning_hard',
                'title': 'Meta-Learning for Few-Shot Scientific Discovery',
                'research_question': 'Can meta-learning accelerate scientific breakthroughs?',
                'paper_type': 'research',
                'venue': 'Nature',
                'difficulty': 'hard',
                'expected_sections': 6,
                'expected_quality': 0.85,
                'max_pages': 15
            },
            {
                'id': 'interdisciplinary_hard',
                'title': 'Neuro-Symbolic AI for Climate Science Applications',
                'research_question': 'How can AI bridge symbolic reasoning and neural learning for climate modeling?',
                'paper_type': 'research',
                'venue': 'Journal',
                'difficulty': 'hard',
                'expected_sections': 6,
                'expected_quality': 0.85,
                'max_pages': 16
            }
        ]
    
    def evaluate_paper_quality(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single paper comprehensively"""
        return self.quality_evaluator.evaluate_overall_quality(paper)
    
    def evaluate_learning_progress(self, training_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate learning progress from training metrics"""
        if not training_metrics:
            return {'error': 'No training metrics provided'}
        
        # Extract key metrics
        rewards = training_metrics.get('advanced_rewards', training_metrics.get('episode_rewards', []))
        quality_scores = training_metrics.get('paper_quality_scores', [])
        coordination_scores = training_metrics.get('agent_coordination_scores', [])
        sections_completed = training_metrics.get('sections_completed', [])
        efficiency_scores = training_metrics.get('efficiency_scores', [])
        
        if not rewards:
            return {'error': 'No reward data available'}
        
        episodes = len(rewards)
        
        # Calculate improvement trends using linear regression
        def calculate_trend(values):
            if len(values) < 2:
                return 0.0
            x = np.arange(len(values))
            return np.polyfit(x, values, 1)[0]  # Linear slope
        
        # Learning efficiency (how quickly did it improve?)
        def calculate_learning_efficiency(values, threshold=0.8):
            if not values:
                return 0.0
            max_val = max(values)
            target = threshold * max_val
            for i, val in enumerate(values):
                if val >= target:
                    return (episodes - i) / episodes  # Efficiency: reach target early
            return 0.0
        
        # Stability analysis (how consistent are the improvements?)
        def calculate_stability(values, window=10):
            if len(values) < window:
                return np.std(values) / (np.mean(values) + 1e-8) if values else 0.0
            
            recent_values = values[-window:]
            return 1.0 - (np.std(recent_values) / (np.mean(recent_values) + 1e-8))
        
        # Convergence analysis
        def analyze_convergence(values, patience=10):
            if len(values) < patience * 2:
                return {'converged': False, 'plateau_length': 0}
            
            # Check for plateau (no significant improvement)
            recent_mean = np.mean(values[-patience:])
            previous_mean = np.mean(values[-patience*2:-patience])
            
            improvement = recent_mean - previous_mean
            converged = abs(improvement) < 0.01  # Very small improvement
            
            # Count plateau length
            plateau_length = 0
            for i in range(len(values) - 1, 0, -1):
                if abs(values[i] - values[i-1]) < 0.01:
                    plateau_length += 1
                else:
                    break
            
            return {
                'converged': converged,
                'plateau_length': plateau_length,
                'recent_improvement': improvement
            }
        
        learning_analysis = {
            'trends': {
                'reward_trend': calculate_trend(rewards),
                'quality_trend': calculate_trend(quality_scores),
                'coordination_trend': calculate_trend(coordination_scores),
                'sections_trend': calculate_trend(sections_completed)
            },
            
            'efficiency': {
                'reward_efficiency': calculate_learning_efficiency(rewards),
                'quality_efficiency': calculate_learning_efficiency(quality_scores)
            },
            
            'stability': {
                'reward_stability': calculate_stability(rewards),
                'quality_stability': calculate_stability(quality_scores),
                'coordination_stability': calculate_stability(coordination_scores)
            },
            
            'convergence': {
                'reward_convergence': analyze_convergence(rewards),
                'quality_convergence': analyze_convergence(quality_scores)
            },
            
            'final_performance': {
                'reward': rewards[-1] if rewards else 0,
                'quality': quality_scores[-1] if quality_scores else 0,
                'coordination': coordination_scores[-1] if coordination_scores else 0,
                'sections': sections_completed[-1] if sections_completed else 0,
                'efficiency': efficiency_scores[-1] if efficiency_scores else 0
            },
            
            'improvement_metrics': {
                'total_reward_improvement': rewards[-1] - rewards[0] if len(rewards) > 1 else 0,
                'total_quality_improvement': quality_scores[-1] - quality_scores[0] if len(quality_scores) > 1 else 0,
                'relative_reward_improvement': ((rewards[-1] - rewards[0]) / max(abs(rewards[0]), 0.001)) if len(rewards) > 1 else 0,
                'relative_quality_improvement': ((quality_scores[-1] - quality_scores[0]) / max(quality_scores[0], 0.001)) if len(quality_scores) > 1 else 0,
                'peak_performance': {
                    'best_reward': max(rewards) if rewards else 0,
                    'best_quality': max(quality_scores) if quality_scores else 0,
                    'best_coordination': max(coordination_scores) if coordination_scores else 0,
                    'best_sections': max(sections_completed) if sections_completed else 0
                }
            },
            
            'learning_characteristics': {
                'learning_speed': self._calculate_learning_speed(rewards),
                'consistency': self._calculate_consistency(quality_scores),
                'exploration_effectiveness': self._analyze_exploration(coordination_scores)
            }
        }
        
        return learning_analysis
    
    def _calculate_learning_speed(self, values: List[float]) -> str:
        """Categorize learning speed"""
        if len(values) < 10:
            return 'insufficient_data'
        
        # Check how quickly it reaches 80% of final performance
        final_value = values[-1]
        target = 0.8 * final_value
        
        for i, val in enumerate(values):
            if val >= target:
                speed_ratio = i / len(values)
                if speed_ratio < 0.2:
                    return 'very_fast'
                elif speed_ratio < 0.4:
                    return 'fast'
                elif speed_ratio < 0.6:
                    return 'moderate'
                elif speed_ratio < 0.8:
                    return 'slow'
                else:
                    return 'very_slow'
        
        return 'no_convergence'
    
    def _calculate_consistency(self, values: List[float]) -> float:
        """Calculate consistency of performance"""
        if len(values) < 5:
            return 0.0
        
        # Use coefficient of variation (lower is more consistent)
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if mean_val == 0:
            return 0.0
        
        cv = std_val / mean_val
        consistency = max(0.0, 1.0 - cv)  # Convert to consistency score
        
        return consistency
    
    def _analyze_exploration(self, coordination_scores: List[float]) -> Dict[str, Any]:
        """Analyze exploration effectiveness"""
        if not coordination_scores:
            return {'effectiveness': 'no_data'}
        
        # Look for variety in coordination scores (good exploration)
        unique_scores = len(set(round(score, 2) for score in coordination_scores))
        total_scores = len(coordination_scores)
        
        diversity_ratio = unique_scores / total_scores if total_scores > 0 else 0
        
        if diversity_ratio > 0.7:
            effectiveness = 'high'
        elif diversity_ratio > 0.4:
            effectiveness = 'medium'
        else:
            effectiveness = 'low'
        
        return {
            'effectiveness': effectiveness,
            'diversity_ratio': diversity_ratio,
            'unique_behaviors': unique_scores,
            'total_episodes': total_scores
        }
    
    def run_benchmark_evaluation(self, orchestrator) -> Dict[str, Any]:
        """Run evaluation on benchmark test cases"""
        print("🧪 Running Comprehensive Benchmark Evaluation")
        print("=" * 60)
        
        benchmark_results = []
        
        for case in self.benchmark_cases:
            print(f"\n📋 Evaluating: {case['title']}")
            print(f"   Difficulty: {case['difficulty'].upper()} | Type: {case['paper_type']} | Venue: {case['venue']}")
            
            # Set agents to evaluation mode (disable exploration)
            original_epsilons = {}
            for name, agent in orchestrator.agents.items():
                if agent is not None and hasattr(agent, 'epsilon'):
                    original_epsilons[name] = agent.epsilon
                    agent.epsilon = 0.0  # No exploration during evaluation
            
            try:
                start_time = time.time()
                result = orchestrator.orchestrate_paper_generation(case)
                generation_time = time.time() - start_time
                
                # Comprehensive paper evaluation
                quality_eval = self.evaluate_paper_quality(result['paper'])
                
                # Calculate additional metrics
                paper_length = sum(len(str(content)) for content in result['paper'].get('sections', {}).values())
                paper_length += len(str(result['paper'].get('abstract', '')))
                
                benchmark_result = {
                    'case_id': case['id'],
                    'title': case['title'],
                    'difficulty': case['difficulty'],
                    'paper_type': case['paper_type'],
                    'venue': case['venue'],
                    'generation_time': generation_time,
                    'paper_length': paper_length,
                    'sections_completed': result['metrics']['sections_completed'],
                    'expected_sections': case['expected_sections'],
                    'quality_score': quality_eval['overall_score'],
                    'quality_grade': quality_eval['quality_grade'],
                    'expected_quality': case['expected_quality'],
                    'quality_details': quality_eval['aspect_scores'],
                    'detailed_analysis': quality_eval['detailed_evaluations'],
                    'meets_expectations': {
                        'sections': result['metrics']['sections_completed'] >= case['expected_sections'] * 0.8,
                        'quality': quality_eval['overall_score'] >= case['expected_quality'] * 0.9,
                        'overall': (result['metrics']['sections_completed'] >= case['expected_sections'] * 0.8 and 
                                  quality_eval['overall_score'] >= case['expected_quality'] * 0.9)
                    },
                    'performance_ratios': {
                        'quality_ratio': quality_eval['overall_score'] / case['expected_quality'],
                        'sections_ratio': result['metrics']['sections_completed'] / case['expected_sections'],
                        'efficiency': result['metrics']['sections_completed'] / max(generation_time, 0.1)
                    },
                    'strengths': quality_eval.get('strengths', []),
                    'weaknesses': quality_eval.get('weaknesses', []),
                    'suggestions': quality_eval.get('improvement_suggestions', [])
                }
                
                benchmark_results.append(benchmark_result)
                
                print(f"   ✅ Quality: {quality_eval['overall_score']:.3f} (target: {case['expected_quality']:.3f}) - Grade: {quality_eval['quality_grade']}")
                print(f"   ✅ Sections: {result['metrics']['sections_completed']:.1f}/{case['expected_sections']}")
                print(f"   ✅ Time: {generation_time:.2f}s | Length: {paper_length:,} chars")
                print(f"   ✅ Meets expectations: {'YES' if benchmark_result['meets_expectations']['overall'] else 'NO'}")
                
                # Show top strengths and weaknesses
                if quality_eval.get('strengths'):
                    print(f"   💪 Strengths: {', '.join(quality_eval['strengths'][:2])}")
                if quality_eval.get('weaknesses'):
                    print(f"   ⚠️  Weaknesses: {', '.join(quality_eval['weaknesses'][:2])}")
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                benchmark_results.append({
                    'case_id': case['id'],
                    'title': case['title'],
                    'difficulty': case['difficulty'],
                    'error': str(e),
                    'generation_time': 0,
                    'quality_score': 0,
                    'sections_completed': 0,
                    'meets_expectations': {'overall': False}
                })
            
            # Restore exploration rates
            for name, epsilon in original_epsilons.items():
                if name in orchestrator.agents and orchestrator.agents[name] is not None:
                    orchestrator.agents[name].epsilon = epsilon
        
        # Calculate comprehensive benchmark summary
        successful_cases = [r for r in benchmark_results if 'error' not in r]
        
        if successful_cases:
            summary = {
                'total_cases': len(self.benchmark_cases),
                'successful_cases': len(successful_cases),
                'success_rate': len(successful_cases) / len(self.benchmark_cases),
                'performance_metrics': {
                    'average_quality': np.mean([r['quality_score'] for r in successful_cases]),
                    'average_sections': np.mean([r['sections_completed'] for r in successful_cases]),
                    'average_time': np.mean([r['generation_time'] for r in successful_cases]),
                    'average_length': np.mean([r['paper_length'] for r in successful_cases])
                },
                'expectation_analysis': {
                    'quality_expectations_met': sum(1 for r in successful_cases 
                                                   if r.get('meets_expectations', {}).get('quality', False)),
                    'section_expectations_met': sum(1 for r in successful_cases 
                                                  if r.get('meets_expectations', {}).get('sections', False)),
                    'overall_expectations_met': sum(1 for r in successful_cases 
                                                  if r.get('meets_expectations', {}).get('overall', False))
                },
                'difficulty_analysis': self._analyze_difficulty_performance(successful_cases),
                'grade_distribution': self._calculate_grade_distribution(successful_cases)
            }
            
            print(f"\n📊 Comprehensive Benchmark Summary:")
            print(f"   Success Rate: {summary['success_rate']:.1%}")
            print(f"   Average Quality: {summary['performance_metrics']['average_quality']:.3f}")
            print(f"   Average Sections: {summary['performance_metrics']['average_sections']:.1f}")
            print(f"   Overall Expectations Met: {summary['expectation_analysis']['overall_expectations_met']}/{len(successful_cases)}")
            
            # Show grade distribution
            grade_dist = summary['grade_distribution']
            print(f"   Grade Distribution: {', '.join([f'{grade}: {count}' for grade, count in grade_dist.items()])}")
        else:
            summary = {'error': 'No successful benchmark cases'}
        
        # Save comprehensive benchmark results
        benchmark_path = self.results_dir / f'benchmark_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(benchmark_path, 'w') as f:
            json.dump({
                'summary': summary,
                'detailed_results': benchmark_results,
                'benchmark_cases': self.benchmark_cases,
                'evaluation_timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        return {
            'summary': summary,
            'detailed_results': benchmark_results
        }
    
    def _analyze_difficulty_performance(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance across different difficulty levels"""
        difficulty_groups = {}
        
        for result in results:
            difficulty = result.get('difficulty', 'unknown')
            if difficulty not in difficulty_groups:
                difficulty_groups[difficulty] = []
            difficulty_groups[difficulty].append(result)
        
        difficulty_analysis = {}
        for difficulty, cases in difficulty_groups.items():
            if cases:
                difficulty_analysis[difficulty] = {
                    'case_count': len(cases),
                    'average_quality': np.mean([c['quality_score'] for c in cases]),
                    'average_sections': np.mean([c['sections_completed'] for c in cases]),
                    'average_time': np.mean([c['generation_time'] for c in cases]),
                    'success_rate': sum(1 for c in cases if c.get('meets_expectations', {}).get('overall', False)) / len(cases)
                }
        
        return difficulty_analysis
    
    def _calculate_grade_distribution(self, results: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of quality grades"""
        grades = {}
        
        for result in results:
            grade = result.get('quality_grade', 'D')
            grades[grade] = grades.get(grade, 0) + 1
        
        return grades
    
    def run_ablation_studies(self, orchestrator, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive ablation studies"""
        print("🔬 Running Comprehensive Ablation Studies")
        print("=" * 60)
        
        ablation_configs = [
            {
                'name': 'baseline_full',
                'description': 'Full system with all agents and hybrid coordination',
                'disabled_agents': [],
                'coordination_mode': 'hybrid',
                'modifications': {}
            },
            {
                'name': 'no_literature_agent',
                'description': 'System without literature review agent',
                'disabled_agents': ['literature'],
                'coordination_mode': 'hybrid',
                'modifications': {'skip_literature': True}
            },
            {
                'name': 'no_methodology_agent',
                'description': 'System without methodology design agent',
                'disabled_agents': ['methodology'],
                'coordination_mode': 'hybrid',
                'modifications': {'skip_methodology': True}
            },
            {
                'name': 'no_writing_agent',
                'description': 'System without specialized writing agent',
                'disabled_agents': ['writing'],
                'coordination_mode': 'hybrid',
                'modifications': {'basic_writing': True}
            },
            {
                'name': 'no_analysis_agent',
                'description': 'System without analysis agent',
                'disabled_agents': ['analysis'],
                'coordination_mode': 'hybrid',
                'modifications': {}
            },
            {
                'name': 'sequential_only',
                'description': 'Sequential coordination only (no parallelism)',
                'disabled_agents': [],
                'coordination_mode': 'sequential',
                'modifications': {'force_sequential': True}
            },
            {
                'name': 'parallel_only',
                'description': 'Parallel coordination only (no sequencing)',
                'disabled_agents': [],
                'coordination_mode': 'parallel',
                'modifications': {'force_parallel': True}
            },
            {
                'name': 'single_agent_literature',
                'description': 'Single agent system (literature only)',
                'disabled_agents': ['methodology', 'analysis', 'writing'],
                'coordination_mode': 'sequential',
                'modifications': {'single_agent': 'literature'}
            },
            {
                'name': 'dual_agent_lit_method',
                'description': 'Two-agent system (literature + methodology)',
                'disabled_agents': ['analysis', 'writing'],
                'coordination_mode': 'hybrid',
                'modifications': {'dual_agents': ['literature', 'methodology']}
            },
            {
                'name': 'no_rl_random',
                'description': 'No reinforcement learning (random decisions)',
                'disabled_agents': [],
                'coordination_mode': 'random',
                'modifications': {'disable_learning': True}
            }
        ]
        
        # Multiple test cases for robustness
        test_cases = [
            {
                'title': 'Multi-Agent Systems for Research Paper Generation',
                'research_question': 'How effective is multi-agent coordination?',
                'paper_type': 'research',
                'venue': 'Conference',
                'difficulty': 'medium'
            },
            {
                'title': 'Deep Learning Survey and Applications',
                'research_question': 'What are the current trends in deep learning?',
                'paper_type': 'survey',
                'venue': 'Journal',
                'difficulty': 'easy'
            },
            {
                'title': 'Novel Approaches to Automated Scientific Writing',
                'research_question': 'Can AI systems generate high-quality research papers?',
                'paper_type': 'research',
                'venue': 'Nature',
                'difficulty': 'hard'
            }
        ]
        
        ablation_results = []
        
        for config_setup in ablation_configs:
            print(f"\n🧪 Testing Configuration: {config_setup['name']}")
            print(f"   {config_setup['description']}")
            
            config_results = []
            
            for test_idx, test_case in enumerate(test_cases):
                print(f"   📝 Test Case {test_idx + 1}: {test_case['title'][:40]}...")
                
                try:
                    # Create modified orchestrator for this configuration
                    modified_orchestrator = self._create_ablation_orchestrator(
                        orchestrator, config_setup
                    )
                    
                    # Run test
                    start_time = time.time()
                    result = modified_orchestrator.orchestrate_paper_generation(test_case)
                    generation_time = time.time() - start_time
                    
                    # Evaluate result
                    quality_eval = self.evaluate_paper_quality(result['paper'])
                    
                    test_result = {
                        'test_case_id': test_idx,
                        'test_title': test_case['title'],
                        'quality_score': quality_eval['overall_score'],
                        'quality_grade': quality_eval['quality_grade'],
                        'sections_completed': result['metrics']['sections_completed'],
                        'generation_time': generation_time,
                        'coordination_score': result['metrics']['agent_coordination_score'],
                        'paper_length': sum(len(str(content)) for content in result['paper'].get('sections', {}).values()),
                        'quality_breakdown': quality_eval['aspect_scores']
                    }
                    
                    config_results.append(test_result)
                    
                    print(f"      ✅ Quality: {quality_eval['overall_score']:.3f} ({quality_eval['quality_grade']})")
                    print(f"      ✅ Sections: {result['metrics']['sections_completed']:.1f}/6")
                    print(f"      ✅ Time: {generation_time:.2f}s")
                    
                except Exception as e:
                    print(f"      ❌ Failed: {str(e)}")
                    config_results.append({
                        'test_case_id': test_idx,
                        'test_title': test_case['title'],
                        'error': str(e),
                        'quality_score': 0,
                        'sections_completed': 0,
                        'generation_time': 0
                    })
            
            # Calculate average performance for this configuration
            successful_tests = [r for r in config_results if 'error' not in r]
            
            if successful_tests:
                config_summary = {
                    'configuration': config_setup['name'],
                    'description': config_setup['description'],
                    'disabled_agents': config_setup['disabled_agents'],
                    'coordination_mode': config_setup['coordination_mode'],
                    'test_cases_run': len(test_cases),
                    'successful_tests': len(successful_tests),
                    'average_performance': {
                        'quality_score': np.mean([r['quality_score'] for r in successful_tests]),
                        'sections_completed': np.mean([r['sections_completed'] for r in successful_tests]),
                        'generation_time': np.mean([r['generation_time'] for r in successful_tests]),
                        'coordination_score': np.mean([r['coordination_score'] for r in successful_tests])
                    },
                    'performance_std': {
                        'quality_std': np.std([r['quality_score'] for r in successful_tests]),
                        'sections_std': np.std([r['sections_completed'] for r in successful_tests]),
                        'time_std': np.std([r['generation_time'] for r in successful_tests])
                    },
                    'test_results': config_results
                }
                
                avg_perf = config_summary['average_performance']
                print(f"   📊 Average Performance:")
                print(f"      Quality: {avg_perf['quality_score']:.3f} ± {config_summary['performance_std']['quality_std']:.3f}")
                print(f"      Sections: {avg_perf['sections_completed']:.1f} ± {config_summary['performance_std']['sections_std']:.1f}")
                print(f"      Time: {avg_perf['generation_time']:.2f}s ± {config_summary['performance_std']['time_std']:.2f}")
            else:
                config_summary = {
                    'configuration': config_setup['name'],
                    'description': config_setup['description'],
                    'error': 'All test cases failed',
                    'test_results': config_results
                }
            
            ablation_results.append(config_summary)
        
        # Comprehensive analysis of ablation results
        successful_configs = [r for r in ablation_results if 'error' not in r]
        
        if successful_configs:
            baseline = next((r for r in successful_configs if r['configuration'] == 'baseline_full'), None)
            
            analysis = {
                'component_importance': self._analyze_component_importance(successful_configs, baseline),
                'coordination_impact': self._analyze_coordination_impact(successful_configs),
                'efficiency_analysis': self._analyze_efficiency_impact(successful_configs),
                'robustness_analysis': self._analyze_robustness(successful_configs),
                'statistical_significance': self._calculate_statistical_significance(successful_configs, baseline)
            }
            
            print(f"\n🔍 Ablation Analysis Summary:")
            
            # Component importance
            if 'component_rankings' in analysis['component_importance']:
                rankings = analysis['component_importance']['component_rankings']
                print(f"   🏆 Component Importance (by quality impact):")
                for i, (component, impact) in enumerate(rankings[:3]):
                    print(f"      {i+1}. {component}: {impact:.3f} impact")
            
            # Best coordination mode
            if 'best_coordination_mode' in analysis['coordination_impact']:
                best_coord = analysis['coordination_impact']['best_coordination_mode']
                print(f"   🎯 Best Coordination Mode: {best_coord}")
            
            # Most efficient configuration
            if 'most_efficient_config' in analysis['efficiency_analysis']:
                most_efficient = analysis['efficiency_analysis']['most_efficient_config']
                print(f"   ⚡ Most Efficient: {most_efficient}")
                
        else:
            analysis = {'error': 'No successful ablation configurations'}
        
        # Save comprehensive ablation results
        ablation_path = self.results_dir / f'ablation_study_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(ablation_path, 'w') as f:
            json.dump({
                'test_cases': test_cases,
                'configurations': ablation_configs,
                'results': ablation_results,
                'analysis': analysis,
                'evaluation_timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        return {
            'results': ablation_results,
            'analysis': analysis
        }
    
    def _create_ablation_orchestrator(self, original_orchestrator, config_setup):
        """Create modified orchestrator for ablation study"""
        # Create a copy-like orchestrator for ablation
        # In a full implementation, this would properly clone the orchestrator
        modified_orchestrator = original_orchestrator
        
        # Disable specified agents by setting them to None
        for agent_name in config_setup['disabled_agents']:
            if agent_name in modified_orchestrator.agents:
                modified_orchestrator.agents[agent_name] = None
                print(f"      Disabled agent: {agent_name}")
        
        # Force coordination mode if specified
        if config_setup['coordination_mode'] in ['sequential', 'parallel']:
            modified_orchestrator._force_coordination_mode = config_setup['coordination_mode']
        
        return modified_orchestrator
    
    def _analyze_component_importance(self, results: List[Dict], baseline: Dict = None) -> Dict[str, Any]:
        """Analyze importance of different system components"""
        if not baseline:
            return {'error': 'No baseline configuration found'}
        
        baseline_quality = baseline['average_performance']['quality_score']
        baseline_sections = baseline['average_performance']['sections_completed']
        baseline_time = baseline['average_performance']['generation_time']
        
        component_impact = {}
        
        # Analyze impact of removing each component
        for result in results:
            config_name = result['configuration']
            
            if config_name.startswith('no_') and 'average_performance' in result:
                component_name = config_name.replace('no_', '').replace('_agent', '')
                
                # Calculate impact (positive means removing component hurt performance)
                quality_impact = baseline_quality - result['average_performance']['quality_score']
                sections_impact = baseline_sections - result['average_performance']['sections_completed']
                time_impact = result['average_performance']['generation_time'] - baseline_time  # Positive means slower
                
                component_impact[component_name] = {
                    'quality_impact': quality_impact,
                    'sections_impact': sections_impact,
                    'time_impact': time_impact,
                    'overall_impact': (quality_impact * 0.5 + sections_impact * 0.3 + max(0, time_impact * 0.2)),
                    'performance_ratio': result['average_performance']['quality_score'] / baseline_quality
                }
        
        # Rank components by importance
        component_rankings = sorted(component_impact.items(), 
                                  key=lambda x: x[1]['overall_impact'], reverse=True)
        
        return {
            'component_impacts': component_impact,
            'component_rankings': component_rankings,
            'most_important': component_rankings[0][0] if component_rankings else None,
            'least_important': component_rankings[-1][0] if component_rankings else None,
            'baseline_performance': {
                'quality': baseline_quality,
                'sections': baseline_sections,
                'time': baseline_time
            }
        }
    
    def _analyze_coordination_impact(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze impact of different coordination strategies"""
        coordination_results = {}
        
        for result in results:
            coord_mode = result.get('coordination_mode')
            if coord_mode and coord_mode in ['sequential', 'parallel', 'hybrid'] and 'average_performance' in result:
                coordination_results[coord_mode] = {
                    'quality': result['average_performance']['quality_score'],
                    'sections': result['average_performance']['sections_completed'],
                    'time': result['average_performance']['generation_time'],
                    'coordination': result['average_performance']['coordination_score'],
                    'efficiency': result['average_performance']['sections_completed'] / max(result['average_performance']['generation_time'], 0.1)
                }
        
        if len(coordination_results) >= 2:
            # Find best coordination mode for different metrics
            best_modes = {
                'quality': max(coordination_results.keys(), key=lambda k: coordination_results[k]['quality']),
                'speed': min(coordination_results.keys(), key=lambda k: coordination_results[k]['time']),
                'efficiency': max(coordination_results.keys(), key=lambda k: coordination_results[k]['efficiency']),
                'completeness': max(coordination_results.keys(), key=lambda k: coordination_results[k]['sections'])
            }
            
            return {
                'coordination_results': coordination_results,
                'best_modes': best_modes,
                'best_overall': max(coordination_results.keys(), 
                                  key=lambda k: coordination_results[k]['quality'] * 0.5 + 
                                               coordination_results[k]['efficiency'] * 0.3 + 
                                               coordination_results[k]['sections'] * 0.2),
                'coordination_rankings': {
                    'by_quality': sorted(coordination_results.keys(), 
                                       key=lambda k: coordination_results[k]['quality'], reverse=True),
                    'by_speed': sorted(coordination_results.keys(), 
                                     key=lambda k: coordination_results[k]['time']),
                    'by_completeness': sorted(coordination_results.keys(), 
                                            key=lambda k: coordination_results[k]['sections'], reverse=True)
                }
            }
        
        return {'error': 'Insufficient coordination mode data'}
    
    def _analyze_efficiency_impact(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze efficiency vs quality tradeoffs"""
        efficiency_analysis = {}
        
        for result in results:
            if 'average_performance' in result:
                perf = result['average_performance']
                
                # Calculate multiple efficiency metrics
                quality_time_efficiency = perf['quality_score'] / max(perf['generation_time'], 0.1)
                sections_time_efficiency = perf['sections_completed'] / max(perf['generation_time'], 0.1)
                overall_efficiency = (quality_time_efficiency * 0.6 + sections_time_efficiency * 0.4)
                
                efficiency_analysis[result['configuration']] = {
                    'quality_time_efficiency': quality_time_efficiency,
                    'sections_time_efficiency': sections_time_efficiency,
                    'overall_efficiency': overall_efficiency,
                    'quality': perf['quality_score'],
                    'sections': perf['sections_completed'],
                    'time': perf['generation_time'],
                    'quality_sections_product': perf['quality_score'] * perf['sections_completed']
                }
        
        if efficiency_analysis:
            # Find most efficient configurations
            most_efficient_overall = max(efficiency_analysis.keys(), 
                                       key=lambda k: efficiency_analysis[k]['overall_efficiency'])
            most_efficient_quality = max(efficiency_analysis.keys(), 
                                       key=lambda k: efficiency_analysis[k]['quality_time_efficiency'])
            
            # Calculate efficiency rankings
            efficiency_rankings = {
                'by_overall_efficiency': sorted(efficiency_analysis.keys(), 
                                              key=lambda k: efficiency_analysis[k]['overall_efficiency'], reverse=True),
                'by_quality_efficiency': sorted(efficiency_analysis.keys(), 
                                              key=lambda k: efficiency_analysis[k]['quality_time_efficiency'], reverse=True),
                'by_speed': sorted(efficiency_analysis.keys(), 
                                 key=lambda k: efficiency_analysis[k]['time'])
            }
            
            return {
                'efficiency_results': efficiency_analysis,
                'most_efficient_overall': most_efficient_overall,
                'most_efficient_quality': most_efficient_quality,
                'efficiency_rankings': efficiency_rankings,
                'efficiency_vs_quality_analysis': self._analyze_efficiency_quality_tradeoff(efficiency_analysis)
            }
        
        return {'error': 'No efficiency data available'}
    
    def _analyze_efficiency_quality_tradeoff(self, efficiency_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the tradeoff between efficiency and quality"""
        configs = list(efficiency_data.keys())
        qualities = [efficiency_data[c]['quality'] for c in configs]
        times = [efficiency_data[c]['time'] for c in configs]
        
        if len(configs) < 3:
            return {'error': 'Insufficient data for tradeoff analysis'}
        
        # Calculate correlation between quality and time
        quality_time_correlation = np.corrcoef(qualities, times)[0, 1] if len(qualities) > 1 else 0
        
        # Find Pareto efficient configurations (good quality, low time)
        pareto_efficient = []
        for i, config in enumerate(configs):
            is_pareto = True
            for j, other_config in enumerate(configs):
                if i != j:
                    # Check if other config dominates (better quality AND faster)
                    if (qualities[j] >= qualities[i] and times[j] <= times[i] and 
                        (qualities[j] > qualities[i] or times[j] < times[i])):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_efficient.append({
                    'configuration': config,
                    'quality': qualities[i],
                    'time': times[i],
                    'efficiency': efficiency_data[config]['overall_efficiency']
                })
        
        return {
            'quality_time_correlation': quality_time_correlation,
            'correlation_interpretation': 'negative' if quality_time_correlation < -0.3 else 'positive' if quality_time_correlation > 0.3 else 'weak',
            'pareto_efficient_configs': pareto_efficient,
            'best_tradeoff': max(pareto_efficient, key=lambda x: x['efficiency']) if pareto_efficient else None
        }
    
    def _analyze_robustness(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze system robustness across different configurations"""
        if len(results) < 3:
            return {'error': 'Insufficient configurations for robustness analysis'}
        
        # Extract performance metrics across all configurations
        all_qualities = []
        all_sections = []
        all_times = []
        
        for result in results:
            if 'average_performance' in result:
                all_qualities.append(result['average_performance']['quality_score'])
                all_sections.append(result['average_performance']['sections_completed'])
                all_times.append(result['average_performance']['generation_time'])
        
        if not all_qualities:
            return {'error': 'No performance data available'}
        
        # Calculate robustness metrics
        quality_robustness = {
            'mean': np.mean(all_qualities),
            'std': np.std(all_qualities),
            'min': np.min(all_qualities),
            'max': np.max(all_qualities),
            'coefficient_of_variation': np.std(all_qualities) / max(np.mean(all_qualities), 0.001),
            'robustness_score': 1.0 - (np.std(all_qualities) / max(np.mean(all_qualities), 0.001))  # Higher is more robust
        }
        
        sections_robustness = {
            'mean': np.mean(all_sections),
            'std': np.std(all_sections),
            'min': np.min(all_sections),
            'max': np.max(all_sections),
            'robustness_score': 1.0 - (np.std(all_sections) / max(np.mean(all_sections), 0.001))
        }
        
        # Identify most and least robust configurations
        config_robustness = {}
        for result in results:
            if 'average_performance' in result and 'performance_std' in result:
                config_name = result['configuration']
                # Robustness = performance / variability
                quality_robustness_score = result['average_performance']['quality_score'] / max(result['performance_std']['quality_std'], 0.001)
                config_robustness[config_name] = quality_robustness_score
        
        most_robust = max(config_robustness.keys(), key=lambda k: config_robustness[k]) if config_robustness else None
        least_robust = min(config_robustness.keys(), key=lambda k: config_robustness[k]) if config_robustness else None
        
        return {
            'quality_robustness': quality_robustness,
            'sections_robustness': sections_robustness,
            'configuration_robustness': config_robustness,
            'most_robust_config': most_robust,
            'least_robust_config': least_robust,
            'overall_robustness_grade': self._grade_robustness(quality_robustness['robustness_score'])
        }
    
    def _grade_robustness(self, robustness_score: float) -> str:
        """Convert robustness score to letter grade"""
        if robustness_score >= 0.9:
            return 'A+ (Excellent)'
        elif robustness_score >= 0.8:
            return 'A (Very Good)'
        elif robustness_score >= 0.7:
            return 'B+ (Good)'
        elif robustness_score >= 0.6:
            return 'B (Fair)'
        elif robustness_score >= 0.5:
            return 'C+ (Acceptable)'
        else:
            return 'C (Needs Improvement)'
    
    def _calculate_statistical_significance(self, results: List[Dict], baseline: Dict = None) -> Dict[str, Any]:
        """Calculate statistical significance of performance differences"""
        if not baseline or len(results) < 3:
            return {'error': 'Insufficient data for statistical analysis'}
        
        from scipy import stats
        
        # Get baseline performance across test cases
        baseline_qualities = [test['quality_score'] for test in baseline['test_results'] if 'error' not in test]
        
        significance_tests = {}
        
        for result in results:
            if result['configuration'] != 'baseline_full' and 'test_results' in result:
                config_qualities = [test['quality_score'] for test in result['test_results'] if 'error' not in test]
                
                if len(config_qualities) >= 2 and len(baseline_qualities) >= 2:
                    # Perform t-test
                    try:
                        t_stat, p_value = stats.ttest_ind(baseline_qualities, config_qualities)
                        effect_size = (np.mean(config_qualities) - np.mean(baseline_qualities)) / np.sqrt(
                            (np.var(baseline_qualities) + np.var(config_qualities)) / 2
                        )
                        
                        significance_tests[result['configuration']] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'effect_size': effect_size,
                            'significant': p_value < 0.05,
                            'interpretation': self._interpret_effect_size(effect_size)
                        }
                    except:
                        significance_tests[result['configuration']] = {'error': 'Statistical test failed'}
        
        return significance_tests
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_effect = abs(effect_size)
        direction = 'better' if effect_size > 0 else 'worse'
        
        if abs_effect < 0.2:
            magnitude = 'negligible'
        elif abs_effect < 0.5:
            magnitude = 'small'
        elif abs_effect < 0.8:
            magnitude = 'medium'
        else:
            magnitude = 'large'
        
        return f"{magnitude} effect ({direction} than baseline)"
    
    def create_evaluation_dashboard(self, training_metrics: Dict[str, Any], 
                                  benchmark_results: Dict[str, Any] = None,
                                  ablation_results: Dict[str, Any] = None) -> str:
        """Create comprehensive evaluation dashboard"""
        
        dashboard_path = self.results_dir / 'evaluation_dashboard.html'
        
        # Generate comprehensive HTML dashboard
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Agent RL Paper Generation - Comprehensive Evaluation Dashboard</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: linear-gradient(145deg, #f8f9fa, #e9ecef); border: none; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1); transition: transform 0.3s ease; }}
        .metric-card:hover {{ transform: translateY(-5px); }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #007bff; margin: 10px 0; }}
        .metric-label {{ font-size: 14px; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; }}
        .improvement {{ color: #28a745; font-weight: bold; }}
        .decline {{ color: #dc3545; font-weight: bold; }}
        .neutral {{ color: #6c757d; }}
        .section-header {{ background: linear-gradient(90deg, #007bff, #0056b3); color: white; padding: 15px 20px; border-radius: 8px; margin: 30px 0 15px 0; font-size: 18px; font-weight: bold; }}
        .subsection {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; margin: 15px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: white; }}
        th {{ background: linear-gradient(90deg, #343a40, #495057); color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #dee2e6; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        tr:hover {{ background-color: #e3f2fd; }}
        .progress-bar {{ width: 100%; height: 20px; background-color: #e9ecef; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: linear-gradient(90deg, #28a745, #20c997); transition: width 0.3s ease; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
        .badge-success {{ background-color: #d4edda; color: #155724; }}
        .badge-warning {{ background-color: #fff3cd; color: #856404; }}
        .badge-danger {{ background-color: #f8d7da; color: #721c24; }}
        .badge-info {{ background-color: #d1ecf1; color: #0c5460; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Multi-Agent Reinforcement Learning Evaluation Dashboard</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><em>Comprehensive evaluation of collaborative research paper generation system</em></p>
        </div>
"""
        
        # Add training performance overview
        if training_metrics:
            learning_analysis = self.evaluate_learning_progress(training_metrics)
            
            if 'final_performance' in learning_analysis:
                final_perf = learning_analysis['final_performance']
                improvements = learning_analysis.get('improvement_metrics', {})
                trends = learning_analysis.get('trends', {})
                
                html_content += f"""
        <div class="section-header">📊 Training Performance Overview</div>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Final Reward</div>
                <div class="metric-value">{final_perf.get('reward', 0):.3f}</div>
                <p>Change: <span class="{'improvement' if improvements.get('total_reward_improvement', 0) > 0 else 'decline' if improvements.get('total_reward_improvement', 0) < 0 else 'neutral'}">{improvements.get('total_reward_improvement', 0):+.3f}</span></p>
                <p>Trend: <span class="{'improvement' if trends.get('reward_trend', 0) > 0 else 'decline' if trends.get('reward_trend', 0) < 0 else 'neutral'}">{trends.get('reward_trend', 0):+.4f}/ep</span></p>
            </div>
            <div class="metric-card">
                <div class="metric-label">Paper Quality</div>
                <div class="metric-value">{final_perf.get('quality', 0):.3f}</div>
                <p>Change: <span class="{'improvement' if improvements.get('total_quality_improvement', 0) > 0 else 'decline' if improvements.get('total_quality_improvement', 0) < 0 else 'neutral'}">{improvements.get('total_quality_improvement', 0):+.3f}</span></p>
                <p>Trend: <span class="{'improvement' if trends.get('quality_trend', 0) > 0 else 'decline' if trends.get('quality_trend', 0) < 0 else 'neutral'}">{trends.get('quality_trend', 0):+.4f}/ep</span></p>
            </div>
            <div class="metric-card">
                <div class="metric-label">Agent Coordination</div>
                <div class="metric-value">{final_perf.get('coordination', 0):.3f}</div>
                <p>Trend: <span class="{'improvement' if trends.get('coordination_trend', 0) > 0 else 'decline' if trends.get('coordination_trend', 0) < 0 else 'neutral'}">{trends.get('coordination_trend', 0):+.4f}/ep</span></p>
                <p>Stability: {learning_analysis.get('stability', {}).get('coordination_stability', 0):.3f}</p>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sections Completed</div>
                <div class="metric-value">{final_perf.get('sections', 0):.1f}/6</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {(final_perf.get('sections', 0)/6)*100}%"></div>
                </div>
                <p>Completion Rate: {(final_perf.get('sections', 0)/6)*100:.1f}%</p>
            </div>
        </div>
        
        <div class="section-header">📈 Detailed Learning Analysis</div>
        <div class="subsection">
            <h3>Learning Characteristics</h3>
            <table>
                <tr><th>Metric</th><th>Trend (per episode)</th><th>Learning Efficiency</th><th>Stability Score</th><th>Status</th></tr>
                <tr>
                    <td><strong>Reward Learning</strong></td>
                    <td class="{'improvement' if trends.get('reward_trend', 0) > 0 else 'decline' if trends.get('reward_trend', 0) < 0 else 'neutral'}">{trends.get('reward_trend', 0):+.4f}</td>
                    <td>{learning_analysis.get('efficiency', {}).get('reward_efficiency', 0):.3f}</td>
                    <td>{learning_analysis.get('stability', {}).get('reward_stability', 0):.3f}</td>
                    <td>
                        {"<span class='badge badge-success'>Improving</span>" if trends.get('reward_trend', 0) > 0.01 else 
                         "<span class='badge badge-warning'>Plateauing</span>" if abs(trends.get('reward_trend', 0)) <= 0.01 else 
                         "<span class='badge badge-danger'>Declining</span>"}
                    </td>
                </tr>
                <tr>
                    <td><strong>Quality Learning</strong></td>
                    <td class="{'improvement' if trends.get('quality_trend', 0) > 0 else 'decline' if trends.get('quality_trend', 0) < 0 else 'neutral'}">{trends.get('quality_trend', 0):+.4f}</td>
                    <td>{learning_analysis.get('efficiency', {}).get('quality_efficiency', 0):.3f}</td>
                    <td>{learning_analysis.get('stability', {}).get('quality_stability', 0):.3f}</td>
                    <td>
                        {"<span class='badge badge-success'>Improving</span>" if trends.get('quality_trend', 0) > 0.005 else 
                         "<span class='badge badge-warning'>Plateauing</span>" if abs(trends.get('quality_trend', 0)) <= 0.005 else 
                         "<span class='badge badge-danger'>Declining</span>"}
                    </td>
                </tr>
                <tr>
                    <td><strong>Sections Learning</strong></td>
                    <td class="{'improvement' if trends.get('sections_trend', 0) > 0 else 'decline' if trends.get('sections_trend', 0) < 0 else 'neutral'}">{trends.get('sections_trend', 0):+.4f}</td>
                    <td>N/A</td>
                    <td>N/A</td>
                    <td>
                        {"<span class='badge badge-success'>Improving</span>" if trends.get('sections_trend', 0) > 0.01 else 
                         "<span class='badge badge-warning'>Plateauing</span>" if abs(trends.get('sections_trend', 0)) <= 0.01 else 
                         "<span class='badge badge-danger'>Declining</span>"}
                    </td>
                </tr>
            </table>
        </div>
"""
        
        # Add convergence analysis
        convergence = learning_analysis.get('convergence', {})
        if convergence:
            html_content += f"""
        <div class="subsection">
            <h3>Convergence Analysis</h3>
            <table>
                <tr><th>Metric</th><th>Converged</th><th>Plateau Length</th><th>Recent Improvement</th></tr>
                <tr>
                    <td>Reward</td>
                    <td>{"<span class='badge badge-info'>Yes</span>" if convergence.get('reward_convergence', {}).get('converged', False) else "<span class='badge badge-warning'>No</span>"}</td>
                    <td>{convergence.get('reward_convergence', {}).get('plateau_length', 0)} episodes</td>
                    <td>{convergence.get('reward_convergence', {}).get('recent_improvement', 0):+.4f}</td>
                </tr>
                <tr>
                    <td>Quality</td>
                    <td>{"<span class='badge badge-info'>Yes</span>" if convergence.get('quality_convergence', {}).get('converged', False) else "<span class='badge badge-warning'>No</span>"}</td>
                    <td>{convergence.get('quality_convergence', {}).get('plateau_length', 0)} episodes</td>
                    <td>{convergence.get('quality_convergence', {}).get('recent_improvement', 0):+.4f}</td>
                </tr>
            </table>
        </div>
"""
        
        # Add benchmark results if available
        if benchmark_results and 'summary' in benchmark_results:
            summary = benchmark_results['summary']
            if 'error' not in summary:
                perf_metrics = summary.get('performance_metrics', {})
                expectations = summary.get('expectation_analysis', {})
                
                html_content += f"""
        <div class="section-header">🎯 Benchmark Evaluation Results</div>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value">{summary['success_rate']:.1%}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {summary['success_rate']*100}%"></div>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Average Quality</div>
                <div class="metric-value">{perf_metrics.get('average_quality', 0):.3f}</div>
                <p>{"<span class='badge badge-success'>Excellent</span>" if perf_metrics.get('average_quality', 0) > 0.8 else 
                   "<span class='badge badge-info'>Good</span>" if perf_metrics.get('average_quality', 0) > 0.7 else 
                   "<span class='badge badge-warning'>Fair</span>" if perf_metrics.get('average_quality', 0) > 0.6 else 
                   "<span class='badge badge-danger'>Poor</span>"}</p>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Sections</div>
                <div class="metric-value">{perf_metrics.get('average_sections', 0):.1f}/6</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {(perf_metrics.get('average_sections', 0)/6)*100}%"></div>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Generation Time</div>
                <div class="metric-value">{perf_metrics.get('average_time', 0):.1f}s</div>
                <p>{"<span class='badge badge-success'>Fast</span>" if perf_metrics.get('average_time', 0) < 3 else 
                   "<span class='badge badge-info'>Moderate</span>" if perf_metrics.get('average_time', 0) < 6 else 
                   "<span class='badge badge-warning'>Slow</span>"}</p>
            </div>
        </div>
        
        <div class="subsection">
            <h3>Expectation Analysis</h3>
            <table>
                <tr><th>Expectation Type</th><th>Cases Met</th><th>Total Cases</th><th>Success Rate</th><th>Status</th></tr>
                <tr>
                    <td>Quality Expectations</td>
                    <td>{expectations.get('quality_expectations_met', 0)}</td>
                    <td>{summary['successful_cases']}</td>
                    <td>{(expectations.get('quality_expectations_met', 0) / max(summary['successful_cases'], 1)):.1%}</td>
                    <td>{"<span class='badge badge-success'>Excellent</span>" if (expectations.get('quality_expectations_met', 0) / max(summary['successful_cases'], 1)) > 0.8 else "<span class='badge badge-warning'>Needs Improvement</span>"}</td>
                </tr>
                <tr>
                    <td>Section Expectations</td>
                    <td>{expectations.get('section_expectations_met', 0)}</td>
                    <td>{summary['successful_cases']}</td>
                    <td>{(expectations.get('section_expectations_met', 0) / max(summary['successful_cases'], 1)):.1%}</td>
                    <td>{"<span class='badge badge-success'>Excellent</span>" if (expectations.get('section_expectations_met', 0) / max(summary['successful_cases'], 1)) > 0.8 else "<span class='badge badge-warning'>Needs Improvement</span>"}</td>
                </tr>
                <tr>
                    <td>Overall Expectations</td>
                    <td>{expectations.get('overall_expectations_met', 0)}</td>
                    <td>{summary['successful_cases']}</td>
                    <td>{(expectations.get('overall_expectations_met', 0) / max(summary['successful_cases'], 1)):.1%}</td>
                    <td>{"<span class='badge badge-success'>Excellent</span>" if (expectations.get('overall_expectations_met', 0) / max(summary['successful_cases'], 1)) > 0.8 else "<span class='badge badge-warning'>Needs Improvement</span>"}</td>
                </tr>
            </table>
        </div>
"""
                
                # Add difficulty analysis if available
                if 'difficulty_analysis' in summary:
                    diff_analysis = summary['difficulty_analysis']
                    html_content += f"""
        <div class="subsection">
            <h3>Performance by Difficulty Level</h3>
            <table>
                <tr><th>Difficulty</th><th>Cases</th><th>Avg Quality</th><th>Avg Sections</th><th>Success Rate</th></tr>
"""
                    for difficulty, data in diff_analysis.items():
                        html_content += f"""
                <tr>
                    <td><strong>{difficulty.title()}</strong></td>
                    <td>{data['case_count']}</td>
                    <td>{data['average_quality']:.3f}</td>
                    <td>{data['average_sections']:.1f}</td>
                    <td>{data['success_rate']:.1%}</td>
                </tr>
"""
                    html_content += "</table></div>"
        
        # Add ablation study results if available
        if ablation_results and 'analysis' in ablation_results:
            analysis = ablation_results['analysis']
            
            html_content += f"""
        <div class="section-header">🔬 Ablation Study Results</div>
"""
            
            # Component importance analysis
            if 'component_importance' in analysis:
                comp_imp = analysis['component_importance']
                if 'component_impacts' in comp_imp:
                    html_content += f"""
        <div class="subsection">
            <h3>Component Importance Analysis</h3>
            <p><strong>Most Important Component:</strong> {comp_imp.get('most_important', 'Unknown').title()}</p>
            <p><strong>Least Important Component:</strong> {comp_imp.get('least_important', 'Unknown').title()}</p>
            <table>
                <tr><th>Component</th><th>Quality Impact</th><th>Sections Impact</th><th>Importance Rank</th><th>Performance Ratio</th></tr>
"""
                    
                    for component, data in comp_imp.get('component_impacts', {}).items():
                        html_content += f"""
                <tr>
                    <td><strong>{component.title()}</strong></td>
                    <td>{data['quality_impact']:.3f}</td>
                    <td>{data['sections_impact']:.3f}</td>
                    <td>#{data['importance_rank']}</td>
                    <td>{data['performance_ratio']:.3f}</td>
                </tr>
"""
                    html_content += "</table></div>"
            
            # Coordination strategy analysis
            if 'coordination_impact' in analysis:
                coord_impact = analysis['coordination_impact']
                if 'coordination_results' in coord_impact:
                    html_content += f"""
        <div class="subsection">
            <h3>Coordination Strategy Analysis</h3>
            <p><strong>Best Overall Mode:</strong> {coord_impact.get('best_overall', 'Unknown').title()}</p>
            <table>
                <tr><th>Coordination Mode</th><th>Quality Score</th><th>Generation Time</th><th>Sections Completed</th><th>Efficiency</th></tr>
"""
                    
                    for mode, data in coord_impact.get('coordination_results', {}).items():
                        efficiency = data['sections'] / max(data['time'], 0.1)
                        html_content += f"""
                <tr>
                    <td><strong>{mode.title()}</strong></td>
                    <td>{data['quality']:.3f}</td>
                    <td>{data['time']:.2f}s</td>
                    <td>{data['sections']:.1f}</td>
                    <td>{efficiency:.2f}</td>
                </tr>
"""
                    html_content += "</table></div>"
            
            # Efficiency analysis
            if 'efficiency_analysis' in analysis:
                eff_analysis = analysis['efficiency_analysis']
                if 'most_efficient_overall' in eff_analysis:
                    html_content += f"""
        <div class="subsection">
            <h3>Efficiency Analysis</h3>
            <p><strong>Most Efficient Configuration:</strong> {eff_analysis['most_efficient_overall']}</p>
            <p><strong>Best Quality-Time Balance:</strong> {eff_analysis.get('most_efficient_quality', 'Unknown')}</p>
        </div>
"""
        
        # Add recommendations section
        html_content += f"""
        <div class="section-header">💡 Comprehensive Recommendations</div>
        <div class="subsection">
            <h3>Training Recommendations</h3>
            <ul>
"""
        
        # Generate dynamic recommendations based on analysis
        if training_metrics:
            final_quality = learning_analysis.get('final_performance', {}).get('quality', 0)
            quality_trend = learning_analysis.get('trends', {}).get('quality_trend', 0)
            
            if final_quality < 0.7:
                html_content += "<li><strong>Quality Improvement:</strong> Current quality is below target. Consider extending training or adjusting reward functions.</li>"
            
            if quality_trend < 0.001:
                html_content += "<li><strong>Learning Plateau:</strong> Quality improvements have plateaued. Consider curriculum adjustments or hyperparameter tuning.</li>"
            else:
                html_content += "<li><strong>Positive Learning:</strong> Quality is still improving. Continue training for better results.</li>"
            
            coordination_score = learning_analysis.get('final_performance', {}).get('coordination', 0)
            if coordination_score < 0.8:
                html_content += "<li><strong>Coordination:</strong> Agent coordination could be improved. Consider adjusting communication mechanisms.</li>"
            
            sections_completed = learning_analysis.get('final_performance', {}).get('sections', 0)
            if sections_completed < 5:
                html_content += "<li><strong>Section Completion:</strong> Focus on improving section completion rates through better task allocation.</li>"
        
        html_content += """
            </ul>
        </div>
        
        <div class="subsection">
            <h3>Architecture Recommendations</h3>
            <ul>
"""
        
        if ablation_results and 'analysis' in ablation_results:
            comp_importance = ablation_results['analysis'].get('component_importance', {})
            if 'most_important' in comp_importance:
                most_important = comp_importance['most_important']
                html_content += f"<li><strong>Critical Component:</strong> The {most_important} agent shows highest impact. Ensure robust implementation.</li>"
            
            coord_impact = ablation_results['analysis'].get('coordination_impact', {})
            if 'best_overall' in coord_impact:
                best_coord = coord_impact['best_overall']
                html_content += f"<li><strong>Optimal Coordination:</strong> {best_coord.title()} coordination mode shows best overall performance.</li>"
        
        html_content += """
                <li><strong>Scalability:</strong> Consider computational requirements for larger-scale deployment.</li>
                <li><strong>Robustness:</strong> Implement error handling and fallback mechanisms for production use.</li>
            </ul>
        </div>
        
        <div class="subsection">
            <h3>Future Development</h3>
            <ul>
                <li><strong>Advanced Evaluation:</strong> Implement human evaluation for generated papers</li>
                <li><strong>Domain Extension:</strong> Test performance on papers from different research domains</li>
                <li><strong>Real-world Integration:</strong> Develop API for integration with existing research workflows</li>
                <li><strong>Continuous Learning:</strong> Implement online learning for continuous improvement</li>
            </ul>
        </div>
        
        <div class="section-header">ℹ️ System Information</div>
        <div class="subsection">
            <table>
                <tr><th>Component</th><th>Details</th></tr>
                <tr><td><strong>Framework</strong></td><td>PyTorch Multi-Agent Reinforcement Learning</td></tr>
                <tr><td><strong>Algorithms</strong></td><td>DQN (Literature, Methodology) + PPO (Writing, Orchestrator)</td></tr>
                <tr><td><strong>Data Sources</strong></td><td>arXiv API, Semantic Scholar API</td></tr>
                <tr><td><strong>Evaluation Metrics</strong></td><td>Quality, Coherence, Citations, Efficiency, Novelty</td></tr>
                <tr><td><strong>Agents</strong></td><td>Literature Review, Methodology Design, Scientific Writing, Analysis</td></tr>
                <tr><td><strong>Coordination Modes</strong></td><td>Sequential, Parallel, Hybrid</td></tr>
            </table>
        </div>
        
        <div style="text-align: center; margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
            <p><em>Dashboard generated by Comprehensive Multi-Agent Evaluation System</em></p>
            <p><strong>Evaluation completed:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        print(f"📊 Comprehensive evaluation dashboard created: {dashboard_path}")
        return str(dashboard_path)
    
    def generate_before_after_comparison(self, untrained_orchestrator, trained_orchestrator) -> Dict[str, Any]:
        """Generate comprehensive before/after training comparison"""
        
        comparison_test_case = {
            'title': 'Reinforcement Learning for Multi-Agent Coordination',
            'research_question': 'How can reinforcement learning improve multi-agent coordination?',
            'paper_type': 'research',
            'venue': 'Conference'
        }
        
        print("📊 Generating Before/After Training Comparison")
        print("=" * 50)
        
        # Generate paper with untrained system
        print("🔄 Generating paper with UNTRAINED agents...")
        
        # Disable learning for untrained version
        for agent_name, agent in untrained_orchestrator.agents.items():
            if agent is not None and hasattr(agent, 'epsilon'):
                agent.epsilon = 1.0  # Maximum exploration (random behavior)
        
        before_start = time.time()
        before_result = untrained_orchestrator.orchestrate_paper_generation(comparison_test_case)
        before_time = time.time() - before_start
        
        # Generate paper with trained system
        print("🎯 Generating paper with TRAINED agents...")
        
        # Set to evaluation mode for trained version
        for agent_name, agent in trained_orchestrator.agents.items():
            if agent is not None and hasattr(agent, 'epsilon'):
                agent.epsilon = 0.0  # No exploration (use learned policy)
        
        after_start = time.time()
        after_result = trained_orchestrator.orchestrate_paper_generation(comparison_test_case)
        after_time = time.time() - after_start
        
        # Comprehensive evaluation of both papers
        before_quality = self.evaluate_paper_quality(before_result['paper'])
        after_quality = self.evaluate_paper_quality(after_result['paper'])
        
        # Calculate improvements
        comparison = {
            'test_case': comparison_test_case,
            'before_training': {
                'generation_time': before_time,
                'total_reward': before_result['total_reward'],
                'sections_completed': before_result['metrics']['sections_completed'],
                'coordination_score': before_result['metrics']['agent_coordination_score'],
                'quality_evaluation': before_quality,
                'paper_length': sum(len(str(content)) for content in before_result['paper'].get('sections', {}).values()),
                'paper_content': before_result['paper']
            },
            'after_training': {
                'generation_time': after_time,
                'total_reward': after_result['total_reward'],
                'sections_completed': after_result['metrics']['sections_completed'],
                'coordination_score': after_result['metrics']['agent_coordination_score'],
                'quality_evaluation': after_quality,
                'paper_length': sum(len(str(content)) for content in after_result['paper'].get('sections', {}).values()),
                'paper_content': after_result['paper']
            },
            'improvements': {
                'reward_improvement': after_result['total_reward'] - before_result['total_reward'],
                'quality_improvement': after_quality['overall_score'] - before_quality['overall_score'],
                'sections_improvement': after_result['metrics']['sections_completed'] - before_result['metrics']['sections_completed'],
                'coordination_improvement': after_result['metrics']['agent_coordination_score'] - before_result['metrics']['agent_coordination_score'],
                'time_improvement': before_time - after_time,  # Positive means faster
                'length_improvement': sum(len(str(content)) for content in after_result['paper'].get('sections', {}).values()) - sum(len(str(content)) for content in before_result['paper'].get('sections', {}).values())
            },
            'percentage_improvements': {},
            'aspect_improvements': {}
        }
        
        # Calculate percentage improvements
        for metric, improvement in comparison['improvements'].items():
            if metric.endswith('_improvement'):
                base_metric = metric.replace('_improvement', '')
                before_value = comparison['before_training'].get(base_metric.replace('reward', 'total_reward'), 0)
                if base_metric == 'quality':
                    before_value = before_quality['overall_score']
                elif base_metric == 'sections':
                    before_value = before_result['metrics']['sections_completed']
                elif base_metric == 'coordination':
                    before_value = before_result['metrics']['agent_coordination_score']
                elif base_metric == 'time':
                    before_value = before_time
                elif base_metric == 'length':
                    before_value = sum(len(str(content)) for content in before_result['paper'].get('sections', {}).values())
                
                if before_value != 0:
                    percentage = (improvement / abs(before_value)) * 100
                    comparison['percentage_improvements'][base_metric] = percentage
        
        # Detailed aspect improvements
        for aspect in ['structure', 'coherence', 'citations', 'writing', 'novelty']:
            before_score = before_quality['aspect_scores'].get(aspect, 0)
            after_score = after_quality['aspect_scores'].get(aspect, 0)
            improvement = after_score - before_score
            percentage = (improvement / max(before_score, 0.001)) * 100
            
            comparison['aspect_improvements'][aspect] = {
                'before': before_score,
                'after': after_score,
                'improvement': improvement,
                'percentage': percentage
            }
        
        # Print comprehensive comparison
        print(f"\n📈 Comprehensive Training Impact Analysis:")
        print(f"{'Metric':<20} {'Before':<10} {'After':<10} {'Change':<12} {'% Change':<10}")
        print("-" * 65)
        
        metrics_to_show = [
            ('Quality', before_quality['overall_score'], after_quality['overall_score']),
            ('Sections', before_result['metrics']['sections_completed'], after_result['metrics']['sections_completed']),
            ('Coordination', before_result['metrics']['agent_coordination_score'], after_result['metrics']['agent_coordination_score']),
            ('Time (s)', before_time, after_time),
            ('Length (chars)', comparison['before_training']['paper_length'], comparison['after_training']['paper_length'])
        ]
        
        for metric_name, before_val, after_val in metrics_to_show:
            change = after_val - before_val
            pct_change = (change / max(abs(before_val), 0.001)) * 100
            
            print(f"{metric_name:<20} {before_val:<10.3f} {after_val:<10.3f} {change:<+12.3f} {pct_change:<+10.1f}%")
        
        print(f"\n🎨 Quality Aspect Improvements:")
        for aspect, data in comparison['aspect_improvements'].items():
            print(f"  {aspect.title():<15}: {data['before']:.3f} → {data['after']:.3f} ({data['percentage']:+.1f}%)")
        
        # Save comparison results
        comparison_path = self.results_dir / f'before_after_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        return comparison
    
    def create_comprehensive_evaluation_report(self, training_metrics: Dict[str, Any], 
                                             benchmark_results: Dict[str, Any] = None,
                                             ablation_results: Dict[str, Any] = None,
                                             comparison_results: Dict[str, Any] = None) -> str:
        """Create comprehensive evaluation report"""
        
        report_path = self.results_dir / 'comprehensive_evaluation_report.md'
        
        # Generate comprehensive markdown report
        report_content = f"""# Multi-Agent Reinforcement Learning Paper Generation System
## Comprehensive Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**System:** Collaborative Research Paper Generator  
**Framework:** PyTorch Multi-Agent Reinforcement Learning  

---

## Executive Summary

This report presents a comprehensive evaluation of a multi-agent reinforcement learning system designed for collaborative research paper generation. The system employs specialized agents for literature review, methodology design, and scientific writing, coordinated through a learned orchestration policy.

### Key Findings
"""
        
        # Add key findings based on available data
        if training_metrics:
            learning_analysis = self.evaluate_learning_progress(training_metrics)
            final_quality = learning_analysis.get('final_performance', {}).get('quality', 0)
            total_episodes = len(training_metrics.get('episode_rewards', []))
            
            report_content += f"""
- **Training Effectiveness:** System trained for {total_episodes} episodes with final quality score of {final_quality:.3f}
- **Learning Progress:** {"Positive trends observed" if learning_analysis.get('trends', {}).get('quality_trend', 0) > 0 else "Learning plateau reached"}
- **Agent Coordination:** {"Effective coordination achieved" if learning_analysis.get('final_performance', {}).get('coordination', 0) > 0.8 else "Coordination needs improvement"}
"""
        
        if benchmark_results and 'summary' in benchmark_results:
            summary = benchmark_results['summary']
            if 'error' not in summary:
                report_content += f"""
- **Benchmark Performance:** {summary['success_rate']:.1%} success rate across difficulty levels
- **Quality Achievement:** Average quality of {summary.get('performance_metrics', {}).get('average_quality', 0):.3f} across test cases
- **Expectation Meeting:** {summary.get('expectation_analysis', {}).get('overall_expectations_met', 0)}/{summary['successful_cases']} cases met overall expectations
"""
        
        if ablation_results and 'analysis' in ablation_results:
            comp_imp = ablation_results['analysis'].get('component_importance', {})
            if 'most_important' in comp_imp:
                report_content += f"""
- **Component Analysis:** {comp_imp['most_important'].title()} agent identified as most critical component
- **Coordination Strategy:** {"Hybrid coordination optimal" if ablation_results['analysis'].get('coordination_impact', {}).get('best_overall') == 'hybrid' else "Alternative coordination modes may be better"}
"""
        
        # Detailed sections
        report_content += f"""

---

## Training Analysis

### Learning Progress Evaluation
"""
        
        if training_metrics:
            learning_analysis = self.evaluate_learning_progress(training_metrics)
            trends = learning_analysis.get('trends', {})
            stability = learning_analysis.get('stability', {})
            final_perf = learning_analysis.get('final_performance', {})
            
            report_content += f"""
| Metric | Final Value | Trend (per episode) | Stability Score | Status |
|--------|-------------|--------------------| --------------- |--------|
| Reward | {final_perf.get('reward', 0):.4f} | {trends.get('reward_trend', 0):+.4f} | {stability.get('reward_stability', 0):.3f} | {"🟢 Improving" if trends.get('reward_trend', 0) > 0.01 else "🟡 Stable" if abs(trends.get('reward_trend', 0)) <= 0.01 else "🔴 Declining"} |
| Quality | {final_perf.get('quality', 0):.4f} | {trends.get('quality_trend', 0):+.4f} | {stability.get('quality_stability', 0):.3f} | {"🟢 Improving" if trends.get('quality_trend', 0) > 0.005 else "🟡 Stable" if abs(trends.get('quality_trend', 0)) <= 0.005 else "🔴 Declining"} |
| Coordination | {final_perf.get('coordination', 0):.4f} | {trends.get('coordination_trend', 0):+.4f} | {stability.get('coordination_stability', 0):.3f} | {"🟢 Improving" if trends.get('coordination_trend', 0) > 0.01 else "🟡 Stable" if abs(trends.get('coordination_trend', 0)) <= 0.01 else "🔴 Declining"} |
| Sections | {final_perf.get('sections', 0):.2f}/6 | {trends.get('sections_trend', 0):+.4f} | N/A | {"🟢 Improving" if trends.get('sections_trend', 0) > 0.01 else "🟡 Stable" if abs(trends.get('sections_trend', 0)) <= 0.01 else "🔴 Declining"} |

### Learning Characteristics
- **Learning Speed:** {learning_analysis.get('learning_characteristics', {}).get('learning_speed', 'Unknown')}
- **Consistency:** {learning_analysis.get('learning_characteristics', {}).get('consistency', 0):.3f}
- **Exploration Effectiveness:** {learning_analysis.get('learning_characteristics', {}).get('exploration_effectiveness', {}).get('effectiveness', 'Unknown')}
"""
        
        # Add benchmark evaluation section
        if benchmark_results:
            report_content += f"""

---

## Benchmark Evaluation

### Test Case Performance
"""
            
            if 'detailed_results' in benchmark_results:
                for result in benchmark_results['detailed_results']:
                    if 'error' not in result:
                        report_content += f"""
#### {result['title']}
- **Difficulty:** {result['difficulty'].title()}
- **Type:** {result['paper_type']} | **Venue:** {result['venue']}
- **Quality Score:** {result['quality_score']:.3f} ({result.get('quality_grade', 'N/A')})
- **Sections Completed:** {result['sections_completed']:.1f}/{result['expected_sections']}
- **Generation Time:** {result['generation_time']:.2f}s
- **Meets Expectations:** {"✅ Yes" if result.get('meets_expectations', {}).get('overall', False) else "❌ No"}

**Quality Breakdown:**
"""
                        for aspect, score in result.get('quality_details', {}).items():
                            report_content += f"- {aspect.title()}: {score:.3f}\n"
                        
                        if result.get('strengths'):
                            report_content += f"\n**Strengths:** {', '.join(result['strengths'])}\n"
                        if result.get('weaknesses'):
                            report_content += f"**Areas for Improvement:** {', '.join(result['weaknesses'])}\n"
        
        # Add ablation study section
        if ablation_results:
            report_content += f"""

---

## Ablation Study Analysis

### Component Importance
"""
            
            if 'analysis' in ablation_results:
                analysis = ablation_results['analysis']
                
                if 'component_importance' in analysis:
                    comp_imp = analysis['component_importance']
                    
                    report_content += f"""
**Most Critical Component:** {comp_imp.get('most_important', 'Unknown').title()}  
**Least Critical Component:** {comp_imp.get('least_important', 'Unknown').title()}

| Component | Quality Impact | Sections Impact | Overall Impact | Rank |
|-----------|----------------|-----------------|----------------|------|
"""
                    
                    for component, data in comp_imp.get('component_impacts', {}).items():
                        report_content += f"| {component.title()} | {data['quality_impact']:.3f} | {data['sections_impact']:.3f} | {data['overall_impact']:.3f} | #{data['importance_rank']} |\n"
                
                # Coordination strategy analysis
                if 'coordination_impact' in analysis:
                    coord_impact = analysis['coordination_impact']
                    report_content += f"""

### Coordination Strategy Analysis

**Best Overall Mode:** {coord_impact.get('best_overall', 'Unknown').title()}

| Mode | Quality | Time | Sections | Efficiency |
|------|---------|------|----------|------------|
"""
                    
                    for mode, data in coord_impact.get('coordination_results', {}).items():
                        efficiency = data['sections'] / max(data['time'], 0.1)
                        report_content += f"| {mode.title()} | {data['quality']:.3f} | {data['time']:.2f}s | {data['sections']:.1f} | {efficiency:.2f} |\n"
        
        # Add before/after comparison section
        if comparison_results:
            report_content += f"""

---

## Before/After Training Comparison

### Performance Improvements
"""
            
            improvements = comparison_results.get('improvements', {})
            pct_improvements = comparison_results.get('percentage_improvements', {})
            
            report_content += f"""
| Metric | Before Training | After Training | Absolute Change | Percentage Change |
|--------|----------------|----------------|----------------|-------------------|
| Quality Score | {comparison_results['before_training']['quality_evaluation']['overall_score']:.3f} | {comparison_results['after_training']['quality_evaluation']['overall_score']:.3f} | {improvements.get('quality_improvement', 0):+.3f} | {pct_improvements.get('quality', 0):+.1f}% |
| Sections Completed | {comparison_results['before_training']['sections_completed']:.1f} | {comparison_results['after_training']['sections_completed']:.1f} | {improvements.get('sections_improvement', 0):+.1f} | {pct_improvements.get('sections', 0):+.1f}% |
| Coordination Score | {comparison_results['before_training']['coordination_score']:.3f} | {comparison_results['after_training']['coordination_score']:.3f} | {improvements.get('coordination_improvement', 0):+.3f} | {pct_improvements.get('coordination', 0):+.1f}% |
| Generation Time | {comparison_results['before_training']['generation_time']:.2f}s | {comparison_results['after_training']['generation_time']:.2f}s | {improvements.get('time_improvement', 0):+.2f}s | {pct_improvements.get('time', 0):+.1f}% |

### Quality Aspect Improvements
"""
            
            for aspect, data in comparison_results.get('aspect_improvements', {}).items():
                report_content += f"- **{aspect.title()}:** {data['before']:.3f} → {data['after']:.3f} ({data['percentage']:+.1f}%)\n"
        
        # Conclusions and recommendations
        report_content += f"""

---

## Conclusions and Recommendations

### System Strengths
"""
        
        # Generate conclusions based on available data
        strengths = []
        recommendations = []
        
        if training_metrics:
            learning_analysis = self.evaluate_learning_progress(training_metrics)
            if learning_analysis.get('trends', {}).get('reward_trend', 0) > 0:
                strengths.append("Demonstrates positive learning trends with consistent improvement")
            if learning_analysis.get('final_performance', {}).get('coordination', 0) > 0.8:
                strengths.append("Achieves effective multi-agent coordination")
            
            if learning_analysis.get('final_performance', {}).get('quality', 0) < 0.7:
                recommendations.append("Extend training duration or adjust reward mechanisms to improve quality")
            if learning_analysis.get('final_performance', {}).get('sections', 0) < 5:
                recommendations.append("Focus on improving section completion rates through enhanced task allocation")
        
        if ablation_results:
            strengths.append("Modular architecture allows for component analysis and optimization")
            recommendations.append("Leverage ablation study insights to optimize system configuration")
        
        for strength in strengths:
            report_content += f"- {strength}\n"
        
        report_content += f"""

### Recommendations for Improvement
"""
        
        for recommendation in recommendations:
            report_content += f"- {recommendation}\n"
        
        report_content += f"""
- Implement more sophisticated content generation algorithms
- Add human evaluation components for validation
- Extend to additional research domains beyond computer science
- Develop production deployment capabilities with API integration

### Future Research Directions
- **Advanced Learning:** Investigate meta-learning and transfer learning capabilities
- **Human-AI Collaboration:** Develop interfaces for human-in-the-loop paper generation
- **Domain Adaptation:** Extend system to handle multiple research disciplines
- **Quality Enhancement:** Implement more sophisticated natural language generation
- **Real-world Deployment:** Develop production-ready system with robust error handling

---

## Technical Specifications

### System Architecture
- **Agents:** Literature Review (DQN), Methodology Design (DQN), Scientific Writing (PPO), Analysis (DQN)
- **Orchestrator:** PPO-based coordination with 32-dimensional continuous action space
- **Communication:** Shared memory system with message passing protocols
- **Learning:** Experience replay for DQN agents, advantage estimation for PPO agents

### Evaluation Framework
- **Quality Metrics:** Structure, coherence, citations, writing style, novelty
- **Performance Metrics:** Generation time, coordination efficiency, section completion
- **Robustness Testing:** Multiple difficulty levels and paper types
- **Statistical Analysis:** Trend analysis, significance testing, effect size calculation

### Data Sources
- **Literature:** arXiv API for recent publications
- **Citations:** Semantic Scholar API for citation networks
- **Evaluation:** Custom quality assessment framework

---

*Report generated by the Comprehensive Multi-Agent Evaluation System*
"""
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"📋 Comprehensive evaluation report saved: {report_path}")
        return str(report_path)

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Comprehensive Evaluation of Multi-Agent System')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing training results')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Configuration file')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark evaluation')
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation studies')
    parser.add_argument('--comparison', action='store_true',
                       help='Run before/after comparison')
    parser.add_argument('--full', action='store_true',
                       help='Run all evaluations')
    
    args = parser.parse_args()
    
    print("🚀 Comprehensive Multi-Agent System Evaluation")
    print("=" * 50)
    
    # Load configuration and training results
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        return
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    results_dir = Path(args.results_dir)
    metrics_file = results_dir / 'comprehensive_metrics.json'
    
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            training_metrics = json.load(f)
        print(f"✅ Loaded training metrics from {metrics_file}")
    else:
        print(f"❌ Training metrics not found at {metrics_file}")
        training_metrics = {}
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(results_dir / 'evaluation')
    
    # Run requested evaluations
    benchmark_results = None
    ablation_results = None
    comparison_results = None
    
    if args.benchmark or args.full:
        print("\n🧪 Running benchmark evaluation...")
        # Note: This would require loading trained orchestrator
        print("⚠️  Benchmark evaluation requires trained model - implement model loading for full functionality")
        
    if args.ablation or args.full:
        print("\n🔬 Running ablation studies...")
        # Note: This would require loading trained orchestrator
        print("⚠️  Ablation studies require trained model - implement model loading for full functionality")
    
    if args.comparison or args.full:
        print("\n📊 Running before/after comparison...")
        # Note: This would require both untrained and trained orchestrators
        print("⚠️  Before/after comparison requires both untrained and trained models")
    
    # Analyze available training progress
    if training_metrics:
        learning_analysis = evaluator.evaluate_learning_progress(training_metrics)
        print(f"\n📈 Training Analysis Complete:")
        print(f"   Final Quality: {learning_analysis.get('final_performance', {}).get('quality', 0):.3f}")
        print(f"   Quality Trend: {learning_analysis.get('trends', {}).get('quality_trend', 0):+.4f} per episode")
        print(f"   Learning Speed: {learning_analysis.get('learning_characteristics', {}).get('learning_speed', 'Unknown')}")
    
    # Create dashboard and report
    dashboard_path = evaluator.create_evaluation_dashboard(
        training_metrics, benchmark_results, ablation_results
    )
    
    report_path = evaluator.create_comprehensive_evaluation_report(
        training_metrics, benchmark_results, ablation_results, comparison_results
    )
    
    print(f"\n✅ Comprehensive evaluation completed!")
    print(f"📊 Dashboard: {dashboard_path}")
    print(f"📋 Report: {report_path}")
    print(f"📁 Results directory: {evaluator.results_dir}")

if __name__ == "__main__":
    main()