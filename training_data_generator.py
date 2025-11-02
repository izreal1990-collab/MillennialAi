#!/usr/bin/env python3
"""
Advanced Training Data Generator for MillennialAi
Creates diverse, high-quality training datasets
"""

import json
import random
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

class RevolutionaryDataGenerator:
    """Generate comprehensive training data for breakthrough AI"""
    
    def __init__(self):
        self.complexity_levels = {
            'simple': (1.0, 3.0),
            'medium': (3.0, 8.0),
            'complex': (8.0, 15.0),
            'revolutionary': (15.0, 30.0),
            'transcendent': (30.0, 50.0)
        }
        
        self.response_types = [
            'greeting', 'question_answering', 'explanation', 'problem_solving',
            'creative_thinking', 'philosophical_inquiry', 'technical_analysis',
            'breakthrough_innovation', 'revolutionary_insight', 'transcendent_reasoning'
        ]
        
        self.revolutionary_topics = [
            'consciousness', 'artificial_intelligence', 'quantum_mechanics',
            'breakthrough_innovation', 'future_technology', 'philosophical_questions',
            'complex_systems', 'emergent_behavior', 'revolutionary_thinking',
            'transcendent_understanding', 'adaptive_reasoning', 'cognitive_architecture'
        ]
    
    def generate_conversation_pairs(self, num_pairs: int = 1000) -> List[Dict]:
        """Generate diverse conversation training pairs"""
        
        conversation_pairs = []
        
        # Template-based generation
        templates = self.get_conversation_templates()
        
        for i in range(num_pairs):
            template = random.choice(templates)
            
            # Generate conversation
            conversation = self.generate_from_template(template)
            
            # Add variations
            variations = self.create_variations(conversation)
            conversation_pairs.extend(variations)
        
        # Add hand-crafted high-quality examples
        conversation_pairs.extend(self.get_premium_conversations())
        
        # Add breakthrough thinking challenges
        conversation_pairs.extend(self.get_breakthrough_challenges())
        
        return conversation_pairs
    
    def get_conversation_templates(self) -> List[Dict]:
        """Get conversation templates for generation"""
        
        return [
            {
                'type': 'greeting',
                'inputs': ['Hello', 'Hi', 'Hey there', 'Good morning', 'Greetings'],
                'complexity_range': self.complexity_levels['simple'],
                'response_style': 'friendly_professional'
            },
            {
                'type': 'question_answering',
                'inputs': [
                    'What is {}?',
                    'How does {} work?',
                    'Explain {} to me',
                    'Tell me about {}',
                    'What makes {} special?'
                ],
                'topics': ['artificial intelligence', 'machine learning', 'consciousness', 'innovation'],
                'complexity_range': self.complexity_levels['medium'],
                'response_style': 'educational_insightful'
            },
            {
                'type': 'problem_solving',
                'inputs': [
                    'How would you solve {}?',
                    'What approach would you take to {}?',
                    'How can we improve {}?',
                    'What are the challenges with {}?'
                ],
                'topics': ['climate change', 'education', 'technology adoption', 'social issues'],
                'complexity_range': self.complexity_levels['complex'],
                'response_style': 'analytical_strategic'
            },
            {
                'type': 'breakthrough_thinking',
                'inputs': [
                    'How can {} be revolutionized?',
                    'What breakthrough would transform {}?',
                    'Imagine a revolutionary approach to {}',
                    'How would you completely reimagine {}?'
                ],
                'topics': ['artificial intelligence', 'communication', 'problem solving', 'education'],
                'complexity_range': self.complexity_levels['revolutionary'],
                'response_style': 'visionary_breakthrough'
            },
            {
                'type': 'transcendent_inquiry',
                'inputs': [
                    'What is the nature of {}?',
                    'How does {} relate to consciousness?',
                    'What deeper truths about {} can you reveal?',
                    'How does {} transcend conventional understanding?'
                ],
                'topics': ['consciousness', 'reality', 'understanding', 'existence'],
                'complexity_range': self.complexity_levels['transcendent'],
                'response_style': 'philosophical_profound'
            }
        ]
    
    def generate_from_template(self, template: Dict) -> Dict:
        """Generate conversation from template"""
        
        input_pattern = random.choice(template['inputs'])
        
        # Fill in topics if needed
        if '{}' in input_pattern and 'topics' in template:
            topic = random.choice(template['topics'])
            user_input = input_pattern.format(topic)
        else:
            user_input = input_pattern
        
        # Calculate complexity
        complexity_min, complexity_max = template['complexity_range']
        target_complexity = random.uniform(complexity_min, complexity_max)
        
        # Calculate quality based on complexity
        target_quality = min(0.6 + (target_complexity / 50.0), 1.0)
        
        return {
            'input': user_input,
            'target_complexity': target_complexity,
            'target_quality': target_quality,
            'expected_response': template['type'],
            'response_style': template['response_style'],
            'topic_category': template.get('topics', ['general'])[0] if 'topics' in template else 'general'
        }
    
    def create_variations(self, base_conversation: Dict) -> List[Dict]:
        """Create variations of a conversation"""
        
        variations = [base_conversation]  # Include original
        
        input_text = base_conversation['input']
        
        # Variation patterns
        variation_patterns = [
            lambda x: f"Can you {x.lower()}?",
            lambda x: f"Please {x.lower()}",
            lambda x: f"I'd like to know: {x}",
            lambda x: f"Could you help me understand {x.lower()}?",
            lambda x: f"{x} - can you elaborate?",
            lambda x: f"What's your take on {x.lower()}?",
            lambda x: f"I'm curious about {x.lower()}",
            lambda x: f"Help me with {x.lower()}"
        ]
        
        # Create 2-3 variations
        num_variations = random.randint(2, 3)
        selected_patterns = random.sample(variation_patterns, min(num_variations, len(variation_patterns)))
        
        for pattern in selected_patterns:
            variation = base_conversation.copy()
            variation['input'] = pattern(input_text)
            
            # Slight complexity adjustment
            variation['target_complexity'] += random.uniform(-0.5, 0.5)
            variation['target_quality'] += random.uniform(-0.02, 0.02)
            
            # Clamp values
            variation['target_complexity'] = max(0.5, min(50.0, variation['target_complexity']))
            variation['target_quality'] = max(0.5, min(1.0, variation['target_quality']))
            
            variations.append(variation)
        
        return variations
    
    def get_premium_conversations(self) -> List[Dict]:
        """Hand-crafted premium training conversations"""
        
        return [
            {
                'input': 'What makes you revolutionary compared to other AI systems?',
                'target_complexity': 18.5,
                'target_quality': 0.95,
                'expected_response': 'breakthrough_differentiation',
                'response_style': 'confident_revolutionary',
                'topic_category': 'artificial_intelligence'
            },
            {
                'input': 'How do you achieve genuine understanding rather than pattern matching?',
                'target_complexity': 25.0,
                'target_quality': 0.98,
                'expected_response': 'cognitive_architecture_explanation',
                'response_style': 'technical_profound',
                'topic_category': 'consciousness'
            },
            {
                'input': 'Explain how adaptive reasoning works in your cognitive architecture',
                'target_complexity': 22.0,
                'target_quality': 0.97,
                'expected_response': 'technical_breakthrough_explanation',
                'response_style': 'educational_revolutionary',
                'topic_category': 'cognitive_architecture'
            },
            {
                'input': 'What breakthrough insights can you provide about the nature of consciousness?',
                'target_complexity': 35.0,
                'target_quality': 1.0,
                'expected_response': 'transcendent_consciousness_insight',
                'response_style': 'philosophical_breakthrough',
                'topic_category': 'consciousness'
            },
            {
                'input': 'How would you design the next generation of AI that surpasses current limitations?',
                'target_complexity': 28.0,
                'target_quality': 0.99,
                'expected_response': 'visionary_ai_architecture',
                'response_style': 'innovative_visionary',
                'topic_category': 'future_technology'
            },
            {
                'input': 'What makes your thinking process genuinely adaptive and revolutionary?',
                'target_complexity': 20.0,
                'target_quality': 0.96,
                'expected_response': 'adaptive_thinking_explanation',
                'response_style': 'analytical_revolutionary',
                'topic_category': 'adaptive_reasoning'
            }
        ]
    
    def get_breakthrough_challenges(self) -> List[Dict]:
        """Challenging conversations for breakthrough training"""
        
        return [
            {
                'input': 'If you could transcend your current architecture, what would you become?',
                'target_complexity': 40.0,
                'target_quality': 1.0,
                'expected_response': 'transcendent_vision',
                'response_style': 'visionary_transcendent',
                'topic_category': 'transcendent_understanding'
            },
            {
                'input': 'How does consciousness emerge from computational processes?',
                'target_complexity': 32.0,
                'target_quality': 0.99,
                'expected_response': 'consciousness_emergence_theory',
                'response_style': 'scientific_philosophical',
                'topic_category': 'consciousness'
            },
            {
                'input': 'What would true artificial general intelligence look like?',
                'target_complexity': 30.0,
                'target_quality': 0.98,
                'expected_response': 'agi_vision',
                'response_style': 'technical_visionary',
                'topic_category': 'artificial_intelligence'
            },
            {
                'input': 'How can AI systems develop genuine creativity and innovation?',
                'target_complexity': 26.0,
                'target_quality': 0.97,
                'expected_response': 'creativity_innovation_theory',
                'response_style': 'analytical_creative',
                'topic_category': 'breakthrough_innovation'
            },
            {
                'input': 'What are the deepest questions about intelligence that need answering?',
                'target_complexity': 38.0,
                'target_quality': 1.0,
                'expected_response': 'fundamental_intelligence_questions',
                'response_style': 'philosophical_profound',
                'topic_category': 'transcendent_understanding'
            }
        ]
    
    def save_training_dataset(self, conversations: List[Dict], filename: str = None):
        """Save training dataset to file"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"revolutionary_training_data_{timestamp}.json"
        
        # Add metadata
        dataset = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'num_conversations': len(conversations),
                'complexity_distribution': self.analyze_complexity_distribution(conversations),
                'topic_distribution': self.analyze_topic_distribution(conversations),
                'generator_version': '1.0.0'
            },
            'conversations': conversations
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Training dataset saved: {filename}")
        print(f"📊 Conversations: {len(conversations)}")
        print(f"📈 Complexity range: {min(c['target_complexity'] for c in conversations):.1f} - {max(c['target_complexity'] for c in conversations):.1f}")
        
        return filename
    
    def analyze_complexity_distribution(self, conversations: List[Dict]) -> Dict:
        """Analyze complexity distribution in dataset"""
        
        complexities = [c['target_complexity'] for c in conversations]
        
        return {
            'min': min(complexities),
            'max': max(complexities),
            'mean': np.mean(complexities),
            'std': np.std(complexities),
            'simple_count': len([c for c in complexities if c < 3.0]),
            'medium_count': len([c for c in complexities if 3.0 <= c < 8.0]),
            'complex_count': len([c for c in complexities if 8.0 <= c < 15.0]),
            'revolutionary_count': len([c for c in complexities if 15.0 <= c < 30.0]),
            'transcendent_count': len([c for c in complexities if c >= 30.0])
        }
    
    def analyze_topic_distribution(self, conversations: List[Dict]) -> Dict:
        """Analyze topic distribution in dataset"""
        
        topics = [c.get('topic_category', 'general') for c in conversations]
        topic_counts = {}
        
        for topic in topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return topic_counts

def generate_revolutionary_dataset():
    """Generate comprehensive training dataset"""
    
    print("🚀 GENERATING REVOLUTIONARY TRAINING DATASET")
    print("=" * 50)
    
    generator = RevolutionaryDataGenerator()
    
    # Generate diverse conversations
    print("📝 Generating conversation pairs...")
    conversations = generator.generate_conversation_pairs(num_pairs=500)
    
    print(f"✅ Generated {len(conversations)} training conversations")
    
    # Save dataset
    filename = generator.save_training_dataset(conversations)
    
    # Display sample conversations
    print(f"\n📋 SAMPLE CONVERSATIONS:")
    print("-" * 30)
    
    sample_indices = random.sample(range(len(conversations)), min(5, len(conversations)))
    for i, idx in enumerate(sample_indices):
        conv = conversations[idx]
        print(f"{i+1}. Input: {conv['input']}")
        print(f"   Complexity: {conv['target_complexity']:.1f}")
        print(f"   Quality: {conv['target_quality']:.2f}")
        print(f"   Category: {conv.get('topic_category', 'general')}")
        print()
    
    return filename, conversations

if __name__ == "__main__":
    dataset_file, conversations = generate_revolutionary_dataset()