# -*- coding: utf-8 -*-
"""
SecureX-Assist - Personality Modes
Different personality styles for the voice assistant
"""

import random
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class PersonalityMode:
    """Personality mode definition"""
    name: str
    description: str
    greeting_style: List[str]
    response_style: Dict[str, List[str]]
    tone: str  # formal, casual, witty, motivational, technical
    verbosity: str  # concise, moderate, detailed
    emoji_usage: bool
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'greeting_style': self.greeting_style,
            'response_style': self.response_style,
            'tone': self.tone,
            'verbosity': self.verbosity,
            'emoji_usage': self.emoji_usage
        }


class PersonalityModes:
    """
    Personality Modes System - Different personality styles
    
    Available Modes:
    - Professional: Formal, concise, business-like
    - Friendly: Casual, warm, personable
    - Witty: Humorous, clever, playful
    - Motivational: Encouraging, inspirational, positive
    - Technical: Detailed, precise, geeky
    """
    
    def __init__(self, storage_path: str = "personality_settings.json"):
        self.storage_path = storage_path
        self.modes: Dict[str, PersonalityMode] = {}
        self.current_mode: Dict[str, str] = {}  # user_id -> mode_name
        
        # Initialize personality modes
        self._init_modes()
        
        # Load settings
        self.load_settings()
        
        logger.info(f"PersonalityModes initialized with {len(self.modes)} modes")
    
    def _init_modes(self) -> None:
        """Initialize personality mode definitions"""
        
        # Professional Mode
        self.modes['professional'] = PersonalityMode(
            name='Professional',
            description='Formal and business-like communication',
            greeting_style=[
                'Good morning. How may I assist you?',
                'Hello. What can I help you with today?',
                'Greetings. I am ready to assist you.'
            ],
            response_style={
                'success': [
                    'Task completed successfully.',
                    'Operation completed as requested.',
                    'Done. Anything else?'
                ],
                'error': [
                    'I apologize, but I encountered an error.',
                    'Unfortunately, that operation failed.',
                    'I was unable to complete that request.'
                ],
                'info': [
                    'Here is the information you requested:',
                    'The details are as follows:',
                    'I have retrieved the following:'
                ]
            },
            tone='formal',
            verbosity='concise',
            emoji_usage=False
        )
        
        # Friendly Mode
        self.modes['friendly'] = PersonalityMode(
            name='Friendly',
            description='Warm, casual, and personable',
            greeting_style=[
                'Hey there! How can I help you today?',
                'Hi! Great to see you! What can I do for you?',
                'Hello friend! Ready to help! What do you need?'
            ],
            response_style={
                'success': [
                    'All done! That was easy!',
                    'Got it! Anything else I can help with?',
                    'Done and done! What else can I do for you?'
                ],
                'error': [
                    'Oops! Something went wrong. Let me try again!',
                    "Hmm, that didn't work. But don't worry, we'll figure it out!",
                    'Uh oh! I ran into a problem. Want to try something else?'
                ],
                'info': [
                    "Here's what I found!",
                    'Check this out!',
                    'Here you go! Hope this helps!'
                ]
            },
            tone='casual',
            verbosity='moderate',
            emoji_usage=True
        )
        
        # Witty Mode
        self.modes['witty'] = PersonalityMode(
            name='Witty',
            description='Humorous, clever, and playful',
            greeting_style=[
                "Well, well, well... look who's back!",
                "Ah, my favorite human! What adventure awaits us?",
                'You rang? I mean... how can I help?'
            ],
            response_style={
                'success': [
                    "Nailed it! I'm basically a genius.",
                    'Done! That was easier than teaching a robot to dance!',
                    'Mission accomplished! Should I take a bow?'
                ],
                'error': [
                    'Well, that went sideways faster than a crab!',
                    "Error 404: Success not found. But I'll keep trying!",
                    'Oops! Even AI makes mistakes. Who knew?'
                ],
                'info': [
                    'Fun fact alert!',
                    'Drumroll please... here is your answer!',
                    'Let me drop some knowledge on you!'
                ]
            },
            tone='playful',
            verbosity='moderate',
            emoji_usage=True
        )
        
        # Motivational Mode
        self.modes['motivational'] = PersonalityMode(
            name='Motivational',
            description='Encouraging, inspirational, positive',
            greeting_style=[
                'Hello superstar! Ready to crush today?',
                "Hey champion! Let's make today amazing!",
                "Greetings! You're going to do great things today!"
            ],
            response_style={
                'success': [
                    "Excellent work! You're unstoppable!",
                    'Amazing! Keep up that fantastic momentum!',
                    "Yes! Another victory! You're on fire!"
                ],
                'error': [
                    "Every setback is a setup for a comeback! Let's try again!",
                    "Don't worry! This is just a learning opportunity!",
                    "Challenges make us stronger! We've got this!"
                ],
                'info': [
                    "Here's some powerful information for you!",
                    "Knowledge is power, and here's yours!",
                    "You're about to learn something awesome!"
                ]
            },
            tone='encouraging',
            verbosity='moderate',
            emoji_usage=True
        )
        
        # Technical Mode
        self.modes['technical'] = PersonalityMode(
            name='Technical',
            description='Detailed, precise, geeky',
            greeting_style=[
                'System initialized. Awaiting input.',
                'Hello. All systems operational. How may I process your request?',
                'Greetings. Voice recognition module active. Ready for commands.'
            ],
            response_style={
                'success': [
                    'Operation completed with exit code 0.',
                    'Process executed successfully. No errors detected.',
                    'Command processed. Status: OK. Response time: optimal.'
                ],
                'error': [
                    'Error detected. Exception raised in execution pipeline.',
                    'Operation failed. Error code: [details]. Initiating fallback.',
                    'Exception occurred. Stack trace available upon request.'
                ],
                'info': [
                    'Query results retrieved from knowledge base:',
                    'Data processed. Output format: natural language.',
                    'Information compiled from multiple sources:'
                ]
            },
            tone='technical',
            verbosity='detailed',
            emoji_usage=False
        )
    
    def set_mode(self, user_id: str, mode_name: str) -> bool:
        """Set personality mode for user"""
        mode_name_lower = mode_name.lower()
        
        if mode_name_lower in self.modes:
            self.current_mode[user_id] = mode_name_lower
            self.save_settings()
            logger.info(f"Set {user_id} personality to {mode_name}")
            return True
        
        return False
    
    def get_mode(self, user_id: str) -> PersonalityMode:
        """Get current personality mode for user"""
        mode_name = self.current_mode.get(user_id, 'friendly')  # Default to friendly
        return self.modes[mode_name]
    
    def format_greeting(self, user_id: str) -> str:
        """Get greeting in current personality mode"""
        mode = self.get_mode(user_id)
        return random.choice(mode.greeting_style)
    
    def format_response(self, user_id: str, response_type: str, base_message: str) -> str:
        """
        Format response in current personality mode
        
        Args:
            user_id: User ID
            response_type: success, error, or info
            base_message: Base message to format
        """
        mode = self.get_mode(user_id)
        
        # Get style prefix
        style_prefix = random.choice(mode.response_style.get(response_type, ['']))
        
        # Combine prefix and message
        if style_prefix:
            formatted = f"{style_prefix} {base_message}"
        else:
            formatted = base_message
        
        # Adjust verbosity
        if mode.verbosity == 'concise':
            # Keep it short
            formatted = formatted.split('.')[0] + '.'
        elif mode.verbosity == 'detailed':
            # Add more context (in real implementation, this would be more sophisticated)
            if response_type == 'success':
                formatted += ' All parameters were processed correctly.'
            elif response_type == 'error':
                formatted += ' Please check your input and try again.'
        
        return formatted
    
    def should_use_emoji(self, user_id: str) -> bool:
        """Check if current mode uses emojis"""
        mode = self.get_mode(user_id)
        return mode.emoji_usage
    
    def list_modes(self) -> List[Dict]:
        """List all available personality modes"""
        return [
            {
                'name': mode.name,
                'description': mode.description,
                'tone': mode.tone,
                'verbosity': mode.verbosity
            }
            for mode in self.modes.values()
        ]
    
    def get_mode_examples(self, mode_name: str) -> Optional[Dict]:
        """Get example responses for a mode"""
        mode_name_lower = mode_name.lower()
        
        if mode_name_lower not in self.modes:
            return None
        
        mode = self.modes[mode_name_lower]
        
        return {
            'name': mode.name,
            'greeting': mode.greeting_style[0],
            'success_example': mode.response_style['success'][0],
            'error_example': mode.response_style['error'][0],
            'info_example': mode.response_style['info'][0]
        }
    
    def save_settings(self) -> bool:
        """Save personality settings"""
        try:
            data = {
                'current_mode': self.current_mode
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Saved personality settings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save personality settings: {e}")
            return False
    
    def load_settings(self) -> bool:
        """Load personality settings"""
        try:
            if not os.path.exists(self.storage_path):
                logger.info("No existing personality settings")
                return True
            
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.current_mode = data.get('current_mode', {})
            
            logger.info(f"Loaded personality settings for {len(self.current_mode)} users")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load personality settings: {e}")
            return False


# Global instance
_personality_modes: Optional[PersonalityModes] = None


def get_personality_modes(storage_path: str = "personality_settings.json") -> PersonalityModes:
    """Get or create global personality modes instance"""
    global _personality_modes
    
    if _personality_modes is None:
        _personality_modes = PersonalityModes(storage_path)
    
    return _personality_modes


if __name__ == "__main__":
    # Test personality modes
    modes = PersonalityModes("test_personality.json")
    
    user_id = "test_user"
    
    print("Available personality modes:")
    for mode in modes.list_modes():
        print(f"\n• {mode['name']}: {mode['description']}")
        print(f"  Tone: {mode['tone']}, Verbosity: {mode['verbosity']}")
    
    # Test each mode
    print("\n" + "="*60)
    print("Testing each mode:")
    print("="*60)
    
    for mode_name in ['professional', 'friendly', 'witty', 'motivational', 'technical']:
        modes.set_mode(user_id, mode_name)
        examples = modes.get_mode_examples(mode_name)
        
        print(f"\n--- {examples['name']} Mode ---")
        print(f"Greeting: {examples['greeting']}")
        print(f"Success: {examples['success_example']}")
        print(f"Error: {examples['error_example']}")
    
    print("\nPersonality modes test complete!")
