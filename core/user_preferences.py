"""
SecureX-Assist - User Preference Learning System
Learns user preferences, habits, and patterns for ultra-personalized experience
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import threading

logger = logging.getLogger(__name__)


class UserPreferenceLearning:
    """Learn and adapt to user preferences for personalized experience"""
    
    def __init__(self, config: dict):
        self.config = config
        self.preferences_file = Path("user_preferences.json")
        self.preferences = {}
        self.lock = threading.Lock()
        self.load_preferences()
        
        logger.info("📚 User Preference Learning System initialized")
    
    def load_preferences(self):
        """Load user preferences from file"""
        try:
            if self.preferences_file.exists():
                with open(self.preferences_file, 'r') as f:
                    self.preferences = json.load(f)
                logger.info(f"✅ Loaded preferences for {len(self.preferences)} users")
        except Exception as e:
            logger.error(f"Failed to load preferences: {e}")
            self.preferences = {}
    
    def save_preferences(self):
        """Save preferences to file"""
        try:
            with self.lock:
                with open(self.preferences_file, 'w') as f:
                    json.dump(self.preferences, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get or create user preference profile"""
        with self.lock:
            if user_id not in self.preferences:
                self.preferences[user_id] = {
                    'created_at': datetime.now().isoformat(),
                    'interaction_count': 0,
                    'favorite_commands': {},
                    'preferred_voice_speed': 1.0,
                    'preferred_volume': 1.0,
                    'preferred_personality': 'friendly',
                    'command_history': [],
                    'time_preferences': {
                        'morning': [],
                        'afternoon': [],
                        'evening': [],
                        'night': []
                    },
                    'topics_of_interest': [],
                    'frequently_asked': [],
                    'app_preferences': {},
                    'custom_shortcuts': {},
                    'learning_enabled': True
                }
            return self.preferences[user_id]
    
    def learn_from_interaction(self, user_id: str, command: str, success: bool, 
                               emotion: Optional[str] = None):
        """Learn from user interaction"""
        with self.lock:
            profile = self.get_user_profile(user_id)
            profile['interaction_count'] += 1
            
            # Track command usage
            if command not in profile['favorite_commands']:
                profile['favorite_commands'][command] = 0
            profile['favorite_commands'][command] += 1
            
            # Track by time of day
            hour = datetime.now().hour
            time_slot = self._get_time_slot(hour)
            if command not in profile['time_preferences'][time_slot]:
                profile['time_preferences'][time_slot].append(command)
            
            # Add to history (keep last 100)
            profile['command_history'].append({
                'command': command,
                'timestamp': datetime.now().isoformat(),
                'success': success,
                'emotion': emotion
            })
            profile['command_history'] = profile['command_history'][-100:]
            
            # Extract topics
            self._extract_topics(profile, command)
            
            self.save_preferences()
    
    def _get_time_slot(self, hour: int) -> str:
        """Get time slot from hour"""
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'
    
    def _extract_topics(self, profile: Dict, command: str):
        """Extract and track topics of interest"""
        keywords = {
            'weather': ['weather', 'temperature', 'rain', 'forecast'],
            'music': ['music', 'song', 'play', 'spotify'],
            'productivity': ['task', 'reminder', 'schedule', 'calendar'],
            'smart_home': ['light', 'temperature', 'thermostat', 'door'],
            'information': ['what', 'who', 'when', 'where', 'how', 'search'],
            'entertainment': ['game', 'joke', 'story', 'fun'],
            'system': ['open', 'close', 'launch', 'shut down']
        }
        
        command_lower = command.lower()
        for topic, words in keywords.items():
            if any(word in command_lower for word in words):
                if topic not in profile['topics_of_interest']:
                    profile['topics_of_interest'].append(topic)
    
    def get_personalized_greeting(self, user_id: str, user_name: str = "User") -> str:
        """Generate personalized greeting based on history"""
        profile = self.get_user_profile(user_id)
        count = profile['interaction_count']
        
        hour = datetime.now().hour
        time_greeting = self._get_time_greeting(hour)
        
        if count == 0:
            return f"{time_greeting}, {user_name}! I'm excited to assist you today! 🎉"
        elif count < 10:
            return f"{time_greeting}, {user_name}! Great to see you again! What can I help with?"
        else:
            # Use favorite topic
            topics = profile.get('topics_of_interest', [])
            if topics:
                topic = topics[0]
                return f"{time_greeting}, {user_name}! Ready to help with {topic} or anything else!"
            return f"{time_greeting}, {user_name}! Always here to assist you!"
    
    def _get_time_greeting(self, hour: int) -> str:
        """Get time-appropriate greeting"""
        if 5 <= hour < 12:
            return "Good morning"
        elif 12 <= hour < 17:
            return "Good afternoon"
        elif 17 <= hour < 21:
            return "Good evening"
        else:
            return "Hello"
    
    def get_suggested_commands(self, user_id: str, limit: int = 5) -> List[str]:
        """Get personalized command suggestions"""
        profile = self.get_user_profile(user_id)
        
        # Get top commands
        favorites = profile.get('favorite_commands', {})
        sorted_commands = sorted(favorites.items(), key=lambda x: x[1], reverse=True)
        
        # Get time-based suggestions
        hour = datetime.now().hour
        time_slot = self._get_time_slot(hour)
        time_commands = profile.get('time_preferences', {}).get(time_slot, [])
        
        # Combine and deduplicate
        suggestions = []
        for cmd, _ in sorted_commands[:limit]:
            if cmd not in suggestions:
                suggestions.append(cmd)
        
        for cmd in time_commands:
            if len(suggestions) >= limit:
                break
            if cmd not in suggestions:
                suggestions.append(cmd)
        
        return suggestions[:limit]
    
    def get_quick_actions(self, user_id: str) -> List[Dict[str, str]]:
        """Get quick action suggestions based on user patterns"""
        profile = self.get_user_profile(user_id)
        hour = datetime.now().hour
        
        quick_actions = []
        
        # Morning suggestions
        if 5 <= hour < 12:
            quick_actions.extend([
                {'icon': '☀️', 'text': 'Weather', 'command': 'What\'s the weather?'},
                {'icon': '📰', 'text': 'News', 'command': 'Tell me the news'},
                {'icon': '☕', 'text': 'Good Morning', 'command': 'Good morning assistant'}
            ])
        
        # Afternoon suggestions
        elif 12 <= hour < 17:
            quick_actions.extend([
                {'icon': '📝', 'text': 'Tasks', 'command': 'What are my tasks?'},
                {'icon': '💬', 'text': 'Chat', 'command': 'Let\'s chat'},
                {'icon': '🎵', 'text': 'Music', 'command': 'Play music'}
            ])
        
        # Evening suggestions
        else:
            quick_actions.extend([
                {'icon': '🎮', 'text': 'Games', 'command': 'Play a game'},
                {'icon': '📖', 'text': 'Story', 'command': 'Tell me a story'},
                {'icon': '🌙', 'text': 'Goodnight', 'command': 'Goodnight assistant'}
            ])
        
        # Add user favorites
        favorites = profile.get('favorite_commands', {})
        for cmd, count in sorted(favorites.items(), key=lambda x: x[1], reverse=True)[:2]:
            if len(quick_actions) >= 6:
                break
            quick_actions.append({
                'icon': '⭐',
                'text': cmd[:20],
                'command': cmd
            })
        
        return quick_actions[:6]
    
    def adjust_voice_settings(self, user_id: str, speed: Optional[float] = None,
                            volume: Optional[float] = None):
        """Learn and adjust voice settings"""
        with self.lock:
            profile = self.get_user_profile(user_id)
            if speed is not None:
                profile['preferred_voice_speed'] = speed
            if volume is not None:
                profile['preferred_volume'] = volume
            self.save_preferences()
    
    def get_voice_settings(self, user_id: str) -> Dict[str, float]:
        """Get personalized voice settings"""
        profile = self.get_user_profile(user_id)
        return {
            'speed': profile.get('preferred_voice_speed', 1.0),
            'volume': profile.get('preferred_volume', 1.0)
        }
    
    def set_personality(self, user_id: str, personality: str):
        """Set preferred personality mode"""
        with self.lock:
            profile = self.get_user_profile(user_id)
            profile['preferred_personality'] = personality
            self.save_preferences()
    
    def get_personality(self, user_id: str) -> str:
        """Get preferred personality"""
        profile = self.get_user_profile(user_id)
        return profile.get('preferred_personality', 'friendly')
    
    def get_usage_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user usage statistics"""
        profile = self.get_user_profile(user_id)
        
        return {
            'total_interactions': profile.get('interaction_count', 0),
            'member_since': profile.get('created_at', 'Unknown'),
            'favorite_commands': len(profile.get('favorite_commands', {})),
            'topics_explored': len(profile.get('topics_of_interest', [])),
            'preferred_personality': profile.get('preferred_personality', 'friendly')
        }


# Global instance
_preference_learning = None


def get_preference_learning(config: dict = None) -> UserPreferenceLearning:
    """Get global preference learning instance"""
    global _preference_learning
    if _preference_learning is None:
        _preference_learning = UserPreferenceLearning(config or {})
    return _preference_learning
