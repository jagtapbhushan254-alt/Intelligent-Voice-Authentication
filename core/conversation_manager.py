"""
SecureX-Assist - Context-Aware Conversation Manager
Maintains conversation history and provides context-aware responses
"""

import time
import json
import logging
from typing import List, Dict, Optional, Tuple
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import re

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single conversation turn (user input + assistant response)"""
    timestamp: datetime
    user_id: Optional[int]
    user_input: str
    intent: Optional[str]
    assistant_response: str
    confidence: float
    context_used: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    def age_seconds(self) -> float:
        """Get age of this turn in seconds"""
        return (datetime.now() - self.timestamp).total_seconds()


class ConversationContext:
    """Manages conversation context and history"""
    
    def __init__(self, max_history: int = 10, context_timeout: int = 300):
        """
        Args:
            max_history: Maximum number of turns to keep
            context_timeout: Seconds after which context expires
        """
        self.max_history = max_history
        self.context_timeout = context_timeout
        
        # Conversation history per user
        self.history: Dict[int, deque] = {}
        
        # Current topic tracking
        self.current_topics: Dict[int, List[str]] = {}
        
        # User preferences learned over time
        self.user_preferences: Dict[int, Dict] = {}
        
        logger.info(f"ConversationContext initialized: max_history={max_history}, "
                   f"timeout={context_timeout}s")
    
    def add_turn(self, user_id: Optional[int], user_input: str, 
                 assistant_response: str, intent: Optional[str] = None,
                 confidence: float = 0.0) -> None:
        """Add a conversation turn to history"""
        if user_id is None:
            user_id = 0  # Default for non-authenticated sessions
        
        # Initialize history for new user
        if user_id not in self.history:
            self.history[user_id] = deque(maxlen=self.max_history)
            self.current_topics[user_id] = []
            self.user_preferences[user_id] = {}
        
        # Create turn
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_id=user_id,
            user_input=user_input,
            intent=intent,
            assistant_response=assistant_response,
            confidence=confidence
        )
        
        # Add to history
        self.history[user_id].append(turn)
        
        # Update topics
        self._update_topics(user_id, user_input, intent)
        
        # Learn preferences
        self._learn_preferences(user_id, user_input, intent)
        
        logger.debug(f"Added turn for user {user_id}: '{user_input[:50]}...'")
    
    def get_context(self, user_id: Optional[int], 
                   include_responses: bool = True) -> str:
        """Get conversation context as formatted string"""
        if user_id is None:
            user_id = 0
        
        if user_id not in self.history or not self.history[user_id]:
            return ""
        
        # Clean up expired turns
        self._cleanup_expired(user_id)
        
        if not self.history[user_id]:
            return ""
        
        # Build context string
        context_parts = []
        for turn in list(self.history[user_id])[-5:]:  # Last 5 turns
            context_parts.append(f"User: {turn.user_input}")
            if include_responses:
                context_parts.append(f"Assistant: {turn.assistant_response}")
        
        return "\n".join(context_parts)
    
    def get_last_intent(self, user_id: Optional[int]) -> Optional[str]:
        """Get the last intent from history"""
        if user_id is None:
            user_id = 0
        
        if user_id not in self.history or not self.history[user_id]:
            return None
        
        self._cleanup_expired(user_id)
        
        # Get last non-None intent
        for turn in reversed(self.history[user_id]):
            if turn.intent:
                return turn.intent
        
        return None
    
    def get_last_entity(self, user_id: Optional[int], 
                       entity_type: str) -> Optional[str]:
        """Extract last mentioned entity of a type"""
        if user_id is None:
            user_id = 0
        
        if user_id not in self.history:
            return None
        
        # Search recent history for entity
        for turn in reversed(self.history[user_id]):
            # Try to extract entity based on type
            text = turn.user_input.lower()
            
            if entity_type == "time":
                # Extract time references
                time_patterns = [
                    r'\b(tomorrow|today|yesterday)\b',
                    r'\b(\d{1,2}:\d{2})\b',
                    r'\b(morning|afternoon|evening|night)\b'
                ]
                for pattern in time_patterns:
                    match = re.search(pattern, text)
                    if match:
                        return match.group(1)
            
            elif entity_type == "location":
                # Extract location
                location_pattern = r'\b(?:in|at|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
                match = re.search(location_pattern, turn.user_input)
                if match:
                    return match.group(1)
            
            elif entity_type == "app":
                # Extract application names
                apps = ["notepad", "calculator", "chrome", "youtube", 
                       "spotify", "excel", "word", "powerpoint"]
                for app in apps:
                    if app in text:
                        return app
        
        return None
    
    def is_follow_up(self, user_id: Optional[int], current_input: str) -> bool:
        """Determine if current input is a follow-up question"""
        if user_id is None:
            user_id = 0
        
        if user_id not in self.history or not self.history[user_id]:
            return False
        
        current_lower = current_input.lower().strip()
        
        # Follow-up indicators
        follow_up_patterns = [
            r'^(and|also|then|next)\b',
            r'^(what about|how about)\b',
            r'^(show me|tell me) (more|another)',
            r'\b(too|as well|either)$',
            r'^(that|it|this)\b',
            r'^(there|here)\b',
        ]
        
        for pattern in follow_up_patterns:
            if re.search(pattern, current_lower):
                return True
        
        # Check for pronoun references
        pronouns = ['it', 'that', 'this', 'there', 'here', 'they', 'them']
        words = current_lower.split()
        if words and words[0] in pronouns:
            return True
        
        # Short questions are often follow-ups
        if len(words) <= 3 and '?' in current_input:
            return True
        
        return False
    
    def resolve_context(self, user_id: Optional[int], 
                       current_input: str) -> Tuple[str, Dict]:
        """Resolve context and enhance current input with context"""
        if user_id is None:
            user_id = 0
        
        enhanced_input = current_input
        context_info = {}
        
        if not self.is_follow_up(user_id, current_input):
            return enhanced_input, context_info
        
        # Get last intent and entities
        last_intent = self.get_last_intent(user_id)
        if last_intent:
            context_info['last_intent'] = last_intent
        
        # Try to resolve references
        current_lower = current_input.lower()
        
        # Replace pronouns with actual references
        if user_id in self.history and self.history[user_id]:
            last_turn = self.history[user_id][-1]
            
            # Check for "it" or "that" referring to last topic
            if re.search(r'\b(it|that|this)\b', current_lower):
                # Extract noun from last input
                last_input = last_turn.user_input.lower()
                # Simple noun extraction (can be improved)
                words = last_input.split()
                for i, word in enumerate(words):
                    if word in ['open', 'play', 'show', 'find', 'search']:
                        if i + 1 < len(words):
                            noun = words[i + 1]
                            enhanced_input = current_input.replace('it', noun)
                            enhanced_input = enhanced_input.replace('that', noun)
                            enhanced_input = enhanced_input.replace('this', noun)
                            context_info['resolved_entity'] = noun
                            break
        
        # Add context for incomplete queries
        if len(current_input.split()) <= 3:
            recent_context = self.get_context(user_id, include_responses=False)
            if recent_context:
                context_info['recent_context'] = recent_context
        
        logger.debug(f"Context resolution: '{current_input}' -> '{enhanced_input}'")
        return enhanced_input, context_info
    
    def get_suggestions(self, user_id: Optional[int]) -> List[str]:
        """Get smart suggestions based on history and patterns"""
        if user_id is None:
            user_id = 0
        
        suggestions = []
        
        if user_id not in self.history:
            # Default suggestions for new users
            return [
                "What can you do?",
                "Check the weather",
                "Open calculator",
                "What time is it?"
            ]
        
        # Analyze patterns
        if user_id in self.user_preferences:
            prefs = self.user_preferences[user_id]
            
            # Suggest based on frequent intents
            if 'frequent_intents' in prefs:
                for intent, count in prefs['frequent_intents'].items():
                    if count >= 2:
                        if intent == "weather":
                            suggestions.append("Check weather")
                        elif intent == "open_app":
                            suggestions.append("Open application")
                        elif intent == "search":
                            suggestions.append("Search for something")
        
        # Time-based suggestions
        current_hour = datetime.now().hour
        if 6 <= current_hour < 12:
            suggestions.append("Good morning routines")
        elif 18 <= current_hour < 22:
            suggestions.append("Good evening routines")
        
        # Limit suggestions
        return suggestions[:5]
    
    def _update_topics(self, user_id: int, user_input: str, 
                      intent: Optional[str]) -> None:
        """Update current conversation topics"""
        # Extract keywords (simple implementation)
        words = user_input.lower().split()
        keywords = [w for w in words if len(w) > 4]
        
        if keywords:
            self.current_topics[user_id] = keywords[-3:]  # Keep last 3 keywords
    
    def _learn_preferences(self, user_id: int, user_input: str,
                          intent: Optional[str]) -> None:
        """Learn user preferences from interactions"""
        if intent:
            if 'frequent_intents' not in self.user_preferences[user_id]:
                self.user_preferences[user_id]['frequent_intents'] = {}
            
            intents = self.user_preferences[user_id]['frequent_intents']
            intents[intent] = intents.get(intent, 0) + 1
        
        # Track time patterns
        hour = datetime.now().hour
        if 'active_hours' not in self.user_preferences[user_id]:
            self.user_preferences[user_id]['active_hours'] = {}
        
        hours = self.user_preferences[user_id]['active_hours']
        hours[hour] = hours.get(hour, 0) + 1
    
    def _cleanup_expired(self, user_id: int) -> None:
        """Remove expired turns from history"""
        if user_id not in self.history:
            return
        
        # Remove turns older than timeout
        current_history = self.history[user_id]
        valid_turns = deque(maxlen=self.max_history)
        
        for turn in current_history:
            if turn.age_seconds() < self.context_timeout:
                valid_turns.append(turn)
        
        self.history[user_id] = valid_turns
    
    def clear_context(self, user_id: Optional[int] = None) -> None:
        """Clear conversation context"""
        if user_id is None:
            # Clear all
            self.history.clear()
            self.current_topics.clear()
            logger.info("Cleared all conversation contexts")
        else:
            # Clear specific user
            if user_id in self.history:
                self.history[user_id].clear()
                self.current_topics[user_id].clear()
                logger.info(f"Cleared context for user {user_id}")
    
    def get_stats(self, user_id: Optional[int] = None) -> Dict:
        """Get conversation statistics"""
        if user_id is None or user_id not in self.history:
            total_turns = sum(len(h) for h in self.history.values())
            return {
                'total_users': len(self.history),
                'total_turns': total_turns,
                'avg_turns_per_user': total_turns / len(self.history) if self.history else 0
            }
        
        self._cleanup_expired(user_id)
        history = self.history[user_id]
        
        if not history:
            return {'turns': 0}
        
        # Calculate stats
        total_confidence = sum(t.confidence for t in history)
        avg_confidence = total_confidence / len(history) if history else 0
        
        intent_counts = {}
        for turn in history:
            if turn.intent:
                intent_counts[turn.intent] = intent_counts.get(turn.intent, 0) + 1
        
        return {
            'turns': len(history),
            'avg_confidence': avg_confidence,
            'intent_distribution': intent_counts,
            'preferences': self.user_preferences.get(user_id, {}),
            'current_topics': self.current_topics.get(user_id, [])
        }
    
    def export_history(self, user_id: int, filepath: str) -> bool:
        """Export conversation history to JSON"""
        try:
            if user_id not in self.history:
                return False
            
            history_data = [turn.to_dict() for turn in self.history[user_id]]
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'user_id': user_id,
                    'exported_at': datetime.now().isoformat(),
                    'turns': history_data
                }, f, indent=2)
            
            logger.info(f"Exported history for user {user_id} to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export history: {e}")
            return False


# Global conversation manager
_conversation_manager: Optional[ConversationContext] = None


def get_conversation_manager(max_history: int = 10, 
                            context_timeout: int = 300) -> ConversationContext:
    """Get or create global conversation manager"""
    global _conversation_manager
    
    if _conversation_manager is None:
        _conversation_manager = ConversationContext(max_history, context_timeout)
    
    return _conversation_manager


if __name__ == "__main__":
    # Test conversation manager
    manager = ConversationContext(max_history=5)
    
    # Simulate conversation
    manager.add_turn(1, "What's the weather?", "It's 72°F and sunny", "weather", 0.95)
    manager.add_turn(1, "How about tomorrow?", "Tomorrow will be 68°F", "weather", 0.85)
    
    # Test follow-up detection
    print(f"Is follow-up: {manager.is_follow_up(1, 'How about tomorrow?')}")
    
    # Test context resolution
    enhanced, context = manager.resolve_context(1, "And the day after?")
    print(f"Enhanced: {enhanced}")
    print(f"Context: {context}")
    
    # Get stats
    stats = manager.get_stats(1)
    print(f"\nStats: {json.dumps(stats, indent=2)}")
    
    # Get suggestions
    suggestions = manager.get_suggestions(1)
    print(f"\nSuggestions: {suggestions}")
