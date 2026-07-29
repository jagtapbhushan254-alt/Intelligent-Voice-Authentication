"""
SecureX-Assist - Voice Memory System
Remembers personal facts and user information across sessions
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import re

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Single memory entry"""
    key: str
    value: Any
    category: str  # personal, preference, schedule, relationship, skill
    confidence: float
    created_at: str
    last_accessed: str
    access_count: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


class VoiceMemory:
    """
    Voice Memory System - Remembers personal information
    
    Categories:
    - Personal: Name, birthday, age, location, etc.
    - Preferences: Favorite color, food, music, etc.
    - Schedule: Regular activities, appointments
    - Relationships: Family, friends, colleagues
    - Skills: What user knows, interests
    """
    
    def __init__(self, storage_path: str = "voice_memory.json"):
        self.storage_path = storage_path
        self.memories: Dict[str, Dict[str, MemoryEntry]] = {}  # user_id -> {key -> entry}
        
        # Pattern matchers for extracting information
        self.patterns = self._init_patterns()
        
        # Load existing memories
        self.load_memories()
        
        logger.info(f"VoiceMemory initialized with {sum(len(m) for m in self.memories.values())} memories")
    
    def _init_patterns(self) -> Dict[str, List[tuple]]:
        """Initialize regex patterns for extracting information"""
        return {
            'name': [
                (r"my name is (\w+)", 'name'),
                (r"i'm (\w+)", 'name'),
                (r"call me (\w+)", 'name'),
                (r"i am (\w+)", 'name')
            ],
            'birthday': [
                (r"my birthday is (\w+ \d+(?:st|nd|rd|th)?)", 'birthday'),
                (r"i was born on (\w+ \d+(?:st|nd|rd|th)?)", 'birthday'),
                (r"born (\w+ \d+(?:st|nd|rd|th)?)", 'birthday')
            ],
            'favorite': [
                (r"my favorite (\w+) is (\w+(?:\s\w+)*)", 'preference'),
                (r"i like (\w+)", 'preference'),
                (r"i love (\w+)", 'preference'),
                (r"i prefer (\w+)", 'preference')
            ],
            'location': [
                (r"i live in (\w+(?:\s\w+)*)", 'location'),
                (r"i'm from (\w+(?:\s\w+)*)", 'location'),
                (r"my city is (\w+(?:\s\w+)*)", 'location')
            ],
            'relationship': [
                (r"my (\w+) is (\w+)", 'relationship'),
                (r"this is my (\w+) (\w+)", 'relationship')
            ]
        }
    
    def learn_from_text(self, user_id: str, text: str) -> List[str]:
        """
        Extract and store information from user's text
        
        Returns:
            List of learned facts
        """
        learned = []
        text_lower = text.lower()
        
        # Try each pattern category
        for category, patterns in self.patterns.items():
            for pattern, mem_type in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    if category == 'name':
                        name = match.group(1).capitalize()
                        self.store(user_id, 'name', name, 'personal', 0.9)
                        learned.append(f"Your name is {name}")
                    
                    elif category == 'birthday':
                        birthday = match.group(1)
                        self.store(user_id, 'birthday', birthday, 'personal', 0.9)
                        learned.append(f"Your birthday is {birthday}")
                    
                    elif category == 'favorite':
                        groups = match.groups()
                        if len(groups) == 2:
                            item_type, item = groups
                            key = f"favorite_{item_type}"
                            self.store(user_id, key, item, 'preference', 0.8)
                            learned.append(f"Your favorite {item_type} is {item}")
                        elif len(groups) == 1:
                            item = groups[0]
                            self.store(user_id, 'likes', item, 'preference', 0.7)
                            learned.append(f"You like {item}")
                    
                    elif category == 'location':
                        location = match.group(1)
                        self.store(user_id, 'location', location, 'personal', 0.9)
                        learned.append(f"You live in {location}")
                    
                    elif category == 'relationship':
                        relation, name = match.groups()
                        key = f"relationship_{relation}"
                        self.store(user_id, key, name.capitalize(), 'relationship', 0.8)
                        learned.append(f"Your {relation} is {name}")
        
        return learned
    
    def store(self, user_id: str, key: str, value: Any, category: str, 
             confidence: float = 1.0) -> None:
        """Store a memory"""
        if user_id not in self.memories:
            self.memories[user_id] = {}
        
        now = datetime.now().isoformat()
        
        # Check if memory exists
        if key in self.memories[user_id]:
            entry = self.memories[user_id][key]
            entry.value = value
            entry.confidence = max(entry.confidence, confidence)
            entry.last_accessed = now
            entry.access_count += 1
        else:
            entry = MemoryEntry(
                key=key,
                value=value,
                category=category,
                confidence=confidence,
                created_at=now,
                last_accessed=now,
                access_count=1
            )
            self.memories[user_id][key] = entry
        
        self.save_memories()
        logger.info(f"Stored memory for {user_id}: {key} = {value}")
    
    def recall(self, user_id: str, key: str) -> Optional[Any]:
        """Recall a specific memory"""
        if user_id not in self.memories:
            return None
        
        if key in self.memories[user_id]:
            entry = self.memories[user_id][key]
            entry.last_accessed = datetime.now().isoformat()
            entry.access_count += 1
            self.save_memories()
            return entry.value
        
        return None
    
    def search(self, user_id: str, query: str) -> List[tuple]:
        """
        Search memories by query
        
        Returns:
            List of (key, value) tuples
        """
        if user_id not in self.memories:
            return []
        
        query_lower = query.lower()
        results = []
        
        for key, entry in self.memories[user_id].items():
            if query_lower in key.lower() or query_lower in str(entry.value).lower():
                results.append((key, entry.value))
        
        return results
    
    def get_all_memories(self, user_id: str, category: Optional[str] = None) -> Dict[str, Any]:
        """Get all memories for a user, optionally filtered by category"""
        if user_id not in self.memories:
            return {}
        
        memories = self.memories[user_id]
        
        if category:
            return {
                key: entry.value
                for key, entry in memories.items()
                if entry.category == category
            }
        
        return {key: entry.value for key, entry in memories.items()}
    
    def forget(self, user_id: str, key: str) -> bool:
        """Forget a specific memory"""
        if user_id in self.memories and key in self.memories[user_id]:
            del self.memories[user_id][key]
            self.save_memories()
            logger.info(f"Forgot memory for {user_id}: {key}")
            return True
        return False
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user profile"""
        if user_id not in self.memories:
            return {}
        
        profile = {
            'personal': {},
            'preferences': {},
            'relationships': {},
            'schedule': {},
            'skills': {}
        }
        
        for key, entry in self.memories[user_id].items():
            profile[entry.category][key] = entry.value
        
        return profile
    
    def check_birthday(self, user_id: str) -> Optional[str]:
        """Check if today is user's birthday"""
        birthday = self.recall(user_id, 'birthday')
        if not birthday:
            return None
        
        try:
            # Parse birthday (assuming format like "May 15" or "May 15th")
            from datetime import datetime
            birthday_clean = re.sub(r'(st|nd|rd|th)', '', birthday)
            birthday_date = datetime.strptime(f"{birthday_clean} 2000", "%B %d %Y")
            
            today = datetime.now()
            if today.month == birthday_date.month and today.day == birthday_date.day:
                return f"🎉 Happy Birthday! I remembered that your birthday is {birthday}!"
        except Exception as e:
            logger.debug(f"Birthday check failed: {e}")
        
        return None
    
    def generate_summary(self, user_id: str) -> str:
        """Generate a natural language summary of what we know"""
        profile = self.get_user_profile(user_id)
        
        if not any(profile.values()):
            return "I don't know much about you yet. Tell me about yourself!"
        
        summary_parts = []
        
        # Personal info
        if profile['personal']:
            name = profile['personal'].get('name')
            location = profile['personal'].get('location')
            birthday = profile['personal'].get('birthday')
            
            if name:
                summary_parts.append(f"Your name is {name}.")
            if location:
                summary_parts.append(f"You live in {location}.")
            if birthday:
                summary_parts.append(f"Your birthday is {birthday}.")
        
        # Preferences
        if profile['preferences']:
            prefs = []
            for key, value in profile['preferences'].items():
                if key.startswith('favorite_'):
                    item_type = key.replace('favorite_', '')
                    prefs.append(f"favorite {item_type} is {value}")
            
            if prefs:
                summary_parts.append("I know " + ", ".join(prefs) + ".")
        
        # Relationships
        if profile['relationships']:
            rels = []
            for key, value in profile['relationships'].items():
                if key.startswith('relationship_'):
                    relation = key.replace('relationship_', '')
                    rels.append(f"your {relation} is {value}")
            
            if rels:
                summary_parts.append("I know " + ", ".join(rels) + ".")
        
        return " ".join(summary_parts) if summary_parts else "Tell me more about yourself!"
    
    def save_memories(self) -> bool:
        """Save memories to file"""
        try:
            data = {
                user_id: {
                    key: entry.to_dict()
                    for key, entry in memories.items()
                }
                for user_id, memories in self.memories.items()
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved memories to {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")
            return False
    
    def load_memories(self) -> bool:
        """Load memories from file"""
        try:
            if not os.path.exists(self.storage_path):
                logger.info("No existing memory file")
                return True
            
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct memory entries
            for user_id, memories in data.items():
                self.memories[user_id] = {
                    key: MemoryEntry(**entry_data)
                    for key, entry_data in memories.items()
                }
            
            total_memories = sum(len(m) for m in self.memories.values())
            logger.info(f"Loaded {total_memories} memories from {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
            return False


# Global instance
_voice_memory: Optional[VoiceMemory] = None


def get_voice_memory(storage_path: str = "voice_memory.json") -> VoiceMemory:
    """Get or create global voice memory"""
    global _voice_memory
    
    if _voice_memory is None:
        _voice_memory = VoiceMemory(storage_path)
    
    return _voice_memory


if __name__ == "__main__":
    # Test voice memory
    memory = VoiceMemory("test_memory.json")
    
    user_id = "test_user"
    
    # Test learning from text
    print("Testing learning from text:")
    learned = memory.learn_from_text(user_id, "My name is Alice and I live in New York")
    for fact in learned:
        print(f"  Learned: {fact}")
    
    learned = memory.learn_from_text(user_id, "My favorite color is blue")
    for fact in learned:
        print(f"  Learned: {fact}")
    
    learned = memory.learn_from_text(user_id, "My birthday is May 15th")
    for fact in learned:
        print(f"  Learned: {fact}")
    
    # Test recall
    print("\nTesting recall:")
    name = memory.recall(user_id, 'name')
    print(f"  Name: {name}")
    
    # Test search
    print("\nTesting search:")
    results = memory.search(user_id, 'favorite')
    for key, value in results:
        print(f"  {key}: {value}")
    
    # Test summary
    print("\nTesting summary:")
    summary = memory.generate_summary(user_id)
    print(f"  {summary}")
    
    print("\nVoice memory test complete!")
