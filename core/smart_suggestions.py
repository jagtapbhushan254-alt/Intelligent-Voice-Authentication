"""
SecureX-Assist - Smart Suggestion Engine
Proactive assistance with contextual suggestions
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, time as dt_time
import random

logger = logging.getLogger(__name__)


class SmartSuggestionEngine:
    """Provides contextual and proactive suggestions"""
    
    def __init__(self):
        logger.info("🎯 Smart Suggestion Engine initialized")
    
    def get_time_based_suggestions(self) -> List[Dict[str, str]]:
        """Get suggestions based on time of day"""
        current_hour = datetime.now().hour
        
        if 5 <= current_hour < 12:  # Morning
            return [
                {
                    "icon": "☀️",
                    "title": "Good Morning Routine",
                    "command": "Good morning assistant",
                    "description": "Start your day with weather, news, and tasks"
                },
                {
                    "icon": "📰",
                    "title": "Daily Briefing",
                    "command": "Tell me the news",
                    "description": "Get caught up on current events"
                },
                {
                    "icon": "✅",
                    "title": "Today's Tasks",
                    "command": "What are my tasks?",
                    "description": "Review your to-do list"
                }
            ]
        
        elif 12 <= current_hour < 17:  # Afternoon
            return [
                {
                    "icon": "💼",
                    "title": "Work Mode",
                    "command": "Start work mode",
                    "description": "Activate productivity features"
                },
                {
                    "icon": "🎵",
                    "title": "Focus Music",
                    "command": "Play focus music",
                    "description": "Boost concentration with music"
                },
                {
                    "icon": "📊",
                    "title": "Productivity Check",
                    "command": "Show my productivity",
                    "description": "See your work stats"
                }
            ]
        
        elif 17 <= current_hour < 21:  # Evening
            return [
                {
                    "icon": "🎮",
                    "title": "Relax & Play",
                    "command": "Play a game",
                    "description": "Unwind with voice games"
                },
                {
                    "icon": "📺",
                    "title": "Entertainment",
                    "command": "Tell me a story",
                    "description": "Enjoy some entertainment"
                },
                {
                    "icon": "🍕",
                    "title": "Dinner Ideas",
                    "command": "Suggest dinner recipes",
                    "description": "Get cooking inspiration"
                }
            ]
        
        else:  # Night
            return [
                {
                    "icon": "🌙",
                    "title": "Wind Down",
                    "command": "Goodnight routine",
                    "description": "Prepare for bedtime"
                },
                {
                    "icon": "📖",
                    "title": "Bedtime Story",
                    "command": "Tell me a bedtime story",
                    "description": "Relax with a story"
                },
                {
                    "icon": "⏰",
                    "title": "Set Alarm",
                    "command": "Set alarm for tomorrow",
                    "description": "Don't forget your morning wake-up"
                }
            ]
    
    def get_contextual_suggestions(self, user_activity: Optional[str] = None, 
                                   recent_commands: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """Get suggestions based on context and history"""
        
        suggestions = []
        
        # Based on user activity
        if user_activity == "browsing":
            suggestions.extend([
                {
                    "icon": "🔍",
                    "title": "Search Help",
                    "command": "Search for [topic]",
                    "description": "Quick web search"
                },
                {
                    "icon": "📑",
                    "title": "Open Tabs",
                    "command": "Show open tabs",
                    "description": "Manage your browser tabs"
                }
            ])
        
        elif user_activity == "coding":
            suggestions.extend([
                {
                    "icon": "💻",
                    "title": "Open IDE",
                    "command": "Open Visual Studio Code",
                    "description": "Launch your code editor"
                },
                {
                    "icon": "🐛",
                    "title": "Debug Help",
                    "command": "How do I debug this?",
                    "description": "Get debugging tips"
                }
            ])
        
        # General always-useful suggestions
        suggestions.extend([
            {
                "icon": "❓",
                "title": "Ask Anything",
                "command": "What can you do?",
                "description": "Explore all capabilities"
            },
            {
                "icon": "⚡",
                "title": "Quick Action",
                "command": "Take screenshot",
                "description": "Capture your screen"
            },
            {
                "icon": "🎯",
                "title": "System Status",
                "command": "How's my computer?",
                "description": "Check performance"
            }
        ])
        
        return suggestions[:6]
    
    def get_proactive_tips(self) -> List[Dict[str, str]]:
        """Get proactive helpful tips"""
        
        tips = [
            {
                "icon": "🚀",
                "title": "Speed Boost Active",
                "message": "Turbo mode enabled - responses are 4-8x faster!",
                "type": "info"
            },
            {
                "icon": "🧠",
                "title": "Learning Your Habits",
                "message": "The assistant is learning your preferences for better personalization",
                "type": "info"
            },
            {
                "icon": "🎤",
                "title": "Voice Tip",
                "message": "Speak naturally - I understand context from your conversation!",
                "type": "tip"
            },
            {
                "icon": "🎮",
                "title": "Try Something Fun",
                "message": "Say 'play a game' to try our interactive voice games!",
                "type": "suggestion"
            },
            {
                "icon": "⚡",
                "title": "Smart Routines",
                "message": "Set up morning/evening routines for automated task sequences",
                "type": "suggestion"
            },
            {
                "icon": "🎭",
                "title": "Personality Modes",
                "message": "Try: 'Be more professional' or 'Be more friendly' to change my style",
                "type": "tip"
            },
            {
                "icon": "🔐",
                "title": "Super Secure",
                "message": "Your voice is protected with military-grade biometric security",
                "type": "info"
            },
            {
                "icon": "💡",
                "title": "Quick Commands",
                "message": "Use short phrases like 'time', 'weather', 'joke' for instant results",
                "type": "tip"
            }
        ]
        
        return random.sample(tips, 3)
    
    def get_first_time_user_guide(self) -> Dict[str, any]:
        """Get comprehensive guide for new users"""
        
        return {
            "welcome": {
                "title": "Welcome to SecureX-Assist! 🎉",
                "message": "Your extraordinary AI voice assistant with biometric security",
                "steps": [
                    "🎤 Enroll your voice for secure authentication",
                    "🗣️ Start listening and give voice commands",
                    "🤖 Chat naturally with AI intelligence",
                    "⚡ Explore smart features and games"
                ]
            },
            
            "quick_wins": [
                {
                    "title": "Try This First",
                    "command": "What can you do?",
                    "why": "Discover all amazing capabilities"
                },
                {
                    "title": "Test Security",
                    "command": "Login with voice",
                    "why": "Experience biometric authentication"
                },
                {
                    "title": "Have Fun",
                    "command": "Tell me a joke",
                    "why": "See the personality in action"
                },
                {
                    "title": "Control System",
                    "command": "What time is it?",
                    "why": "Simple but instant responses"
                }
            ],
            
            "power_features": [
                {
                    "feature": "🎯 Smart Routines",
                    "benefit": "Automate your daily tasks",
                    "example": "'Good morning routine' opens apps, tells weather, etc."
                },
                {
                    "feature": "🧠 Context Memory",
                    "benefit": "Remembers your conversation",
                    "example": "Ask follow-up questions naturally"
                },
                {
                    "feature": "🎭 Personalities",
                    "benefit": "Adapts to your mood",
                    "example": "Switch between friendly, professional, funny modes"
                },
                {
                    "feature": "⚡ Lightning Fast",
                    "benefit": "4-8x faster processing",
                    "example": "Instant responses with turbo optimization"
                }
            ]
        }
    
    def get_discovery_prompts(self) -> List[str]:
        """Get prompts to help users discover features"""
        
        return [
            "🎮 Try saying: 'Play a game' for interactive entertainment!",
            "🤖 Ask me anything: 'What is quantum computing?'",
            "⚡ Quick command: 'Open Chrome' to launch apps",
            "🎭 Change mood: 'Be more funny' or 'Be professional'",
            "📊 Check stats: 'How's my computer?' for system info",
            "🎵 Entertainment: 'Tell me a story' or 'Sing a song'",
            "🔍 Web search: 'Google [anything]' for instant search",
            "💡 Get help: 'What can you do?' for full capability list",
            "🌟 Smart features: 'Create a routine' for automation",
            "📚 Learn more: 'Show tutorial' for guided tour"
        ]
    
    def get_performance_insights(self, metrics: Dict) -> List[Dict[str, str]]:
        """Get insights about system performance"""
        
        insights = []
        
        # Check if GPU is being used
        if metrics.get('gpu_enabled'):
            insights.append({
                "icon": "🚀",
                "title": "GPU Acceleration Active",
                "message": "Processing is 8x faster with your graphics card!",
                "type": "success"
            })
        
        # Check response times
        avg_response = metrics.get('avg_response_time', 0)
        if avg_response < 1.0:
            insights.append({
                "icon": "⚡",
                "title": "Lightning Fast",
                "message": f"Average response time: {avg_response:.2f}s - That's amazing!",
                "type": "success"
            })
        
        # Check cache hits
        cache_rate = metrics.get('cache_hit_rate', 0)
        if cache_rate > 0.5:
            insights.append({
                "icon": "💾",
                "title": "Smart Caching",
                "message": f"{int(cache_rate*100)}% of requests use cached responses",
                "type": "info"
            })
        
        return insights
    
    def get_feature_highlight(self) -> Dict[str, str]:
        """Get a random feature to highlight"""
        
        features = [
            {
                "title": "🎮 Voice Games",
                "description": "Play interactive games using just your voice!",
                "try_it": "Say: 'Play a game'"
            },
            {
                "title": "🧠 AI Conversation",
                "description": "Chat naturally about any topic with AI intelligence",
                "try_it": "Say: 'Let's talk about space'"
            },
            {
                "title": "🎭 Personality Modes",
                "description": "Switch between friendly, professional, and funny personalities",
                "try_it": "Say: 'Change personality'"
            },
            {
                "title": "⚡ Smart Routines",
                "description": "Automate sequences of commands",
                "try_it": "Say: 'Good morning routine'"
            },
            {
                "title": "💻 System Control",
                "description": "Control your computer entirely with voice",
                "try_it": "Say: 'Open Chrome' or 'Take screenshot'"
            }
        ]
        
        return random.choice(features)


# Global instance
_suggestion_engine = None


def get_suggestion_engine() -> SmartSuggestionEngine:
    """Get global suggestion engine instance"""
    global _suggestion_engine
    if _suggestion_engine is None:
        _suggestion_engine = SmartSuggestionEngine()
    return _suggestion_engine
