"""
SecureX-Assist - Interactive User Guide & Help System
Smart, context-aware guidance with tutorial mode
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class InteractiveGuide:
    """Interactive guide system with tutorials and contextual help"""
    
    def __init__(self):
        self.current_tutorial = None
        self.tutorial_step = 0
        self.user_completed_tutorials = {}
        self.guide_file = Path("user_guide_progress.json")
        self.load_progress()
        
        logger.info("📖 Interactive Guide System initialized")
    
    def load_progress(self):
        """Load user tutorial progress"""
        try:
            if self.guide_file.exists():
                with open(self.guide_file, 'r') as f:
                    self.user_completed_tutorials = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load guide progress: {e}")
    
    def save_progress(self):
        """Save tutorial progress"""
        try:
            with open(self.guide_file, 'w') as f:
                json.dump(self.user_completed_tutorials, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save guide progress: {e}")
    
    def get_capabilities_overview(self) -> Dict[str, List[Dict[str, str]]]:
        """Get comprehensive capabilities overview"""
        return {
            "🔐 Authentication & Security": [
                {
                    "capability": "Voice Biometric Login",
                    "description": "Secure login using your unique voice pattern",
                    "example": "Say: 'Log me in' or 'Authenticate'",
                    "icon": "🎤"
                },
                {
                    "capability": "Face Recognition",
                    "description": "Multi-modal authentication with facial recognition",
                    "example": "Enable camera for face verification",
                    "icon": "👤"
                },
                {
                    "capability": "Continuous Authentication",
                    "description": "Real-time security monitoring during your session",
                    "example": "Automatic verification while you work",
                    "icon": "🛡️"
                }
            ],
            
            "🗣️ Voice Commands": [
                {
                    "capability": "System Control",
                    "description": "Control your computer with voice",
                    "example": "Say: 'Open Chrome', 'Close app', 'Take screenshot'",
                    "icon": "💻"
                },
                {
                    "capability": "Application Launch",
                    "description": "Launch any application hands-free",
                    "example": "Say: 'Open Notepad', 'Launch calculator'",
                    "icon": "🚀"
                },
                {
                    "capability": "Web Search",
                    "description": "Search the web instantly",
                    "example": "Say: 'Search for...' or 'Google...'",
                    "icon": "🔍"
                },
                {
                    "capability": "Volume Control",
                    "description": "Adjust system volume",
                    "example": "Say: 'Volume up', 'Mute', 'Set volume to 50'",
                    "icon": "🔊"
                }
            ],
            
            "🤖 AI Conversation": [
                {
                    "capability": "Natural Chat",
                    "description": "Have natural conversations with AI",
                    "example": "Say: 'Tell me about...', 'What do you think about...'",
                    "icon": "💬"
                },
                {
                    "capability": "Question Answering",
                    "description": "Get instant answers to any question",
                    "example": "Say: 'What is...', 'How do I...', 'Why does...'",
                    "icon": "❓"
                },
                {
                    "capability": "Context Memory",
                    "description": "Remembers conversation context",
                    "example": "Continues previous topics naturally",
                    "icon": "🧠"
                }
            ],
            
            "🎯 Smart Features": [
                {
                    "capability": "Smart Routines",
                    "description": "Automated task sequences",
                    "example": "Say: 'Good morning routine' or 'Start work mode'",
                    "icon": "⚡"
                },
                {
                    "capability": "Voice Macros",
                    "description": "Custom command shortcuts",
                    "example": "Say: 'Create macro' to set up shortcuts",
                    "icon": "🎬"
                },
                {
                    "capability": "Emotion Detection",
                    "description": "Responds to your emotional state",
                    "example": "Adapts tone based on detected mood",
                    "icon": "😊"
                },
                {
                    "capability": "Voice Memory",
                    "description": "Learns your preferences over time",
                    "example": "Remembers your favorite commands and habits",
                    "icon": "💾"
                }
            ],
            
            "🎮 Entertainment": [
                {
                    "capability": "Voice Games",
                    "description": "Play interactive voice games",
                    "example": "Say: 'Play a game', 'Number guessing game'",
                    "icon": "🎮"
                },
                {
                    "capability": "Jokes & Stories",
                    "description": "Entertainment and humor",
                    "example": "Say: 'Tell me a joke', 'Tell me a story'",
                    "icon": "😄"
                },
                {
                    "capability": "Trivia",
                    "description": "Test your knowledge",
                    "example": "Say: 'Trivia time', 'Quiz me'",
                    "icon": "🧩"
                }
            ],
            
            "📊 Information": [
                {
                    "capability": "System Status",
                    "description": "Check computer performance",
                    "example": "Say: 'System status', 'How's my computer?'",
                    "icon": "📈"
                },
                {
                    "capability": "Time & Date",
                    "description": "Get current time and date",
                    "example": "Say: 'What time is it?', 'What's the date?'",
                    "icon": "🕐"
                },
                {
                    "capability": "Weather",
                    "description": "Weather information (if API configured)",
                    "example": "Say: 'What's the weather?'",
                    "icon": "🌤️"
                }
            ],
            
            "🎨 Personalization": [
                {
                    "capability": "Personality Modes",
                    "description": "Choose assistant personality",
                    "example": "Say: 'Be more friendly', 'Professional mode'",
                    "icon": "🎭"
                },
                {
                    "capability": "Voice Settings",
                    "description": "Customize voice speed and volume",
                    "example": "Say: 'Speak faster', 'Increase volume'",
                    "icon": "🎛️"
                },
                {
                    "capability": "Learning Mode",
                    "description": "Assistant learns your preferences",
                    "example": "Automatically adapts to your usage patterns",
                    "icon": "📚"
                }
            ]
        }
    
    def get_quick_start_tutorial(self) -> List[Dict[str, str]]:
        """Get quick start tutorial steps"""
        return [
            {
                "step": 1,
                "title": "🎤 Enroll Your Voice",
                "description": "Click 'Enroll Voice' and say the enrollment phrase 3 times clearly",
                "tip": "Speak naturally at normal volume in a quiet environment",
                "action": "Try it now!"
            },
            {
                "step": 2,
                "title": "🔐 Login with Voice",
                "description": "Click 'Login with Voice' and say the login phrase",
                "tip": "Your voice is your password - speak the same phrase you enrolled",
                "action": "Authenticate yourself"
            },
            {
                "step": 3,
                "title": "🗣️ Start Voice Listening",
                "description": "Click the microphone button to activate voice commands",
                "tip": "The assistant listens continuously once activated",
                "action": "Enable listening"
            },
            {
                "step": 4,
                "title": "💬 Try a Command",
                "description": "Say a simple command like 'What time is it?' or 'Tell me a joke'",
                "tip": "Speak clearly and wait for the response",
                "action": "Test your first command"
            },
            {
                "step": 5,
                "title": "🎯 Explore Features",
                "description": "Try: 'What can you do?' to see all capabilities",
                "tip": "Ask the assistant for help anytime!",
                "action": "Discover more"
            }
        ]
    
    def get_contextual_help(self, context: str) -> str:
        """Get contextual help based on current screen/action"""
        help_texts = {
            "enrollment": """
🎤 **Voice Enrollment Help**

**What to do:**
1. Click the "Enroll Voice" button
2. Speak your enrollment phrase clearly 3 times
3. Wait for confirmation

**Tips for best results:**
✅ Speak naturally at normal volume
✅ Use a quiet environment
✅ Keep consistent distance from microphone
✅ Say the same phrase each time

**Example phrases:**
- "My voice is my password"
- "Open sesame secure system"
- "Authentication code alpha omega"

**Troubleshooting:**
❌ Too much noise? Find a quieter spot
❌ Not detected? Speak louder and clearer
❌ Failed? Try enrolling again with the same phrase
            """,
            
            "login": """
🔐 **Voice Login Help**

**What to do:**
1. Click "Login with Voice"
2. Say your enrolled phrase clearly
3. Wait for authentication

**Important:**
⚠️ Use the EXACT same phrase you enrolled with
⚠️ Speak at similar volume and tone
⚠️ Keep same distance from microphone

**If authentication fails:**
- Check microphone is working
- Speak more clearly
- Reduce background noise
- Try enrolling again if problems persist
            """,
            
            "voice_commands": """
🗣️ **Voice Commands Help**

**Getting Started:**
1. Click the microphone button 🎤
2. Wait for "Listening..." indicator
3. Speak your command clearly
4. Wait for response

**Popular Commands:**
- "What time is it?"
- "Tell me a joke"
- "Open Chrome"
- "What can you do?"
- "System status"
- "Play a game"

**Pro Tips:**
💡 Be specific: "Open Google Chrome" vs "Open browser"
💡 Wait for response before next command
💡 Say "help" or "what can you do?" for full list
💡 The assistant remembers conversation context
            """,
            
            "features": """
🌟 **Available Features**

**Say these to explore:**
- "What can you do?" - Full capability list
- "Help" - General assistance
- "Tutorial" - Interactive guide
- "Examples" - Command examples
- "Settings" - Personalization options

**Quick Feature Overview:**
🎮 Games: "Play a game"
🤖 AI Chat: "Let's chat"  
💻 System Control: "Open [app]"
⚡ Smart Routines: "Good morning routine"
📊 Info: "System status", "Time"
🎭 Personality: "Change personality"
            """,
            
            "troubleshooting": """
🔧 **Troubleshooting Guide**

**Microphone not working?**
✅ Check Windows microphone permissions
✅ Test microphone in Settings
✅ Try restarting the app

**Commands not recognized?**
✅ Speak more clearly and slowly
✅ Reduce background noise
✅ Check if phrase is supported: "What can you do?"
✅ Try typing the command instead

**Authentication fails?**
✅ Re-enroll with same phrase
✅ Use consistent voice tone
✅ Check microphone quality

**Performance slow?**
✅ Close other apps
✅ Check system resources
✅ GPU mode enabled for faster processing

**Need more help?**
📖 Check full documentation
💬 Ask: "How do I [action]?"
            """
        }
        
        return help_texts.get(context, "Ask me: 'What can you do?' or 'Help' for assistance!")
    
    def get_example_commands(self) -> Dict[str, List[str]]:
        """Get example commands by category"""
        return {
            "Basic Commands": [
                "What time is it?",
                "What's the date?",
                "Tell me a joke",
                "How are you?",
                "What can you do?"
            ],
            
            "System Control": [
                "Open Chrome",
                "Open Notepad",
                "Close active window",
                "Take a screenshot",
                "Volume up",
                "Mute volume"
            ],
            
            "Search & Web": [
                "Search for Python tutorials",
                "Google artificial intelligence",
                "Open YouTube",
                "Open Gmail"
            ],
            
            "AI Conversation": [
                "Tell me about quantum physics",
                "Explain machine learning",
                "What do you think about technology?",
                "Write a poem",
                "Tell me a story"
            ],
            
            "Entertainment": [
                "Play a game",
                "Tell me a fun fact",
                "Number guessing game",
                "Quiz time",
                "Sing a song"
            ],
            
            "Information": [
                "System status",
                "CPU usage",
                "Memory usage",
                "Battery status (laptop)",
                "Network status"
            ],
            
            "Smart Features": [
                "Create a routine",
                "Set up a macro",
                "What's my usage today?",
                "Show my preferences",
                "Good morning routine"
            ]
        }
    
    def get_tips_and_tricks(self) -> List[Dict[str, str]]:
        """Get useful tips and tricks"""
        return [
            {
                "title": "🎯 Be Specific",
                "tip": "Instead of 'open browser', say 'open Google Chrome' for better results"
            },
            {
                "title": "💬 Context Awareness",
                "tip": "The assistant remembers your conversation! You can refer to previous topics"
            },
            {
                "title": "⚡ Quick Actions",
                "tip": "Say 'what can you do?' to discover all capabilities instantly"
            },
            {
                "title": "🎭 Personality Modes",
                "tip": "Change assistant personality: 'be more friendly' or 'professional mode'"
            },
            {
                "title": "🎮 Try Games",
                "tip": "Say 'play a game' for interactive entertainment!"
            },
            {
                "title": "🔄 Routines",
                "tip": "Set up morning routines for automated sequences of commands"
            },
            {
                "title": "📊 Track Usage",
                "tip": "Say 'show my stats' to see your usage patterns"
            },
            {
                "title": "🎤 Voice Quality",
                "tip": "For best results, use a good microphone in a quiet environment"
            },
            {
                "title": "⚡ GPU Acceleration",
                "tip": "If you have NVIDIA GPU, it's auto-enabled for 8x faster processing!"
            },
            {
                "title": "💾 Learning Mode",
                "tip": "The assistant learns your preferences automatically over time"
            }
        ]
    
    def format_help_message(self, title: str, content: str) -> str:
        """Format help message nicely"""
        border = "=" * 60
        return f"\n{border}\n✨ {title}\n{border}\n{content}\n{border}\n"


# Global instance
_interactive_guide = None


def get_interactive_guide() -> InteractiveGuide:
    """Get global interactive guide instance"""
    global _interactive_guide
    if _interactive_guide is None:
        _interactive_guide = InteractiveGuide()
    return _interactive_guide
