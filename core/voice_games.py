"""
SecureX-Assist - Interactive Voice Games
Fun voice-controlled games and activities
"""

import random
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class GameState:
    """Current state of a game"""
    game_name: str
    user_id: str
    state_data: Dict[str, Any]
    turn_count: int
    score: int
    
    def to_dict(self) -> Dict:
        return {
            'game_name': self.game_name,
            'user_id': self.user_id,
            'state_data': self.state_data,
            'turn_count': self.turn_count,
            'score': self.score
        }


class BaseGame:
    """Base class for voice games"""
    
    def __init__(self, game_name: str):
        self.game_name = game_name
        self.active_games: Dict[str, GameState] = {}
    
    def start(self, user_id: str) -> str:
        """Start a new game"""
        raise NotImplementedError
    
    def process_input(self, user_id: str, user_input: str) -> Tuple[str, bool]:
        """
        Process user input
        Returns: (response, game_over)
        """
        raise NotImplementedError
    
    def end(self, user_id: str) -> str:
        """End the game"""
        if user_id in self.active_games:
            state = self.active_games[user_id]
            del self.active_games[user_id]
            return f"Game over! Final score: {state.score}"
        return "No active game to end."


class TwentyQuestionsGame(BaseGame):
    """AI guesses what you're thinking in 20 questions"""
    
    def __init__(self):
        super().__init__("Twenty Questions")
        
        # Knowledge base of objects with properties
        self.objects = {
            'dog': {'animal': True, 'pet': True, 'can_fly': False, 'has_fur': True},
            'cat': {'animal': True, 'pet': True, 'can_fly': False, 'has_fur': True},
            'bird': {'animal': True, 'pet': True, 'can_fly': True, 'has_fur': False},
            'car': {'animal': False, 'vehicle': True, 'has_wheels': True},
            'computer': {'animal': False, 'electronic': True, 'useful': True},
            'phone': {'animal': False, 'electronic': True, 'portable': True},
            'tree': {'animal': False, 'plant': True, 'alive': True},
            'book': {'animal': False, 'readable': True, 'portable': True},
        }
        
        self.questions = [
            ("Is it an animal?", "animal"),
            ("Is it a pet?", "pet"),
            ("Can it fly?", "can_fly"),
            ("Does it have fur?", "has_fur"),
            ("Is it a vehicle?", "vehicle"),
            ("Is it electronic?", "electronic"),
            ("Is it portable?", "portable"),
            ("Is it useful?", "useful"),
            ("Does it have wheels?", "has_wheels"),
            ("Is it a plant?", "plant")
        ]
    
    def start(self, user_id: str) -> str:
        state = GameState(
            game_name=self.game_name,
            user_id=user_id,
            state_data={
                'question_count': 0,
                'known_properties': {},
                'candidates': list(self.objects.keys())
            },
            turn_count=0,
            score=0
        )
        self.active_games[user_id] = state
        
        return ("Welcome to 20 Questions! Think of something and I'll try to guess it in 20 questions. "
                "Answer with yes, no, or maybe. Ready? " + self._ask_next_question(user_id))
    
    def process_input(self, user_id: str, user_input: str) -> Tuple[str, bool]:
        if user_id not in self.active_games:
            return "No active game. Say 'start twenty questions' to begin.", True
        
        state = self.active_games[user_id]
        user_input_lower = user_input.lower().strip()
        
        # Parse answer
        if 'yes' in user_input_lower or 'yeah' in user_input_lower or 'yep' in user_input_lower:
            answer = True
        elif 'no' in user_input_lower or 'nope' in user_input_lower:
            answer = False
        else:
            return "Please answer yes or no.", False
        
        # Update known properties
        if state.turn_count < len(self.questions):
            question, property_key = self.questions[state.turn_count]
            state.state_data['known_properties'][property_key] = answer
            
            # Filter candidates
            state.state_data['candidates'] = [
                obj for obj in state.state_data['candidates']
                if self.objects[obj].get(property_key, False) == answer
            ]
        
        state.turn_count += 1
        state.state_data['question_count'] += 1
        
        # Check if we can make a guess
        if len(state.state_data['candidates']) == 1:
            guess = state.state_data['candidates'][0]
            response = f"I think you're thinking of a {guess}! Am I right?"
            return response, True
        elif len(state.state_data['candidates']) == 0:
            return "Hmm, I'm stumped! You win this round. What were you thinking of?", True
        elif state.state_data['question_count'] >= 20:
            candidates = ', '.join(state.state_data['candidates'][:3])
            return f"I'm out of questions! Is it one of these: {candidates}?", True
        
        # Ask next question
        next_q = self._ask_next_question(user_id)
        return next_q, False
    
    def _ask_next_question(self, user_id: str) -> str:
        state = self.active_games[user_id]
        if state.turn_count < len(self.questions):
            question, _ = self.questions[state.turn_count]
            return f"Question {state.turn_count + 1}: {question}"
        return "No more questions!"


class TriviaGame(BaseGame):
    """Voice-based trivia quiz"""
    
    def __init__(self):
        super().__init__("Trivia Quiz")
        
        self.questions = [
            {
                'question': 'What is the capital of France?',
                'answers': ['paris'],
                'difficulty': 'easy'
            },
            {
                'question': 'How many planets are in our solar system?',
                'answers': ['eight', '8', 'eight planets'],
                'difficulty': 'easy'
            },
            {
                'question': 'Who painted the Mona Lisa?',
                'answers': ['leonardo da vinci', 'da vinci', 'leonardo'],
                'difficulty': 'medium'
            },
            {
                'question': 'What is the largest ocean on Earth?',
                'answers': ['pacific', 'pacific ocean'],
                'difficulty': 'easy'
            },
            {
                'question': 'In what year did World War 2 end?',
                'answers': ['1945', 'nineteen forty five'],
                'difficulty': 'medium'
            },
            {
                'question': 'What is the speed of light in meters per second?',
                'answers': ['299792458', '300 million', 'three hundred million'],
                'difficulty': 'hard'
            },
            {
                'question': 'Who wrote Romeo and Juliet?',
                'answers': ['shakespeare', 'william shakespeare'],
                'difficulty': 'easy'
            },
            {
                'question': 'What is the smallest prime number?',
                'answers': ['2', 'two'],
                'difficulty': 'medium'
            }
        ]
    
    def start(self, user_id: str) -> str:
        # Shuffle questions
        questions = self.questions.copy()
        random.shuffle(questions)
        
        state = GameState(
            game_name=self.game_name,
            user_id=user_id,
            state_data={
                'questions': questions,
                'current_index': 0,
                'correct_answers': 0
            },
            turn_count=0,
            score=0
        )
        self.active_games[user_id] = state
        
        return ("Welcome to Trivia Quiz! I'll ask you questions and you answer. Let's begin! " +
                self._ask_current_question(user_id))
    
    def process_input(self, user_id: str, user_input: str) -> Tuple[str, bool]:
        if user_id not in self.active_games:
            return "No active game. Say 'start trivia' to begin.", True
        
        state = self.active_games[user_id]
        user_answer = user_input.lower().strip()
        
        # Check answer
        current_q = state.state_data['questions'][state.state_data['current_index']]
        correct = any(ans in user_answer for ans in current_q['answers'])
        
        if correct:
            state.state_data['correct_answers'] += 1
            state.score += 10
            response = "Correct! 🎉 "
        else:
            correct_answer = current_q['answers'][0]
            response = f"Sorry, the answer was {correct_answer}. "
        
        # Move to next question
        state.state_data['current_index'] += 1
        state.turn_count += 1
        
        # Check if game over
        if state.state_data['current_index'] >= len(state.state_data['questions']):
            total = len(state.state_data['questions'])
            correct_count = state.state_data['correct_answers']
            response += f"\nGame over! You got {correct_count} out of {total} correct. Score: {state.score}"
            return response, True
        
        # Ask next question
        response += self._ask_current_question(user_id)
        return response, False
    
    def _ask_current_question(self, user_id: str) -> str:
        state = self.active_games[user_id]
        idx = state.state_data['current_index']
        total = len(state.state_data['questions'])
        question = state.state_data['questions'][idx]['question']
        return f"Question {idx + 1} of {total}: {question}"


class WordAssociationGame(BaseGame):
    """Creative word association game"""
    
    def __init__(self):
        super().__init__("Word Association")
        
        self.starter_words = [
            'sun', 'moon', 'ocean', 'mountain', 'forest', 'city', 'music',
            'dream', 'love', 'time', 'space', 'fire', 'water', 'earth'
        ]
    
    def start(self, user_id: str) -> str:
        starter_word = random.choice(self.starter_words)
        
        state = GameState(
            game_name=self.game_name,
            user_id=user_id,
            state_data={
                'word_chain': [starter_word],
                'last_word': starter_word
            },
            turn_count=0,
            score=0
        )
        self.active_games[user_id] = state
        
        return f"Welcome to Word Association! I'll say a word and you say the first word that comes to mind. My word is: {starter_word}"
    
    def process_input(self, user_id: str, user_input: str) -> Tuple[str, bool]:
        if user_id not in self.active_games:
            return "No active game. Say 'start word association' to begin.", True
        
        state = self.active_games[user_id]
        user_word = user_input.lower().strip().split()[0]  # Get first word
        
        # Add to chain
        state.state_data['word_chain'].append(user_word)
        state.turn_count += 1
        state.score += 1
        
        # Generate AI response
        ai_word = self._generate_association(user_word)
        state.state_data['word_chain'].append(ai_word)
        state.state_data['last_word'] = ai_word
        
        # Continue or end
        if state.turn_count >= 10:
            chain = ' → '.join(state.state_data['word_chain'][-10:])
            return f"Great game! Our word chain: {chain}. Score: {state.score}", True
        
        return f"{ai_word}. Your turn!", False
    
    def _generate_association(self, word: str) -> str:
        """Generate a word associated with the input"""
        # Simple associations (in real app, could use word embeddings)
        associations = {
            'sun': ['bright', 'warm', 'day', 'light'],
            'moon': ['night', 'dark', 'stars', 'tide'],
            'ocean': ['water', 'blue', 'waves', 'deep'],
            'fire': ['hot', 'red', 'burn', 'heat'],
            'water': ['drink', 'wet', 'clear', 'flow'],
            'love': ['heart', 'happy', 'care', 'together'],
            'music': ['sound', 'song', 'rhythm', 'melody'],
        }
        
        if word in associations:
            return random.choice(associations[word])
        
        # Random fallback
        return random.choice(['interesting', 'cool', 'nice', 'great'])


class StoryBuilderGame(BaseGame):
    """Collaborative storytelling"""
    
    def __init__(self):
        super().__init__("Story Builder")
        
        self.story_starters = [
            "Once upon a time in a faraway land",
            "It was a dark and stormy night when",
            "In the year 2050, humanity discovered",
            "Deep in the enchanted forest, there lived",
            "The detective walked into the room and saw"
        ]
    
    def start(self, user_id: str) -> str:
        starter = random.choice(self.story_starters)
        
        state = GameState(
            game_name=self.game_name,
            user_id=user_id,
            state_data={
                'story_parts': [starter],
                'turn': 'user'
            },
            turn_count=0,
            score=0
        )
        self.active_games[user_id] = state
        
        return f"Welcome to Story Builder! Let's create a story together. I'll start: {starter}... What happens next?"
    
    def process_input(self, user_id: str, user_input: str) -> Tuple[str, bool]:
        if user_id not in self.active_games:
            return "No active game. Say 'start story builder' to begin.", True
        
        state = self.active_games[user_id]
        
        # Add user's contribution
        state.state_data['story_parts'].append(user_input)
        state.turn_count += 1
        state.score += len(user_input.split())  # Score by word count
        
        # Check if done
        if state.turn_count >= 6:
            full_story = ' '.join(state.state_data['story_parts'])
            return f"What a great story! Here's our creation:\n\n{full_story}\n\nScore: {state.score} words!", True
        
        # Generate AI continuation
        ai_part = self._generate_story_continuation(state.state_data['story_parts'])
        state.state_data['story_parts'].append(ai_part)
        
        return f"{ai_part}... What happens next?", False
    
    def _generate_story_continuation(self, story_parts: List[str]) -> str:
        """Generate next part of story"""
        continuations = [
            "Suddenly, a mysterious figure appeared",
            "Without warning, everything changed",
            "The hero realized something important",
            "A loud noise echoed through the halls",
            "Nobody could have predicted what came next",
            "Time seemed to stand still as",
            "The truth finally revealed itself when"
        ]
        return random.choice(continuations)


class VoiceGames:
    """Manager for all voice games"""
    
    def __init__(self):
        self.games: Dict[str, BaseGame] = {
            'twenty questions': TwentyQuestionsGame(),
            '20 questions': TwentyQuestionsGame(),
            'trivia': TriviaGame(),
            'trivia quiz': TriviaGame(),
            'word association': WordAssociationGame(),
            'story builder': StoryBuilderGame(),
            'story': StoryBuilderGame()
        }
        
        logger.info(f"VoiceGames initialized with {len(set(self.games.values()))} games")
    
    def start_game(self, game_name: str, user_id: str) -> Optional[str]:
        """Start a game"""
        game_name_lower = game_name.lower().strip()
        
        if game_name_lower in self.games:
            game = self.games[game_name_lower]
            return game.start(user_id)
        
        return None
    
    def process_game_input(self, user_id: str, user_input: str) -> Optional[Tuple[str, bool]]:
        """Process input for active game"""
        # Find active game for user
        for game in set(self.games.values()):
            if user_id in game.active_games:
                return game.process_input(user_id, user_input)
        
        return None
    
    def end_game(self, user_id: str) -> Optional[str]:
        """End active game"""
        for game in set(self.games.values()):
            if user_id in game.active_games:
                return game.end(user_id)
        
        return None
    
    def get_active_game(self, user_id: str) -> Optional[str]:
        """Get name of active game for user"""
        for game in set(self.games.values()):
            if user_id in game.active_games:
                return game.game_name
        return None
    
    def list_games(self) -> List[str]:
        """List available games"""
        return list(set(game.game_name for game in self.games.values()))


# Global instance
_voice_games: Optional[VoiceGames] = None


def get_voice_games() -> VoiceGames:
    """Get or create global voice games instance"""
    global _voice_games
    
    if _voice_games is None:
        _voice_games = VoiceGames()
    
    return _voice_games


if __name__ == "__main__":
    # Test voice games
    games = VoiceGames()
    
    print("Available games:")
    for game in games.list_games():
        print(f"  • {game}")
    
    # Test trivia
    print("\n--- Testing Trivia ---")
    response = games.start_game('trivia', 'test_user')
    print(response)
    
    print("\nVoice games test complete!")
