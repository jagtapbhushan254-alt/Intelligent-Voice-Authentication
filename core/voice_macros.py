"""
SecureX-Assist - Voice Macros and Custom Commands
User-defined voice shortcuts and multi-step automation
"""

import json
import logging
import os
import subprocess
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


@dataclass
class CommandStep:
    """Single step in a macro"""
    action_type: str  # 'command', 'wait', 'condition', 'speak'
    action_data: Dict[str, Any]
    optional: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class VoiceMacro:
    """Voice macro definition"""
    name: str
    trigger_phrases: List[str]
    description: str
    steps: List[CommandStep]
    enabled: bool = True
    created_at: str = ""
    usage_count: int = 0
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['steps'] = [step.to_dict() for step in self.steps]
        return data
    
    @staticmethod
    def from_dict(data: Dict) -> 'VoiceMacro':
        steps = [CommandStep(**step) for step in data.get('steps', [])]
        macro_data = {**data, 'steps': steps}
        return VoiceMacro(**macro_data)


class VoiceMacroManager:
    """
    Manages voice macros and custom commands
    
    Features:
    - User-defined voice shortcuts
    - Multi-step automation
    - Conditional logic
    - Variable substitution
    - Macro chaining
    """
    
    def __init__(self, storage_path: str = "voice_macros.json"):
        self.storage_path = storage_path
        self.macros: Dict[str, VoiceMacro] = {}
        self.variables: Dict[str, Any] = {}
        
        # Built-in action handlers
        self.action_handlers: Dict[str, Callable] = {
            'command': self._execute_command,
            'wait': self._execute_wait,
            'condition': self._execute_condition,
            'speak': self._execute_speak,
            'set_variable': self._execute_set_variable,
            'open_app': self._execute_open_app,
            'key_press': self._execute_key_press,
            'run_macro': self._execute_run_macro
        }
        
        # Load existing macros
        self.load_macros()
        
        logger.info(f"VoiceMacroManager initialized with {len(self.macros)} macros")
    
    def create_macro(self, name: str, trigger_phrases: List[str],
                    description: str, steps: List[CommandStep]) -> bool:
        """Create a new voice macro"""
        try:
            from datetime import datetime
            
            macro = VoiceMacro(
                name=name,
                trigger_phrases=[p.lower().strip() for p in trigger_phrases],
                description=description,
                steps=steps,
                created_at=datetime.now().isoformat()
            )
            
            self.macros[name] = macro
            self.save_macros()
            
            logger.info(f"Created macro '{name}' with {len(steps)} steps")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create macro: {e}")
            return False
    
    def match_macro(self, user_input: str) -> Optional[VoiceMacro]:
        """Find macro matching user input"""
        user_input_lower = user_input.lower().strip()
        
        for macro in self.macros.values():
            if not macro.enabled:
                continue
            
            for trigger in macro.trigger_phrases:
                # Exact match
                if trigger == user_input_lower:
                    return macro
                
                # Starts with match
                if user_input_lower.startswith(trigger):
                    return macro
                
                # Contains match (for longer phrases)
                if trigger in user_input_lower:
                    return macro
        
        return None
    
    def execute_macro(self, macro: VoiceMacro, context: Dict = None) -> Dict:
        """Execute a voice macro"""
        try:
            logger.info(f"Executing macro '{macro.name}'")
            
            # Update usage count
            macro.usage_count += 1
            self.save_macros()
            
            # Initialize execution context
            exec_context = context or {}
            exec_context['variables'] = self.variables.copy()
            
            results = []
            
            # Execute each step
            for i, step in enumerate(macro.steps):
                try:
                    logger.debug(f"Step {i+1}/{len(macro.steps)}: {step.action_type}")
                    
                    # Get handler for action type
                    handler = self.action_handlers.get(step.action_type)
                    if not handler:
                        logger.warning(f"No handler for action type: {step.action_type}")
                        if step.optional:
                            continue
                        else:
                            raise ValueError(f"Unknown action type: {step.action_type}")
                    
                    # Execute step
                    result = handler(step.action_data, exec_context)
                    results.append(result)
                    
                    # Check for failure
                    if not result.get('success', True) and not step.optional:
                        raise Exception(f"Step {i+1} failed: {result.get('error')}")
                
                except Exception as e:
                    if step.optional:
                        logger.warning(f"Optional step {i+1} failed: {e}")
                        continue
                    else:
                        logger.error(f"Macro execution failed at step {i+1}: {e}")
                        return {
                            'success': False,
                            'error': str(e),
                            'completed_steps': i,
                            'results': results
                        }
            
            logger.info(f"Macro '{macro.name}' completed successfully")
            return {
                'success': True,
                'completed_steps': len(macro.steps),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Macro execution error: {e}")
            return {
                'success': False,
                'error': str(e),
                'completed_steps': 0,
                'results': []
            }
    
    def _execute_command(self, data: Dict, context: Dict) -> Dict:
        """Execute system command"""
        try:
            command = data.get('command', '')
            
            # Variable substitution
            command = self._substitute_variables(command, context)
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _execute_wait(self, data: Dict, context: Dict) -> Dict:
        """Wait for specified duration"""
        try:
            seconds = float(data.get('seconds', 1.0))
            time.sleep(seconds)
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _execute_condition(self, data: Dict, context: Dict) -> Dict:
        """Evaluate condition"""
        try:
            condition_type = data.get('type', 'variable')
            
            if condition_type == 'variable':
                var_name = data.get('variable')
                expected = data.get('equals')
                actual = context['variables'].get(var_name)
                
                success = actual == expected
                return {'success': success, 'result': success}
            
            elif condition_type == 'time':
                # Time-based conditions
                import datetime
                current_hour = datetime.datetime.now().hour
                
                start_hour = data.get('start_hour', 0)
                end_hour = data.get('end_hour', 24)
                
                success = start_hour <= current_hour < end_hour
                return {'success': True, 'result': success}
            
            return {'success': True, 'result': False}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _execute_speak(self, data: Dict, context: Dict) -> Dict:
        """Speak text (requires TTS engine from context)"""
        try:
            text = data.get('text', '')
            text = self._substitute_variables(text, context)
            
            # Store for caller to handle TTS
            return {
                'success': True,
                'action': 'speak',
                'text': text
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _execute_set_variable(self, data: Dict, context: Dict) -> Dict:
        """Set a variable"""
        try:
            var_name = data.get('name')
            value = data.get('value')
            
            # Variable substitution in value
            if isinstance(value, str):
                value = self._substitute_variables(value, context)
            
            context['variables'][var_name] = value
            self.variables[var_name] = value
            
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _execute_open_app(self, data: Dict, context: Dict) -> Dict:
        """Open application"""
        try:
            app_name = data.get('app')
            
            # Common application paths
            app_commands = {
                'notepad': 'notepad.exe',
                'calculator': 'calc.exe',
                'chrome': 'start chrome',
                'edge': 'start msedge',
                'explorer': 'explorer.exe',
                'cmd': 'cmd.exe',
                'powershell': 'powershell.exe'
            }
            
            command = app_commands.get(app_name.lower(), f'start {app_name}')
            
            subprocess.Popen(command, shell=True)
            time.sleep(0.5)  # Allow app to start
            
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _execute_key_press(self, data: Dict, context: Dict) -> Dict:
        """Simulate key press"""
        try:
            import pyautogui
            
            keys = data.get('keys', '')
            modifiers = data.get('modifiers', [])
            
            if modifiers:
                # Press with modifiers (e.g., Ctrl+C)
                pyautogui.hotkey(*modifiers, keys)
            else:
                # Simple key press
                pyautogui.press(keys)
            
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _execute_run_macro(self, data: Dict, context: Dict) -> Dict:
        """Run another macro (chaining)"""
        try:
            macro_name = data.get('macro')
            
            if macro_name not in self.macros:
                return {'success': False, 'error': f"Macro '{macro_name}' not found"}
            
            macro = self.macros[macro_name]
            result = self.execute_macro(macro, context)
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _substitute_variables(self, text: str, context: Dict) -> str:
        """Substitute variables in text using {variable_name} syntax"""
        import re
        
        variables = context.get('variables', {})
        
        def replacer(match):
            var_name = match.group(1)
            return str(variables.get(var_name, match.group(0)))
        
        return re.sub(r'\{(\w+)\}', replacer, text)
    
    def list_macros(self, enabled_only: bool = False) -> List[Dict]:
        """List all macros"""
        macros = self.macros.values()
        
        if enabled_only:
            macros = [m for m in macros if m.enabled]
        
        return [
            {
                'name': m.name,
                'triggers': m.trigger_phrases,
                'description': m.description,
                'steps': len(m.steps),
                'enabled': m.enabled,
                'usage_count': m.usage_count
            }
            for m in macros
        ]
    
    def delete_macro(self, name: str) -> bool:
        """Delete a macro"""
        if name in self.macros:
            del self.macros[name]
            self.save_macros()
            logger.info(f"Deleted macro '{name}'")
            return True
        return False
    
    def enable_macro(self, name: str, enabled: bool = True) -> bool:
        """Enable or disable a macro"""
        if name in self.macros:
            self.macros[name].enabled = enabled
            self.save_macros()
            logger.info(f"{'Enabled' if enabled else 'Disabled'} macro '{name}'")
            return True
        return False
    
    def save_macros(self) -> bool:
        """Save macros to file"""
        try:
            data = {
                'macros': {name: macro.to_dict() for name, macro in self.macros.items()},
                'variables': self.variables
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.macros)} macros to {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save macros: {e}")
            return False
    
    def load_macros(self) -> bool:
        """Load macros from file"""
        try:
            if not os.path.exists(self.storage_path):
                logger.info("No existing macros file, creating with defaults")
                self._create_default_macros()
                return True
            
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load macros
            macros_data = data.get('macros', {})
            self.macros = {
                name: VoiceMacro.from_dict(macro_data)
                for name, macro_data in macros_data.items()
            }
            
            # Load variables
            self.variables = data.get('variables', {})
            
            logger.info(f"Loaded {len(self.macros)} macros from {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load macros: {e}")
            return False
    
    def _create_default_macros(self) -> None:
        """Create default example macros"""
        # Example 1: Goodnight routine
        self.create_macro(
            name="goodnight",
            trigger_phrases=["goodnight", "good night", "sleep mode"],
            description="Close apps and lock PC",
            steps=[
                CommandStep('speak', {'text': 'Preparing goodnight routine'}),
                CommandStep('command', {'command': 'taskkill /F /IM chrome.exe'}, optional=True),
                CommandStep('command', {'command': 'taskkill /F /IM spotify.exe'}, optional=True),
                CommandStep('wait', {'seconds': 1.0}),
                CommandStep('speak', {'text': 'Locking computer. Goodnight!'}),
                CommandStep('command', {'command': 'rundll32.exe user32.dll,LockWorkStation'})
            ]
        )
        
        # Example 2: Morning routine
        self.create_macro(
            name="morning",
            trigger_phrases=["good morning", "start day"],
            description="Open morning apps",
            steps=[
                CommandStep('speak', {'text': 'Good morning! Starting your day'}),
                CommandStep('open_app', {'app': 'chrome'}),
                CommandStep('wait', {'seconds': 2.0}),
                CommandStep('open_app', {'app': 'notepad'}),
                CommandStep('speak', {'text': 'Your morning apps are ready'})
            ]
        )
        
        # Example 3: Screenshot
        self.create_macro(
            name="screenshot",
            trigger_phrases=["take screenshot", "capture screen"],
            description="Take a screenshot",
            steps=[
                CommandStep('speak', {'text': 'Taking screenshot'}),
                CommandStep('key_press', {'keys': 'printscreen'}),
                CommandStep('speak', {'text': 'Screenshot captured'})
            ]
        )
        
        self.save_macros()


# Global macro manager
_macro_manager: Optional[VoiceMacroManager] = None


def get_macro_manager(storage_path: str = "voice_macros.json") -> VoiceMacroManager:
    """Get or create global macro manager"""
    global _macro_manager
    
    if _macro_manager is None:
        _macro_manager = VoiceMacroManager(storage_path)
    
    return _macro_manager


if __name__ == "__main__":
    # Test macro manager
    manager = VoiceMacroManager("test_macros.json")
    
    # List macros
    print("Available macros:")
    for macro in manager.list_macros():
        print(f"  - {macro['name']}: {macro['description']}")
    
    # Test macro matching
    macro = manager.match_macro("good morning")
    if macro:
        print(f"\nMatched macro: {macro.name}")
        
        # Execute macro
        result = manager.execute_macro(macro)
        print(f"Execution result: {result}")
