"""
SecureX-Assist - Smart Routines System
Time-based and trigger-based automation routines
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
import schedule

logger = logging.getLogger(__name__)


@dataclass
class RoutineAction:
    """Single action in a routine"""
    action_type: str  # speak, command, notification, macro
    action_data: Dict[str, Any]
    delay: float = 0.0  # Delay before this action (seconds)


@dataclass
class SmartRoutine:
    """Smart routine definition"""
    name: str
    description: str
    trigger_type: str  # time, event, location, condition
    trigger_data: Dict[str, Any]
    actions: List[RoutineAction]
    enabled: bool = True
    last_executed: Optional[str] = None
    execution_count: int = 0
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['actions'] = [asdict(a) for a in self.actions]
        return data
    
    @staticmethod
    def from_dict(data: Dict) -> 'SmartRoutine':
        actions = [RoutineAction(**a) for a in data.get('actions', [])]
        routine_data = {**data, 'actions': actions}
        return SmartRoutine(**routine_data)


class SmartRoutines:
    """
    Smart Routines System - Automated time-based and event-based actions
    
    Trigger Types:
    - time: Execute at specific time (e.g., 7:00 AM)
    - interval: Execute every N minutes/hours
    - event: Execute on specific event (user_login, user_home, etc.)
    - condition: Execute when condition is met
    """
    
    def __init__(self, storage_path: str = "smart_routines.json"):
        self.storage_path = storage_path
        self.routines: Dict[str, SmartRoutine] = {}
        self.event_handlers: Dict[str, Callable] = {}
        
        # Scheduler thread
        self.scheduler_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        
        # Load existing routines
        self.load_routines()
        
        logger.info(f"SmartRoutines initialized with {len(self.routines)} routines")
    
    def create_routine(self, name: str, description: str, trigger_type: str,
                      trigger_data: Dict, actions: List[RoutineAction]) -> bool:
        """Create a new smart routine"""
        try:
            routine = SmartRoutine(
                name=name,
                description=description,
                trigger_type=trigger_type,
                trigger_data=trigger_data,
                actions=actions
            )
            
            self.routines[name] = routine
            self.save_routines()
            
            # Schedule if time-based
            if trigger_type == 'time':
                self._schedule_routine(routine)
            
            logger.info(f"Created routine '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create routine: {e}")
            return False
    
    def execute_routine(self, routine_name: str, context: Dict = None) -> Dict:
        """Execute a routine"""
        if routine_name not in self.routines:
            return {'success': False, 'error': 'Routine not found'}
        
        routine = self.routines[routine_name]
        
        if not routine.enabled:
            return {'success': False, 'error': 'Routine is disabled'}
        
        try:
            logger.info(f"Executing routine '{routine_name}'")
            
            results = []
            
            for action in routine.actions:
                # Delay if specified
                if action.delay > 0:
                    time.sleep(action.delay)
                
                # Execute action
                result = self._execute_action(action, context or {})
                results.append(result)
            
            # Update execution stats
            routine.last_executed = datetime.now().isoformat()
            routine.execution_count += 1
            self.save_routines()
            
            logger.info(f"Routine '{routine_name}' completed successfully")
            return {
                'success': True,
                'routine': routine_name,
                'actions_executed': len(results),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Routine execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _execute_action(self, action: RoutineAction, context: Dict) -> Dict:
        """Execute a single action"""
        try:
            action_type = action.action_type
            action_data = action.action_data
            
            if action_type == 'speak':
                # Return text to be spoken (caller handles TTS)
                return {
                    'success': True,
                    'type': 'speak',
                    'text': action_data.get('text', '')
                }
            
            elif action_type == 'notification':
                # Return notification data
                return {
                    'success': True,
                    'type': 'notification',
                    'title': action_data.get('title', ''),
                    'message': action_data.get('message', '')
                }
            
            elif action_type == 'command':
                # Return command to be executed
                return {
                    'success': True,
                    'type': 'command',
                    'command': action_data.get('command', '')
                }
            
            elif action_type == 'macro':
                # Return macro to be executed
                return {
                    'success': True,
                    'type': 'macro',
                    'macro_name': action_data.get('macro_name', '')
                }
            
            elif action_type == 'custom':
                # Custom action with callback
                callback = self.event_handlers.get(action_data.get('handler', ''))
                if callback:
                    return callback(action_data, context)
                return {'success': False, 'error': 'No handler found'}
            
            return {'success': False, 'error': f'Unknown action type: {action_type}'}
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _schedule_routine(self, routine: SmartRoutine) -> None:
        """Schedule a time-based routine"""
        if routine.trigger_type != 'time':
            return
        
        time_str = routine.trigger_data.get('time', '09:00')
        days = routine.trigger_data.get('days', ['monday', 'tuesday', 'wednesday', 
                                                   'thursday', 'friday', 'saturday', 'sunday'])
        
        # Create schedule for each day
        for day in days:
            day_lower = day.lower()
            if day_lower == 'monday':
                schedule.every().monday.at(time_str).do(self.execute_routine, routine.name)
            elif day_lower == 'tuesday':
                schedule.every().tuesday.at(time_str).do(self.execute_routine, routine.name)
            elif day_lower == 'wednesday':
                schedule.every().wednesday.at(time_str).do(self.execute_routine, routine.name)
            elif day_lower == 'thursday':
                schedule.every().thursday.at(time_str).do(self.execute_routine, routine.name)
            elif day_lower == 'friday':
                schedule.every().friday.at(time_str).do(self.execute_routine, routine.name)
            elif day_lower == 'saturday':
                schedule.every().saturday.at(time_str).do(self.execute_routine, routine.name)
            elif day_lower == 'sunday':
                schedule.every().sunday.at(time_str).do(self.execute_routine, routine.name)
        
        logger.info(f"Scheduled routine '{routine.name}' for {time_str} on {', '.join(days)}")
    
    def start_scheduler(self) -> None:
        """Start the background scheduler"""
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Routine scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop the background scheduler"""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=2.0)
        
        logger.info("Routine scheduler stopped")
    
    def _scheduler_loop(self) -> None:
        """Background scheduler loop"""
        while self.scheduler_running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(5)
    
    def trigger_event(self, event_name: str, context: Dict = None) -> List[Dict]:
        """Trigger event-based routines"""
        results = []
        
        for routine in self.routines.values():
            if not routine.enabled:
                continue
            
            if routine.trigger_type == 'event':
                if routine.trigger_data.get('event') == event_name:
                    result = self.execute_routine(routine.name, context)
                    results.append(result)
        
        return results
    
    def register_event_handler(self, handler_name: str, callback: Callable) -> None:
        """Register custom event handler"""
        self.event_handlers[handler_name] = callback
        logger.info(f"Registered event handler: {handler_name}")
    
    def list_routines(self, enabled_only: bool = False) -> List[Dict]:
        """List all routines"""
        routines = self.routines.values()
        
        if enabled_only:
            routines = [r for r in routines if r.enabled]
        
        return [
            {
                'name': r.name,
                'description': r.description,
                'trigger_type': r.trigger_type,
                'enabled': r.enabled,
                'last_executed': r.last_executed,
                'execution_count': r.execution_count
            }
            for r in routines
        ]
    
    def enable_routine(self, name: str, enabled: bool = True) -> bool:
        """Enable or disable a routine"""
        if name in self.routines:
            self.routines[name].enabled = enabled
            self.save_routines()
            logger.info(f"{'Enabled' if enabled else 'Disabled'} routine '{name}'")
            return True
        return False
    
    def delete_routine(self, name: str) -> bool:
        """Delete a routine"""
        if name in self.routines:
            del self.routines[name]
            self.save_routines()
            logger.info(f"Deleted routine '{name}'")
            return True
        return False
    
    def save_routines(self) -> bool:
        """Save routines to file"""
        try:
            data = {
                'routines': {name: routine.to_dict() for name, routine in self.routines.items()}
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.routines)} routines")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save routines: {e}")
            return False
    
    def load_routines(self) -> bool:
        """Load routines from file"""
        try:
            if not os.path.exists(self.storage_path):
                logger.info("No existing routines file, creating defaults")
                self._create_default_routines()
                return True
            
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            routines_data = data.get('routines', {})
            self.routines = {
                name: SmartRoutine.from_dict(routine_data)
                for name, routine_data in routines_data.items()
            }
            
            # Schedule time-based routines
            for routine in self.routines.values():
                if routine.trigger_type == 'time' and routine.enabled:
                    self._schedule_routine(routine)
            
            logger.info(f"Loaded {len(self.routines)} routines")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load routines: {e}")
            return False
    
    def _create_default_routines(self) -> None:
        """Create default example routines"""
        # Morning routine
        self.create_routine(
            name="morning_briefing",
            description="Morning briefing at 7 AM",
            trigger_type="time",
            trigger_data={'time': '07:00', 'days': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']},
            actions=[
                RoutineAction('speak', {'text': 'Good morning! Starting your day.'}),
                RoutineAction('notification', {'title': 'Morning', 'message': 'Time to start your day!'}, delay=1.0),
                RoutineAction('speak', {'text': 'Have a great day!'}, delay=2.0)
            ]
        )
        
        # Evening routine
        self.create_routine(
            name="evening_routine",
            description="Evening reminder at 9 PM",
            trigger_type="time",
            trigger_data={'time': '21:00', 'days': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']},
            actions=[
                RoutineAction('speak', {'text': 'Good evening! Time to wind down.'}),
                RoutineAction('notification', {'title': 'Evening', 'message': 'Consider preparing for bed.'})
            ]
        )
        
        # User login event
        self.create_routine(
            name="welcome_back",
            description="Welcome message on login",
            trigger_type="event",
            trigger_data={'event': 'user_login'},
            actions=[
                RoutineAction('speak', {'text': 'Welcome back! How can I help you today?'})
            ]
        )
        
        self.save_routines()


# Global instance
_smart_routines: Optional[SmartRoutines] = None


def get_smart_routines(storage_path: str = "smart_routines.json") -> SmartRoutines:
    """Get or create global smart routines instance"""
    global _smart_routines
    
    if _smart_routines is None:
        _smart_routines = SmartRoutines(storage_path)
    
    return _smart_routines


if __name__ == "__main__":
    # Test smart routines
    routines = SmartRoutines("test_routines.json")
    
    print("Available routines:")
    for routine in routines.list_routines():
        print(f"  • {routine['name']}: {routine['description']}")
    
    print("\nTesting event trigger:")
    results = routines.trigger_event('user_login')
    for result in results:
        print(f"  Result: {result}")
    
    print("\nSmart routines test complete!")
