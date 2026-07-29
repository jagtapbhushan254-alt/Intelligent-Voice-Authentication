"""
SecureX-Assist - Voice Analytics and Statistics
Track usage patterns, performance metrics, and security audit trails
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import threading

logger = logging.getLogger(__name__)


@dataclass
class VoiceEvent:
    """Single voice interaction event"""
    timestamp: str
    event_type: str  # 'command', 'verification', 'enrollment', 'error'
    user_id: Optional[str]
    command: Optional[str]
    success: bool
    response_time: float  # in seconds
    confidence: Optional[float]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class UserStatistics:
    """Statistics for a single user"""
    user_id: str
    total_commands: int
    successful_commands: int
    failed_commands: int
    avg_response_time: float
    avg_confidence: float
    most_used_commands: List[tuple]  # [(command, count), ...]
    first_seen: str
    last_seen: str
    verification_attempts: int
    verification_failures: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


class VoiceAnalytics:
    """
    Voice Analytics and Statistics System
    
    Features:
    - Real-time usage tracking
    - Performance metrics
    - Security audit trails
    - Command frequency analysis
    - User behavior patterns
    - Trend analysis
    """
    
    def __init__(self, storage_path: str = "voice_analytics.json", max_events: int = 10000):
        self.storage_path = storage_path
        self.max_events = max_events
        
        self.events: List[VoiceEvent] = []
        self.lock = threading.Lock()
        
        # Cached statistics
        self._stats_cache: Dict[str, Any] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)
        
        # Load existing data
        self.load_analytics()
        
        logger.info(f"VoiceAnalytics initialized with {len(self.events)} events")
    
    def log_event(self, event_type: str, user_id: Optional[str] = None,
                  command: Optional[str] = None, success: bool = True,
                  response_time: float = 0.0, confidence: Optional[float] = None,
                  metadata: Dict = None) -> None:
        """Log a voice interaction event"""
        try:
            with self.lock:
                event = VoiceEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type=event_type,
                    user_id=user_id,
                    command=command,
                    success=success,
                    response_time=response_time,
                    confidence=confidence,
                    metadata=metadata or {}
                )
                
                self.events.append(event)
                
                # Limit event history
                if len(self.events) > self.max_events:
                    self.events = self.events[-self.max_events:]
                
                # Invalidate cache
                self._cache_time = None
                
                # Auto-save periodically
                if len(self.events) % 100 == 0:
                    self.save_analytics()
            
            logger.debug(f"Logged {event_type} event for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
    
    def get_statistics(self, user_id: Optional[str] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       force_refresh: bool = False) -> Dict:
        """
        Get comprehensive statistics
        
        Args:
            user_id: Filter by specific user (None = all users)
            start_date: Filter events after this date
            end_date: Filter events before this date
            force_refresh: Force recalculation of cached stats
        """
        try:
            # Check cache
            cache_key = f"{user_id}_{start_date}_{end_date}"
            if not force_refresh and self._is_cache_valid():
                cached = self._stats_cache.get(cache_key)
                if cached:
                    return cached
            
            with self.lock:
                # Filter events
                filtered_events = self._filter_events(
                    self.events, user_id, start_date, end_date
                )
                
                if not filtered_events:
                    return {'total_events': 0}
                
                # Calculate statistics
                stats = self._calculate_statistics(filtered_events)
                
                # Cache result
                self._stats_cache[cache_key] = stats
                self._cache_time = datetime.now()
                
                return stats
        
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {'error': str(e)}
    
    def get_user_statistics(self, user_id: str) -> Optional[UserStatistics]:
        """Get detailed statistics for a specific user"""
        try:
            user_events = [e for e in self.events if e.user_id == user_id]
            
            if not user_events:
                return None
            
            # Calculate metrics
            total_commands = len([e for e in user_events if e.event_type == 'command'])
            successful_commands = len([e for e in user_events 
                                      if e.event_type == 'command' and e.success])
            failed_commands = total_commands - successful_commands
            
            response_times = [e.response_time for e in user_events if e.response_time > 0]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
            
            confidences = [e.confidence for e in user_events if e.confidence is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Most used commands
            commands = [e.command for e in user_events 
                       if e.event_type == 'command' and e.command]
            command_counter = Counter(commands)
            most_used = command_counter.most_common(10)
            
            # Verification stats
            verification_events = [e for e in user_events if e.event_type == 'verification']
            verification_attempts = len(verification_events)
            verification_failures = len([e for e in verification_events if not e.success])
            
            # Timestamps
            timestamps = [datetime.fromisoformat(e.timestamp) for e in user_events]
            first_seen = min(timestamps).isoformat()
            last_seen = max(timestamps).isoformat()
            
            return UserStatistics(
                user_id=user_id,
                total_commands=total_commands,
                successful_commands=successful_commands,
                failed_commands=failed_commands,
                avg_response_time=round(avg_response_time, 3),
                avg_confidence=round(avg_confidence, 4),
                most_used_commands=most_used,
                first_seen=first_seen,
                last_seen=last_seen,
                verification_attempts=verification_attempts,
                verification_failures=verification_failures
            )
            
        except Exception as e:
            logger.error(f"Failed to get user statistics: {e}")
            return None
    
    def get_trend_analysis(self, days: int = 7, metric: str = 'commands') -> Dict:
        """
        Analyze trends over time
        
        Args:
            days: Number of days to analyze
            metric: 'commands', 'verifications', 'errors', 'response_time'
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            # Group events by date
            daily_data = defaultdict(list)
            
            for event in self.events:
                event_date = datetime.fromisoformat(event.timestamp)
                if event_date >= start_date:
                    date_key = event_date.date().isoformat()
                    daily_data[date_key].append(event)
            
            # Calculate daily metrics
            trend_data = []
            
            for date_str in sorted(daily_data.keys()):
                events = daily_data[date_str]
                
                if metric == 'commands':
                    value = len([e for e in events if e.event_type == 'command'])
                elif metric == 'verifications':
                    value = len([e for e in events if e.event_type == 'verification'])
                elif metric == 'errors':
                    value = len([e for e in events if not e.success])
                elif metric == 'response_time':
                    times = [e.response_time for e in events if e.response_time > 0]
                    value = sum(times) / len(times) if times else 0.0
                else:
                    value = len(events)
                
                trend_data.append({
                    'date': date_str,
                    'value': round(value, 3) if isinstance(value, float) else value
                })
            
            return {
                'metric': metric,
                'days': days,
                'data': trend_data,
                'total_events': sum(len(events) for events in daily_data.values())
            }
            
        except Exception as e:
            logger.error(f"Failed to get trend analysis: {e}")
            return {'error': str(e)}
    
    def get_command_frequency(self, top_n: int = 20) -> List[Dict]:
        """Get most frequently used commands"""
        try:
            commands = [e.command for e in self.events 
                       if e.event_type == 'command' and e.command]
            
            counter = Counter(commands)
            
            return [
                {'command': cmd, 'count': count, 'percentage': round(count / len(commands) * 100, 2)}
                for cmd, count in counter.most_common(top_n)
            ]
            
        except Exception as e:
            logger.error(f"Failed to get command frequency: {e}")
            return []
    
    def get_security_audit(self, user_id: Optional[str] = None,
                          event_type: Optional[str] = None,
                          failed_only: bool = False) -> List[Dict]:
        """Get security audit trail"""
        try:
            # Filter events
            filtered = self.events
            
            if user_id:
                filtered = [e for e in filtered if e.user_id == user_id]
            
            if event_type:
                filtered = [e for e in filtered if e.event_type == event_type]
            
            if failed_only:
                filtered = [e for e in filtered if not e.success]
            
            # Format audit entries
            audit_trail = []
            
            for event in filtered[-1000:]:  # Last 1000 events
                entry = {
                    'timestamp': event.timestamp,
                    'event_type': event.event_type,
                    'user_id': event.user_id or 'unknown',
                    'success': event.success,
                    'details': {
                        'command': event.command,
                        'confidence': event.confidence,
                        'response_time': event.response_time,
                        **event.metadata
                    }
                }
                audit_trail.append(entry)
            
            return audit_trail
            
        except Exception as e:
            logger.error(f"Failed to get security audit: {e}")
            return []
    
    def get_performance_summary(self) -> Dict:
        """Get performance metrics summary"""
        try:
            command_events = [e for e in self.events if e.event_type == 'command']
            
            if not command_events:
                return {'total_commands': 0}
            
            # Response times
            response_times = [e.response_time for e in command_events if e.response_time > 0]
            
            # Confidence scores
            confidences = [e.confidence for e in command_events if e.confidence is not None]
            
            # Success rate
            successful = len([e for e in command_events if e.success])
            success_rate = successful / len(command_events) * 100
            
            return {
                'total_commands': len(command_events),
                'successful_commands': successful,
                'failed_commands': len(command_events) - successful,
                'success_rate': round(success_rate, 2),
                'avg_response_time': round(sum(response_times) / len(response_times), 3) if response_times else 0.0,
                'min_response_time': round(min(response_times), 3) if response_times else 0.0,
                'max_response_time': round(max(response_times), 3) if response_times else 0.0,
                'avg_confidence': round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
                'timeframe': {
                    'start': min(e.timestamp for e in command_events),
                    'end': max(e.timestamp for e in command_events)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {'error': str(e)}
    
    def get_active_users(self, days: int = 7) -> List[Dict]:
        """Get list of active users in the specified timeframe"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            # Filter recent events
            recent_events = [
                e for e in self.events
                if datetime.fromisoformat(e.timestamp) >= start_date and e.user_id
            ]
            
            # Group by user
            user_activity = defaultdict(int)
            user_last_seen = {}
            
            for event in recent_events:
                user_activity[event.user_id] += 1
                
                event_time = datetime.fromisoformat(event.timestamp)
                if event.user_id not in user_last_seen or event_time > datetime.fromisoformat(user_last_seen[event.user_id]):
                    user_last_seen[event.user_id] = event.timestamp
            
            # Format results
            active_users = [
                {
                    'user_id': user_id,
                    'activity_count': count,
                    'last_seen': user_last_seen[user_id]
                }
                for user_id, count in sorted(user_activity.items(), key=lambda x: x[1], reverse=True)
            ]
            
            return active_users
            
        except Exception as e:
            logger.error(f"Failed to get active users: {e}")
            return []
    
    def _filter_events(self, events: List[VoiceEvent], user_id: Optional[str],
                      start_date: Optional[datetime], end_date: Optional[datetime]) -> List[VoiceEvent]:
        """Filter events by criteria"""
        filtered = events
        
        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]
        
        if start_date:
            filtered = [e for e in filtered 
                       if datetime.fromisoformat(e.timestamp) >= start_date]
        
        if end_date:
            filtered = [e for e in filtered 
                       if datetime.fromisoformat(e.timestamp) <= end_date]
        
        return filtered
    
    def _calculate_statistics(self, events: List[VoiceEvent]) -> Dict:
        """Calculate comprehensive statistics from events"""
        total_events = len(events)
        
        # Event type breakdown
        event_types = Counter(e.event_type for e in events)
        
        # Success rates
        successful = len([e for e in events if e.success])
        success_rate = successful / total_events * 100 if total_events > 0 else 0.0
        
        # Response times
        response_times = [e.response_time for e in events if e.response_time > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        # Confidence scores
        confidences = [e.confidence for e in events if e.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Unique users
        unique_users = len(set(e.user_id for e in events if e.user_id))
        
        # Commands
        commands = [e.command for e in events if e.command]
        unique_commands = len(set(commands))
        
        return {
            'total_events': total_events,
            'event_types': dict(event_types),
            'successful_events': successful,
            'failed_events': total_events - successful,
            'success_rate': round(success_rate, 2),
            'avg_response_time': round(avg_response_time, 3),
            'avg_confidence': round(avg_confidence, 4),
            'unique_users': unique_users,
            'unique_commands': unique_commands,
            'timeframe': {
                'start': min(e.timestamp for e in events) if events else None,
                'end': max(e.timestamp for e in events) if events else None
            }
        }
    
    def _is_cache_valid(self) -> bool:
        """Check if statistics cache is still valid"""
        if not self._cache_time:
            return False
        return datetime.now() - self._cache_time < self._cache_ttl
    
    def clear_old_events(self, days: int = 30) -> int:
        """Clear events older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self.lock:
                original_count = len(self.events)
                
                self.events = [
                    e for e in self.events
                    if datetime.fromisoformat(e.timestamp) >= cutoff_date
                ]
                
                removed_count = original_count - len(self.events)
                
                if removed_count > 0:
                    self.save_analytics()
                    logger.info(f"Cleared {removed_count} old events")
                
                return removed_count
        
        except Exception as e:
            logger.error(f"Failed to clear old events: {e}")
            return 0
    
    def export_report(self, format: str = 'json', output_path: Optional[str] = None) -> str:
        """
        Export analytics report
        
        Args:
            format: 'json' or 'csv'
            output_path: Optional custom output path
        """
        try:
            if not output_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"voice_analytics_report_{timestamp}.{format}"
            
            if format == 'json':
                report = {
                    'generated_at': datetime.now().isoformat(),
                    'summary': self.get_performance_summary(),
                    'statistics': self.get_statistics(),
                    'command_frequency': self.get_command_frequency(),
                    'active_users': self.get_active_users(),
                    'trend_analysis': self.get_trend_analysis()
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2)
            
            elif format == 'csv':
                import csv
                
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=['timestamp', 'event_type', 'user_id', 'command', 
                                   'success', 'response_time', 'confidence']
                    )
                    writer.writeheader()
                    
                    for event in self.events:
                        writer.writerow({
                            'timestamp': event.timestamp,
                            'event_type': event.event_type,
                            'user_id': event.user_id or '',
                            'command': event.command or '',
                            'success': event.success,
                            'response_time': event.response_time,
                            'confidence': event.confidence or ''
                        })
            
            logger.info(f"Exported analytics report to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            return ""
    
    def save_analytics(self) -> bool:
        """Save analytics data to file"""
        try:
            data = {
                'events': [e.to_dict() for e in self.events],
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.events)} events to {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save analytics: {e}")
            return False
    
    def load_analytics(self) -> bool:
        """Load analytics data from file"""
        try:
            if not os.path.exists(self.storage_path):
                logger.info("No existing analytics file")
                return True
            
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            events_data = data.get('events', [])
            self.events = [VoiceEvent(**e) for e in events_data]
            
            logger.info(f"Loaded {len(self.events)} events from {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load analytics: {e}")
            return False


# Global analytics instance
_analytics: Optional[VoiceAnalytics] = None


def get_analytics(storage_path: str = "voice_analytics.json") -> VoiceAnalytics:
    """Get or create global analytics instance"""
    global _analytics
    
    if _analytics is None:
        _analytics = VoiceAnalytics(storage_path)
    
    return _analytics


if __name__ == "__main__":
    # Test analytics
    analytics = VoiceAnalytics("test_analytics.json")
    
    # Log some test events
    analytics.log_event('command', 'user123', 'what is the weather', True, 0.5, 0.95)
    analytics.log_event('verification', 'user123', None, True, 0.3, 0.98)
    analytics.log_event('command', 'user456', 'open calculator', True, 0.4, 0.92)
    
    # Get statistics
    stats = analytics.get_statistics()
    print("Statistics:", json.dumps(stats, indent=2))
    
    # Get user statistics
    user_stats = analytics.get_user_statistics('user123')
    if user_stats:
        print(f"\nUser statistics: {user_stats.to_dict()}")
    
    # Performance summary
    perf = analytics.get_performance_summary()
    print("\nPerformance:", json.dumps(perf, indent=2))
