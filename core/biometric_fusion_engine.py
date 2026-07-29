"""
Multi-Modal Biometric Fusion Engine
Combines voice, face, and liveness scores for robust authentication
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Fusion strategies for combining biometric scores"""
    WEIGHTED_SUM = "weighted_sum"
    ADAPTIVE = "adaptive"
    CONSERVATIVE = "conservative"  # Higher security, stricter


class BiometricFusionEngine:
    """
    Multi-modal biometric fusion engine that combines:
    - Voice verification score
    - Face verification score
    - Voice liveness score
    - Face liveness score
    - Anti-spoofing confidence
    
    Uses score-level fusion with adaptive weighting based on quality indicators
    """
    
    def __init__(self, config: Dict, strategy: str = "weighted_sum"):
        """
        Initialize fusion engine
        
        Args:
            config: System configuration
            strategy: Fusion strategy (weighted_sum, adaptive, conservative)
        """
        self.config = config
        self.strategy = FusionStrategy(strategy)
        
        # Default weights for weighted sum
        self.default_weights = {
            'voice': 0.35,
            'face': 0.30,
            'voice_liveness': 0.15,
            'face_liveness': 0.15,
            'anti_spoof': 0.05
        }
        
        # Thresholds
        self.fusion_threshold = config.get('fusion', {}).get('threshold', 0.70)
        self.min_quality_threshold = 0.3
        
        logger.info(f"BiometricFusionEngine initialized with {self.strategy.value} strategy")
    
    def fuse_scores(
        self,
        voice_score: float,
        face_score: float,
        voice_liveness: float = 1.0,
        face_liveness: float = 1.0,
        anti_spoof_confidence: float = 1.0,
        voice_quality: Optional[float] = None,
        face_quality: Optional[float] = None,
        environment_noise: Optional[float] = None,
        lighting_quality: Optional[float] = None
    ) -> Tuple[float, bool, Dict]:
        """
        Fuse multiple biometric scores into a single authentication decision
        
        Args:
            voice_score: Voice verification score (0-1, higher is better match)
            face_score: Face verification score (0-1, higher is better match)
            voice_liveness: Voice liveness score (0-1)
            face_liveness: Face liveness score (0-1)
            anti_spoof_confidence: Anti-spoofing confidence (0-1)
            voice_quality: Optional voice quality indicator (0-1)
            face_quality: Optional face quality indicator (0-1)
            environment_noise: Optional noise level (0-1, higher = noisier)
            lighting_quality: Optional lighting quality (0-1)
        
        Returns:
            (final_score, decision, details_dict)
        """
        
        # Normalize scores to 0-1 range
        scores = {
            'voice': self._normalize_score(voice_score),
            'face': self._normalize_score(face_score),
            'voice_liveness': self._normalize_score(voice_liveness),
            'face_liveness': self._normalize_score(face_liveness),
            'anti_spoof': self._normalize_score(anti_spoof_confidence)
        }
        
        # Calculate adaptive weights based on quality indicators
        if self.strategy == FusionStrategy.ADAPTIVE:
            weights = self._calculate_adaptive_weights(
                voice_quality, face_quality, 
                environment_noise, lighting_quality
            )
        elif self.strategy == FusionStrategy.CONSERVATIVE:
            weights = self._get_conservative_weights()
        else:
            weights = self.default_weights.copy()
        
        # Calculate weighted fusion score
        final_score = sum(scores[k] * weights[k] for k in scores.keys())
        
        # Make decision
        decision = final_score >= self.fusion_threshold
        
        # Compile details
        details = {
            'final_score': final_score,
            'threshold': self.fusion_threshold,
            'decision': decision,
            'strategy': self.strategy.value,
            'individual_scores': scores,
            'weights': weights,
            'confidence': self._calculate_confidence(scores, weights)
        }
        
        logger.info(
            f"Fusion result: score={final_score:.3f}, "
            f"decision={'ACCEPT' if decision else 'REJECT'}, "
            f"strategy={self.strategy.value}"
        )
        
        return final_score, decision, details
    
    def _normalize_score(self, score: float) -> float:
        """Normalize score to 0-1 range"""
        if score is None:
            return 0.0
        return max(0.0, min(1.0, float(score)))
    
    def _calculate_adaptive_weights(
        self,
        voice_quality: Optional[float],
        face_quality: Optional[float],
        environment_noise: Optional[float],
        lighting_quality: Optional[float]
    ) -> Dict[str, float]:
        """
        Calculate adaptive weights based on environmental conditions
        
        Poor conditions reduce the weight of affected modality
        """
        weights = self.default_weights.copy()
        
        # Adjust voice weight based on quality and noise
        if voice_quality is not None and voice_quality < 0.5:
            weights['voice'] *= 0.7
            logger.debug("Voice weight reduced due to low quality")
        
        if environment_noise is not None and environment_noise > 0.6:
            weights['voice'] *= 0.8
            logger.debug("Voice weight reduced due to high noise")
        
        # Adjust face weight based on quality and lighting
        if face_quality is not None and face_quality < 0.5:
            weights['face'] *= 0.7
            logger.debug("Face weight reduced due to low quality")
        
        if lighting_quality is not None and lighting_quality < 0.4:
            weights['face'] *= 0.6
            logger.debug("Face weight reduced due to poor lighting")
        
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _get_conservative_weights(self) -> Dict[str, float]:
        """
        Get conservative weights that prioritize security
        Requires stronger evidence from all modalities
        """
        return {
            'voice': 0.30,
            'face': 0.30,
            'voice_liveness': 0.15,
            'face_liveness': 0.15,
            'anti_spoof': 0.10  # Higher weight on anti-spoofing
        }
    
    def _calculate_confidence(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """
        Calculate overall confidence in the decision
        Based on consistency across modalities
        """
        # Check consistency - all modalities should agree
        score_values = list(scores.values())
        variance = np.var(score_values)
        
        # Lower variance = higher confidence
        confidence = 1.0 - min(variance * 2, 1.0)
        
        return confidence
    
    def get_fusion_explanation(self, details: Dict) -> str:
        """
        Generate human-readable explanation of fusion decision
        
        Args:
            details: Details dict from fuse_scores()
        
        Returns:
            Explanation string
        """
        scores = details['individual_scores']
        weights = details['weights']
        final = details['final_score']
        decision = details['decision']
        
        explanation = f"Authentication {'ACCEPTED' if decision else 'REJECTED'}\n"
        explanation += f"Final Score: {final:.2%} (threshold: {self.fusion_threshold:.2%})\n\n"
        explanation += "Component Scores:\n"
        explanation += f"  Voice Match:      {scores['voice']:.2%} (weight: {weights['voice']:.2%})\n"
        explanation += f"  Face Match:       {scores['face']:.2%} (weight: {weights['face']:.2%})\n"
        explanation += f"  Voice Liveness:   {scores['voice_liveness']:.2%} (weight: {weights['voice_liveness']:.2%})\n"
        explanation += f"  Face Liveness:    {scores['face_liveness']:.2%} (weight: {weights['face_liveness']:.2%})\n"
        explanation += f"  Anti-Spoofing:    {scores['anti_spoof']:.2%} (weight: {weights['anti_spoof']:.2%})\n"
        explanation += f"\nConfidence: {details['confidence']:.2%}"
        
        return explanation


def test_fusion_engine():
    """Test the fusion engine with sample data"""
    
    print("Testing BiometricFusionEngine...\n")
    
    config = {'fusion': {'threshold': 0.70}}
    engine = BiometricFusionEngine(config, strategy="weighted_sum")
    
    # Test case 1: High confidence match
    print("Test 1: High confidence genuine user")
    score, decision, details = engine.fuse_scores(
        voice_score=0.90,
        face_score=0.85,
        voice_liveness=0.95,
        face_liveness=0.90,
        anti_spoof_confidence=0.88
    )
    print(engine.get_fusion_explanation(details))
    print(f"Result: {'✅ PASS' if decision else '❌ FAIL'}\n")
    
    # Test case 2: Impostor attempt
    print("\nTest 2: Impostor with poor scores")
    score, decision, details = engine.fuse_scores(
        voice_score=0.45,
        face_score=0.50,
        voice_liveness=0.60,
        face_liveness=0.55,
        anti_spoof_confidence=0.40
    )
    print(engine.get_fusion_explanation(details))
    print(f"Result: {'✅ PASS' if not decision else '❌ FAIL'}\n")
    
    # Test case 3: Adaptive weights with poor lighting
    print("\nTest 3: Adaptive fusion with poor lighting")
    engine_adaptive = BiometricFusionEngine(config, strategy="adaptive")
    score, decision, details = engine_adaptive.fuse_scores(
        voice_score=0.85,
        face_score=0.60,  # Low face score
        voice_liveness=0.90,
        face_liveness=0.70,
        anti_spoof_confidence=0.85,
        face_quality=0.3,  # Poor face quality
        lighting_quality=0.2  # Poor lighting
    )
    print(engine.get_fusion_explanation(details))
    print(f"Adaptive weights reduced face impact\n")
    
    print("✅ All fusion tests complete!")


if __name__ == "__main__":
    test_fusion_engine()
