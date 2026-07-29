"""
SecureX-Assist - Audio Processing Pipeline
Audio capture, Voice Activity Detection, and processing

Robust, stoppable AudioRecorder using sounddevice InputStream callback with
detailed debug logging and a safe fallback. Also includes a simple
VoiceActivityDetector and AudioProcessor helpers.

Dependencies:
  - numpy
  - sounddevice
  - soundfile (optional, for save_audio)
  - scipy (optional, for filters)
  - torch (optional, for Silero VAD)
  - queue (for VAD)
"""
from pathlib import Path
import threading
import logging
from typing import Optional, Callable, Tuple, List
import queue # <-- Added for VAD
import time # <-- Added for VAD

import numpy as np

# sounddevice is used for audio I/O
import sounddevice as sd

logger = logging.getLogger("core.audio_processor")
if not logger.handlers:
    # Basic fallback handler if app hasn't configured logging
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


class AudioRecorder:
    """
    Stoppable audio recorder using sounddevice.InputStream callback.
    - record_audio(duration): blocks up to duration seconds but returns earlier
      if stop_recording() is called.
    - stop_recording(): requests the currently running record_audio to stop.
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        audio_cfg = cfg.get("audio", {}) if cfg else {}
        self.sample_rate: int = int(audio_cfg.get("sample_rate", 16000))
        self.channels: int = int(audio_cfg.get("channels", 1))
        self.blocksize: int = int(audio_cfg.get("blocksize", 1024))
        self.dtype: str = audio_cfg.get("dtype", "float32")
        self.device = audio_cfg.get("device", None)  # can be None or device id/string

        # Internal recording state
        self._frames: List[np.ndarray] = []
        self._stop_event = threading.Event()
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        
        # --- START: Added for VAD ---
        self._vad_queue: Optional[queue.Queue] = None
        # --- END: Added for VAD ---

        logger.info("AudioRecorder initialized: samplerate=%d, channels=%d, blocksize=%d, dtype=%s",
                     self.sample_rate, self.channels, self.blocksize, self.dtype)

    def list_devices(self) -> list:
        """Return available input devices (list of dicts)."""
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d.get("max_input_channels", 0) > 0]
            return input_devices
        except Exception as e:
            logger.exception("Failed to list devices: %s", e)
            return []

    # --- START: ADD record_with_vad METHOD ---
    def record_with_vad(self,
                          vad_detector,
                          min_speech_duration_ms: int = 250,  # Optimized: Keep responsive
                          max_silence_duration_ms: int = 800,  # Optimized: Faster stop (was 1000)
                          padding_duration_ms: int = 150,      # Optimized: Less padding (was 200)
                          device: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Records from the microphone until speech stops, using VAD.
        This is a blocking function that returns when speech is finished.
        """
        self._stop_event.clear()
        with self._lock:
            self._frames = []

        chosen_device = device if device is not None else self.device
        logger.info("Starting VAD recording (device=%s)...", chosen_device)

        min_speech_samples = int(self.sample_rate * min_speech_duration_ms / 1000)
        max_silence_samples = int(self.sample_rate * max_silence_duration_ms / 1000)
        padding_samples = int(self.sample_rate * padding_duration_ms / 1000)
        
        speech_started = False
        silence_samples = 0
        speech_samples = 0
        
        # We need a queue to pass frames from the callback thread to this thread
        self._vad_queue = queue.Queue()
        
        def vad_callback(indata: np.ndarray, frames: int, time_info, status):
            """This runs in a separate thread."""
            if status:
                logger.debug("VAD InputStream status: %s", status)
            if self._stop_event.is_set():
                if self._vad_queue:
                    self._vad_queue.put(None) # Signal stop
                raise sd.CallbackStop()
            if self._vad_queue:
                self._vad_queue.put(indata.copy())

        try:
            self._stream = sd.InputStream(samplerate=self.sample_rate,
                                          channels=self.channels,
                                          dtype=self.dtype,
                                          blocksize=self.blocksize,
                                          device=chosen_device,
                                          callback=vad_callback)
            
            with self._stream:
                while not self._stop_event.is_set():
                    try:
                        # --- FIX: Poll the queue so the stop_event is checked ---
                        chunk = self._vad_queue.get(timeout=0.1) 
                        if chunk is None: # Stop signal
                            break
                    except queue.Empty:
                        # No audio, loop again to check stop_event
                        continue

                    # Flatten to 1D
                    if chunk.ndim > 1 and chunk.shape[1] == 1:
                        chunk = chunk.flatten()
                    elif chunk.ndim > 1:
                        chunk = np.mean(chunk, axis=1) # Mix to mono

                    has_voice, _ = vad_detector.detect_rms(chunk)
                    
                    if has_voice:
                        if not speech_started:
                            logger.info("VAD: Speech detected")
                            speech_started = True
                        
                        with self._lock:
                            self._frames.append(chunk)
                        speech_samples += len(chunk)
                        silence_samples = 0
                    
                    elif speech_started:
                        # We are in speech, but this chunk is silent
                        with self._lock:
                            self._frames.append(chunk)
                        silence_samples += len(chunk)
                        
                        if silence_samples > max_silence_samples:
                            logger.info("VAD: Max silence reached, stopping.")
                            break
                    else:
                        # Still waiting for speech
                        pass
            
            logger.info("VAD stream stopped.")

            with self._lock:
                if not self._frames:
                    logger.info("VAD: No frames captured.")
                    return None
                
                if speech_samples < min_speech_samples:
                    logger.info("VAD: Speech too short (%d < %d samples), discarding.", speech_samples, min_speech_samples)
                    return None

                audio_np = np.concatenate(self._frames, axis=0)

            # Trim final silence if we stopped on silence
            if silence_samples > 0 and audio_np.size > silence_samples:
                audio_np = audio_np[:-silence_samples]

            # Add padding
            padding_start = np.zeros(min(padding_samples, audio_np.size), dtype=self.dtype)
            audio_np = np.concatenate([padding_start, audio_np], axis=0)

            # --- START FIX: Bypassing all processing per user request ---
            logger.debug("Bypassing audio processing. Using raw VAD audio.")
            # audio_np = AudioProcessor.normalize(audio_np)
            # --- END FIX ---
            
            logger.info("VAD Recording complete: samples=%d duration=%.2fs", audio_np.shape[0], audio_np.shape[0] / self.sample_rate)
            return audio_np

        except Exception as e:
            logger.error("VAD recording error: %s", e, exc_info=True)
            return None
        finally:
            self._stop_event.clear()
            self._vad_queue = None # Clear the queue
            if self._stream and self._stream.active:
                self._stream.stop()
                self._stream.close()
            self._stream = None
    # --- END: ADD record_with_vad METHOD ---

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        """sounddevice callback: store frames and stop if requested."""
        if status:
            logger.debug("InputStream status: %s", status)

        # Copy chunk to avoid referencing same memory
        with self._lock:
            chunk = indata.copy()
            # Flatten single-channel to 1-D for convenience
            if chunk.ndim > 1 and chunk.shape[1] == 1:
                chunk = chunk.flatten()
            self._frames.append(chunk)

        # If stop requested, raise to stop stream cleanly
        if self._stop_event.is_set():
            logger.debug("Stop event seen in callback -> raising CallbackStop")
            raise sd.CallbackStop()

    def record_audio(self,
                           duration: float = 300.0,
                           progress_callback: Optional[Callable[[float], None]] = None,
                           device: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Record audio for up to `duration` seconds. Returns earlier if stop_recording() is called.
        """
        logger.info("record_audio called: duration=%s device=%s", duration, device or self.device)
        self._stop_event.clear()
        with self._lock:
            self._frames = []

        chosen_device = device if device is not None else self.device

        try:
            self._stream = sd.InputStream(samplerate=self.sample_rate,
                                          channels=self.channels,
                                          dtype=self.dtype,
                                          blocksize=self.blocksize,
                                          device=chosen_device,
                                          callback=self._callback)

            logger.debug("Opening InputStream (device=%s)...", chosen_device)
            with self._stream:
                waited = 0.0
                poll = 0.05  # 50 ms poll interval for responsive stop
                while waited < duration and not self._stop_event.is_set():
                    sd.sleep(int(poll * 1000))
                    waited += poll
                    if progress_callback:
                        try:
                            progress_callback(min(waited / duration, 1.0))
                        except Exception:
                            logger.debug("progress_callback raised", exc_info=True)

            # At exit collect frames
            with self._lock:
                if not self._frames:
                    logger.warning("No audio frames captured")
                    return None
                audio_np = np.concatenate(self._frames, axis=0)

            logger.info("InputStream recording finished normally or by stop event")

        except Exception as e:
            # If callback raised CallbackStop, or other error occurred, gather what we have
            logger.info("record_audio exception (likely stop): %s", e)
            with self._lock:
                if not self._frames:
                    logger.warning("No audio frames captured after exception")
                    return None
                audio_np = np.concatenate(self._frames, axis=0)

        # Convert to mono 1-D if needed
        if audio_np.ndim > 1:
            # If it's shape (n,1) flatten
            if audio_np.shape[1] == 1:
                audio_np = audio_np.flatten()
            else:
                # mix to mono
                audio_np = np.mean(audio_np, axis=1)

        # Ensure dtype float32
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)

        # --- START FIX: Bypassing all processing per user request ---
        logger.debug("Bypassing audio processing. Using raw audio.")
        # if audio_np.size:
        #    logger.debug("Applying AGC to recorded audio for transcription...")
        #    audio_np = AudioProcessor.apply_automatic_gain_control(audio_np)
        # --- END FIX ---

        logger.info("Recording complete: samples=%d duration%.2fs", audio_np.shape[0], audio_np.shape[0] / self.sample_rate)
        return audio_np

    def stop_recording(self):
        """Signal the running recorder to stop as soon as possible."""
        logger.info("stop_recording() called -> setting stop event")
        self._stop_event.set()
        
        # --- START FIX: Manually wake up VAD queue ---
        if self._vad_queue is not None:
            try:
                self._vad_queue.put(None, timeout=0.1)
            except queue.Full:
                pass # Queue is full, consumer will get it
        # --- END FIX ---


    def save_audio(self, audio: np.ndarray, path: str) -> bool:
        """
        Save float32 audio array to file using soundfile (PCM16).
        """
        try:
            import soundfile as sf  # optional dependency
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            # sf.write accepts float32 in [-1,1]
            sf.write(path, audio, self.sample_rate, subtype="PCM_16")
            logger.info("Saved audio to %s", path)
            return True
        except Exception as e:
            logger.exception("Failed to save audio: %s", e)
            return False


class VoiceActivityDetector:
    """
    Lightweight RMS VAD (fast).
    """
    def __init__(self, config: dict = None):
        cfg = config or {}
        audio_cfg = cfg.get("audio", {}) if cfg else {}
        self.sample_rate = int(audio_cfg.get("sample_rate", 16000))
        self.rms_threshold = float(audio_cfg.get("vad_threshold", 0.001))
        self.min_speech_duration = float(audio_cfg.get("min_speech_duration", 0.3))

        logger.info("VoiceActivityDetector initialized (rms_threshold=%s)", self.rms_threshold)

    def detect_rms(self, audio: np.ndarray) -> Tuple[bool, float]:
        if audio.size == 0:
            return False, 0.0
        rms = float(np.sqrt(np.mean(audio ** 2)))
        has_voice = rms > self.rms_threshold
        # logger.debug("RMS VAD: rms=%.6f threshold=%.6f -> %s", rms, self.rms_threshold, has_voice)
        return has_voice, rms

    # Removed Silero VAD functions for simplicity as it wasn't being used by record_with_vad

class AudioProcessor:
    """
    Utility helpers for audio processing (AGC, normalization, trim, bandpass).
    """

    @staticmethod
    def apply_automatic_gain_control(audio: np.ndarray, target_level: float = 0.15) -> np.ndarray:
        """Applies AGC. NOTE: This is currently BYPASSED in record_audio."""
        if audio.size == 0:
            return audio
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms <= 1e-5: # Avoid division by zero/near-zero
            logger.debug("Skipping AGC, audio is silent")
            return audio
        gain = target_level / rms
        gain = min(gain, 10.0) # Cap gain to avoid extreme amplification
        adjusted = audio * gain
        
        # Soft clipping with tanh to prevent harsh clipping
        adjusted = np.tanh(adjusted).astype(np.float32) 
        
        logger.debug("AGC applied: gain=%.3f", gain)
        return adjusted

    @staticmethod
    def normalize(audio: np.ndarray) -> np.ndarray:
        """Applies peak normalization."""
        if audio.size == 0:
            return audio
        peak = float(np.max(np.abs(audio)))
        if peak <= 1e-5: # Avoid division by zero/near-zero
            logger.debug("Skipping normalization, audio is silent")
            return audio
        return (audio / peak).astype(np.float32)

    @staticmethod
    def trim_silence(audio: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
        if audio.size == 0:
            return audio
        mask = np.abs(audio) > threshold
        if not mask.any():
            return np.array([], dtype=np.float32)
        idx = np.where(mask)[0]
        return audio[idx[0]:idx[-1] + 1].astype(np.float32)

    @staticmethod
    def apply_bandpass(audio: np.ndarray, sr: int, lowcut: float = 300.0, highcut: float = 3400.0) -> np.ndarray:
        logger.warning(
            "Applying 300-3400Hz bandpass filter. This is for telephony simulation "
            "and WILL HARM transcription accuracy for modern speech models. "
            "Disable this if not required."
        )
        try:
            from scipy import signal
            nyq = sr / 2.0
            low = lowcut / nyq
            high = highcut / nyq
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, audio.astype(np.float64))
            return filtered.astype(np.float32)
        except Exception as e:
            logger.warning("Bandpass filter failed (scipy missing?): %s", e)
            return audio