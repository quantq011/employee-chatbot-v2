"""
Text-to-Speech Service
Provides FREE TTS engines (gTTS, pyttsx3) for converting text to speech
"""
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Literal
import logging
 
logger = logging.getLogger(__name__)
 
class TTSEngine(ABC):
    """Abstract base class for TTS engines"""
   
    @abstractmethod
    def synthesize(self, text: str, output_path: str) -> bool:
        """
        Synthesize text to speech and save to file
       
        Args:
            text: Text to convert to speech
            output_path: Path to save the audio file
           
        Returns:
            True if successful, False otherwise
        """
        pass
   
    @abstractmethod
    def get_audio_format(self) -> str:
        """Get the audio format (mp3, wav, etc.)"""
        pass
 
 
class GoogleTTSEngine(TTSEngine):
    """Google Text-to-Speech (gTTS) - Free, requires internet"""
   
    def __init__(self, language: str = "en", slow: bool = False):
        """
        Initialize Google TTS
       
        Args:
            language: Language code (default: 'en')
            slow: Speak slower (default: False)
        """
        try:
            from gtts import gTTS
            self.gTTS = gTTS
            self.language = language
            self.slow = slow
            self.available = True
            logger.info(f"✅ Google TTS (gTTS) initialized with language: {language}")
        except ImportError:
            logger.warning("⚠️ gTTS not installed. Install with: pip install gtts")
            self.available = False
   
    def synthesize(self, text: str, output_path: str) -> bool:
        if not self.available:
            return False
       
        try:
            tts = self.gTTS(text=text, lang=self.language, slow=self.slow)
            tts.save(output_path)
            logger.info(f"✅ Google TTS: Audio saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Google TTS error: {e}")
            return False
   
    def get_audio_format(self) -> str:
        return "mp3"
 
 
class SystemTTSEngine(TTSEngine):
    """System Text-to-Speech (pyttsx3) - Offline, platform-specific"""
   
    def __init__(self, rate: int = 150, volume: float = 1.0):
        """
        Initialize System TTS
       
        Args:
            rate: Speech rate (words per minute, default: 150)
            volume: Volume level 0.0-1.0 (default: 1.0)
        """
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            self.available = True
            logger.info(f"✅ System TTS (pyttsx3) initialized with rate: {rate}")
        except ImportError:
            logger.warning("⚠️ pyttsx3 not installed. Install with: pip install pyttsx3")
            self.available = False
        except Exception as e:
            logger.error(f"❌ Failed to initialize System TTS: {e}")
            self.available = False
   
    def synthesize(self, text: str, output_path: str) -> bool:
        if not self.available:
            return False
       
        try:
            self.engine.save_to_file(text, output_path)
            self.engine.runAndWait()
            logger.info(f"✅ System TTS: Audio saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ System TTS error: {e}")
            return False
   
    def get_audio_format(self) -> str:
        return "wav"
 
 
class TTSService:
    """Main TTS service that manages FREE TTS engines only"""
   
    def __init__(self, engine_type: Literal["google", "system"] = "google"):
        """
        Initialize TTS service (FREE engines only)
       
        Args:
            engine_type: Type of TTS engine to use (google, system)
        """
        self.engine_type = engine_type
        self.engine: Optional[TTSEngine] = None
        self._initialize_engine()
   
    def _initialize_engine(self):
        """Initialize the selected TTS engine"""
        if self.engine_type == "google":
            language = os.getenv("TTS_LANGUAGE", "en")
            self.engine = GoogleTTSEngine(language=language)
       
        elif self.engine_type == "system":
            rate = int(os.getenv("TTS_RATE", "150"))
            self.engine = SystemTTSEngine(rate=rate)
       
        else:
            logger.warning(f"⚠️ Unknown engine type: {self.engine_type}, falling back to Google TTS")
            self.engine_type = "google"
            self.engine = GoogleTTSEngine()  # Fallback to Google
   
    def text_to_speech(self, text: str, output_filename: Optional[str] = None) -> Optional[str]:
        """
        Convert text to speech and save to file
       
        Args:
            text: Text to convert to speech
            output_filename: Optional filename. If None, generates temp file.
           
        Returns:
            Path to audio file if successful, None otherwise
        """
        if not self.engine or not self.engine.available:
            logger.error("❌ No TTS engine available")
            return None
       
        if not text or not text.strip():
            logger.warning("⚠️ Empty text provided")
            return None
       
        # Clean text for better speech
        text = self._clean_text(text)
       
        # Generate output path
        if output_filename:
            output_path = output_filename
        else:
            audio_format = self.engine.get_audio_format()
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"tts_output.{audio_format}")
       
        # Synthesize speech
        success = self.engine.synthesize(text, output_path)
       
        if success:
            return output_path
        else:
            return None
   
    def _clean_text(self, text: str) -> str:
        """Clean text for better TTS output"""
        # Remove markdown code blocks
        import re
        text = re.sub(r'```[\s\S]*?```', ' [code block] ', text)
       
        # Remove inline code
        text = re.sub(r'`[^`]+`', ' ', text)
       
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', ' [link] ', text)
       
        # Remove special markdown characters
        text = text.replace('**', '').replace('*', '').replace('__', '').replace('_', '')
        text = text.replace('#', '').replace('>', '').replace('-', ' ')
       
        # Remove extra whitespace
        text = ' '.join(text.split())
       
        return text
   
    def is_available(self) -> bool:
        """Check if TTS engine is available"""
        return self.engine is not None and self.engine.available
   
    def get_engine_info(self) -> dict:
        """Get information about current engine"""
        return {
            "engine_type": self.engine_type,
            "available": self.is_available(),
            "audio_format": self.engine.get_audio_format() if self.engine else None
        }
 
 
# Global TTS service instance
_tts_service: Optional[TTSService] = None
 
 
def get_tts_service(engine_type: Optional[str] = None) -> TTSService:
    """
    Get or create TTS service instance (Singleton pattern)
   
    Args:
        engine_type: Optional engine type override
       
    Returns:
        TTSService instance
    """
    global _tts_service
   
    if _tts_service is None or engine_type:
        engine = engine_type or os.getenv("TTS_ENGINE", "google")
        _tts_service = TTSService(engine_type=engine)
   
    return _tts_service
 
 
# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
   
    # Test TTS service
    tts = get_tts_service("google")
   
    test_text = """
    Hello! This is a test of the text-to-speech system.
    I can convert any text into natural-sounding speech.
    This works great for reading AI responses aloud!
    """
   
    output_path = tts.text_to_speech(test_text)
   
    if output_path:
        print(f"✅ Audio saved to: {output_path}")
        print(f"🔊 Engine info: {tts.get_engine_info()}")
    else:
        print("❌ TTS failed")
 