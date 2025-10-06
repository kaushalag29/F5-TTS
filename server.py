"""
F5-TTS Server
Hosts the F5-TTS model for audio generation via HTTP API.
"""
import os
import sys
import random
import numpy as np
import torch
import torchaudio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Add F5-TTS to path
f5_tts_path = Path(__file__).parent / "src"
sys.path.insert(0, str(f5_tts_path))

from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import chunk_text, load_vocoder, load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance
f5tts_model = None
vocoder = None
device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    initialize_model()
    yield
    # Shutdown (cleanup if needed)
    pass

app = FastAPI(lifespan=lifespan)

class GenerateAudioRequest(BaseModel):
    text: str
    reference_audio_path: str
    reference_text: str
    output_path: str
    model_type: str = "F5-TTS"  # or "E2-TTS"
    remove_silence: bool = False
    cross_fade_duration: float = 0.15
    speed: float = 1.0
    
def initialize_model():
    """Initialize F5-TTS model on startup."""
    global f5tts_model, vocoder, device
    
    logger.info("Initializing F5-TTS model...")
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    try:
        # Initialize F5-TTS
        # model parameter: "F5TTS_v1_Base" (default), "F5TTS_Base", "E2TTS_Base"
        f5tts_model = F5TTS(
            model="F5TTS_v1_Base",  # Model name (not model_type)
            ckpt_file="",  # Empty string will use default checkpoint
            vocab_file="",  # Empty string will use default vocab
            ode_method="euler",
            use_ema=True,
            device=device
        )
        
        # Vocoder is loaded internally by F5TTS, get reference to it
        vocoder = f5tts_model.vocoder
        
        logger.info("F5-TTS model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize F5-TTS model: {e}", exc_info=True)
        raise

@app.post("/generate-audio")
async def generate_audio(request: GenerateAudioRequest):
    """
    Generate audio using F5-TTS with reference audio and text.
    
    Args:
        text: The text to generate speech for (gentext)
        reference_audio_path: Path to reference audio file
        reference_text: Transcript of reference audio (reftext)
        output_path: Where to save the generated audio
        model_type: "F5-TTS" or "E2-TTS"
        remove_silence: Whether to remove silence from output
        cross_fade_duration: Duration for cross-fading between chunks
        speed: Speech speed adjustment (1.0 = normal)
    """
    try:
        logger.info(f"Received generate-audio request")
        logger.info(f"Text length: {len(request.text)} chars")
        logger.info(f"Reference audio: {request.reference_audio_path}")
        logger.info(f"Reference text: {request.reference_text[:50]}...")
        
        if not os.path.exists(request.reference_audio_path):
            raise HTTPException(status_code=400, detail=f"Reference audio not found: {request.reference_audio_path}")
        
        # Load reference audio
        ref_audio, ref_sr = torchaudio.load(request.reference_audio_path)
        
        # Resample to 24kHz if needed
        if ref_sr != 24000:
            resampler = torchaudio.transforms.Resample(ref_sr, 24000)
            ref_audio = resampler(ref_audio)
            ref_sr = 24000
        
        # Convert to mono if stereo
        if ref_audio.shape[0] > 1:
            ref_audio = ref_audio.mean(dim=0, keepdim=True)
        
        # Clip reference audio to ~12 seconds max as per F5-TTS recommendations
        max_ref_samples = int(12.0 * ref_sr)
        if ref_audio.shape[1] > max_ref_samples:
            logger.info(f"Clipping reference audio from {ref_audio.shape[1]/ref_sr:.2f}s to 12s")
            ref_audio = ref_audio[:, :max_ref_samples]
        
        # Ensure reference text is not empty
        if not request.reference_text or not request.reference_text.strip():
            logger.warning("Reference text is empty, using placeholder")
            request.reference_text = "Reference audio."
        
        # Ensure output directory exists BEFORE generation
        output_dir = os.path.dirname(request.output_path)
        if output_dir:  # Only create if there's a directory component
            os.makedirs(output_dir, exist_ok=True)
        
        # Prepare generation parameters
        logger.info("Starting F5-TTS generation...")
        logger.info(f"  Reference text: '{request.reference_text[:100]}...'")
        logger.info(f"  Generate text: '{request.text[:100]}...'")
        
        # Use a valid seed to avoid PYTHONHASHSEED range errors on Mac
        # Valid range is [0, 4294967295] (Python hash seed constraint)
        seed = random.randint(0, 4294967295)
        
        # Use the F5TTS API for generation
        # The API handles chunking automatically for long text
        # Note: model_obj, vocoder, device are already part of f5tts_model instance
        wav_output, sr_output, _ = f5tts_model.infer(
            ref_file=request.reference_audio_path,
            ref_text=request.reference_text,
            gen_text=request.text,
            show_info=print,
            remove_silence=request.remove_silence,
            cross_fade_duration=request.cross_fade_duration,
            speed=request.speed,
            seed=seed,  # Pass valid seed to avoid PYTHONHASHSEED errors
        )
        
        # Save output audio at 24kHz (F5-TTS standard output)
        # F5TTS.infer() returns numpy array, need to convert to tensor for torchaudio.save
        if isinstance(wav_output, np.ndarray):
            # Convert numpy array to tensor
            wav_output = torch.from_numpy(wav_output)
        
        # Ensure correct shape for torchaudio.save (channels, samples)
        if wav_output.dim() == 1:
            wav_output = wav_output.unsqueeze(0)
        
        torchaudio.save(request.output_path, wav_output.cpu(), sr_output)
        
        logger.info(f"Audio generated successfully: {request.output_path}")
        return {
            "status": "success",
            "output_path": request.output_path,
            "sample_rate": sr_output,
            "duration": wav_output.shape[-1] / sr_output
        }
        
    except Exception as e:
        logger.error(f"Error generating audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Check if server is healthy."""
    return {
        "status": "healthy",
        "model_loaded": f5tts_model is not None,
        "device": device if device else "unknown"
    }

@app.post("/shutdown")
async def shutdown():
    """Shutdown the server gracefully."""
    logger.info("Shutdown request received")
    return {"status": "shutting down"}

if __name__ == "__main__":
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8017,  # Unique port for F5-TTS
        log_level="info"
    )
