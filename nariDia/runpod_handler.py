#!/usr/bin/env python3

import base64
import io
import os
import runpod
from dia.model import Dia
import torch
import soundfile as sf

# Global model loaded once on cold start
MODEL = None

def load_model():
    global MODEL
    if MODEL is None:
        try:
            # Use the cache path where the model was pre-downloaded in Dockerfile
            cache_dir = "/root/.cache/huggingface/hub"
            runpod.log(f"Looking for model in cache dir: {cache_dir}")
            
            runpod.log("About to load model from pretrained")
            # Load model with float16 for better performance
            MODEL = Dia.from_pretrained(
                "nari-labs/Dia-1.6B", 
                compute_dtype="float16",
                cache_dir=cache_dir,
                local_files_only=True  # Force using local files only
            )
            runpod.log("Model loaded from pretrained successfully")
            
            # Print GPU info
            if torch.cuda.is_available():
                runpod.log(f"Using GPU: {torch.cuda.get_device_name(0)}")
                runpod.log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                runpod.log("WARNING: CUDA not available, using CPU")
        except Exception as e:
            runpod.log(f"ERROR in load_model: {str(e)}")
            raise e

def handler(event, context):
    """
    RunPod handler for Nari Dia text-to-speech model
    """
    runpod.log("==== HANDLER START - Received request ====")
    
    # Load model if not already loaded
    if MODEL is None:
        try:
            runpod.log("Model not loaded yet, initializing...")
            load_model()
            runpod.log("Model loaded successfully and ready for inference")
        except Exception as e:
            error_msg = f"CRITICAL ERROR: Failed to load model: {str(e)}"
            runpod.log(error_msg)
            return {"error": error_msg}
    
    try:
        # Extract inputs
        runpod.log("Extracting input data from event")
        input_data = event["input"]
        text = input_data.get("text")
        
        if not text:
            runpod.log("ERROR: Missing required text input")
            return {"error": "Missing required input: text"}
        
        runpod.log(f"Input text received (first 50 chars): {text[:50]}...")
        
        # Check if input starts with speaker tag
        if not text.startswith("[S1]") and not text.startswith("[S2]"):
            runpod.log("Adding default [S1] speaker tag to input")
            text = "[S1] " + text
        
        # Optional parameters
        seed = input_data.get("seed", None)
        audio_prompt = input_data.get("audio_prompt", None)
        use_compile = input_data.get("use_compile", True)
        
        runpod.log(f"Parameters - seed: {seed}, use_compile: {use_compile}")
        
        # Run inference
        runpod.log("Starting inference process...")
        
        # Set seed if provided
        if seed is not None:
            runpod.log(f"Setting torch manual seed to {seed}")
            torch.manual_seed(seed)
            
        # Generate audio
        runpod.log("Calling MODEL.generate()...")
        output = MODEL.generate(
            text, 
            use_torch_compile=use_compile,
            verbose=True
        )
        runpod.log("MODEL.generate() completed successfully")
        
        # Save to a BytesIO buffer instead of a file
        runpod.log("Saving audio to BytesIO buffer...")
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, output.audio.numpy(), output.sample_rate, format='WAV')
        audio_buffer.seek(0)
        runpod.log(f"Audio saved to buffer, size: {audio_buffer.getbuffer().nbytes} bytes")
        
        # Encode as base64
        runpod.log("Encoding audio as base64...")
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        runpod.log(f"Base64 encoding complete, length: {len(audio_base64)}")
        
        runpod.log("==== Inference complete, returning response ====")
        return {
            "output": {
                "audio": audio_base64,
                "sample_rate": output.sample_rate,
                "format": "wav",
                "duration": len(output.audio) / output.sample_rate
            }
        }

    except Exception as e:
        error_message = f"ERROR in handler: {str(e)}"
        runpod.log(error_message)
        import traceback
        runpod.log(f"Traceback: {traceback.format_exc()}")
        return {"error": error_message}

if __name__ == "__main__":
    runpod.log("Starting RunPod serverless handler")
    runpod.serverless.start({"handler": handler})
    runpod.log("RunPod serverless handler terminated")