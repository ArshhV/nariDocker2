#!/usr/bin/env python3

import requests
import json
import os
import base64
import time
from dotenv import load_dotenv

# Load environment variables from .env file (if available)
load_dotenv()

# Set API key and endpoint ID from environment variables
API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("NARI_DIA_ENDPOINT_ID")

# Check if environment variables are set
if not API_KEY or API_KEY == "your_api_key_here":
    print("Warning: RUNPOD_API_KEY is not set. Please set it in your .env file or environment.")
    
if not ENDPOINT_ID or ENDPOINT_ID == "your_endpoint_id_here":
    print("Warning: NARI_DIA_ENDPOINT_ID is not set. Please set it in your .env file or environment.")

# Define API URLs
RUN_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run" if ENDPOINT_ID else "https://api.runpod.ai/v2/your_endpoint_id_here/run"
STATUS_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/" if ENDPOINT_ID else "https://api.runpod.ai/v2/your_endpoint_id_here/status/"
HEALTH_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/health" if ENDPOINT_ID else "https://api.runpod.ai/v2/your_endpoint_id_here/health"

# Define path for output audio files
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

def check_endpoint_health():
    """Check if the RunPod endpoint is healthy."""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    try:
        response = requests.get(HEALTH_URL, headers=headers)
        print(f"Health Check - Status Code: {response.status_code}")
        print(f"Health Check - Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed with error: {e}")
        return False

def save_audio(audio_base64, sample_rate=24000, filename="generated_audio.wav"):
    """Save a base64-encoded audio to a local file"""
    try:
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Save to the output folder
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Decode the base64 audio data
        audio_data = base64.b64decode(audio_base64)
        
        # Write the data to file
        with open(output_path, "wb") as f:
            f.write(audio_data)
            
        # Print the absolute path for easier finding
        abs_path = os.path.abspath(output_path)
        print(f"Audio saved as {output_path}")
        print(f"Absolute path: {abs_path}")
        return True
    except Exception as e:
        print(f"Error saving audio: {e}")
        return False

def generate_audio(text, seed=None, use_compile=True, output_filename="generated_audio.wav"):
    """Generate audio using the Nari Dia TTS model via RunPod endpoint"""
    # Create the request payload
    payload = {
        "input": {
            "text": text,
            "use_compile": use_compile
        }
    }
    
    # Add optional seed if provided
    if seed is not None:
        payload["input"]["seed"] = seed
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print("Sending request to RunPod Nari Dia endpoint...")
    try:
        response = requests.post(RUN_URL, headers=headers, json=payload)
        print(f"Response Status Code: {response.status_code}")
        
        if response.status_code == 200:
            job_id = response.json().get("id")
            print(f"Job submitted with ID: {job_id}")
            
            # Poll for job completion
            while True:
                status_response = requests.get(f"{STATUS_URL}{job_id}", headers=headers)
                status_data = status_response.json()
                
                status = status_data.get("status")
                print(f"Current status: {status}")
                
                if status == "COMPLETED":
                    output = status_data.get("output")
                    print(f"Job completed successfully!")
                    
                    if output and isinstance(output, dict):
                        # Extract the audio data
                        audio_base64 = output.get("output", {}).get("audio")
                        sample_rate = output.get("output", {}).get("sample_rate", 24000)
                        duration = output.get("output", {}).get("duration")
                        
                        if audio_base64:
                            print(f"Received audio data (duration: {duration:.2f}s, sample rate: {sample_rate}Hz)")
                            save_audio(audio_base64, sample_rate, output_filename)
                            return output
                        else:
                            print("No audio data found in the response")
                            print(f"Full response: {json.dumps(output, indent=2)}")
                    else:
                        print("Unexpected output format")
                        print(f"Full response: {json.dumps(status_data, indent=2)}")
                    
                    return output
                
                elif status in ["FAILED", "ERROR"]:
                    error_msg = status_data.get('error', 'Unknown error')
                    print(f"Job failed with error: {error_msg}")
                    return None
                
                # Wait before checking again
                time.sleep(3)
        else:
            print(f"Failed to submit job: {response.text}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def interactive_mode():
    """Run in interactive mode allowing users to enter text and generate audio."""
    print("====================================================")
    print("Welcome to Nari Dia Text-to-Speech")
    print("Type 'quit', 'exit', or 'bye' to end the session")
    print("====================================================")
    
    count = 1
    while True:
        user_input = input("\nEnter text to convert to speech: ").strip()
        
        # Check if user wants to exit
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        # Auto-add speaker tag if not present
        if not user_input.startswith("[S1]") and not user_input.startswith("[S2]"):
            print("Adding default [S1] speaker tag")
            user_input = "[S1] " + user_input
        
        # Generate unique filename for each run
        filename = f"audio_{count}.wav"
        
        # Generate audio
        result = generate_audio(user_input, output_filename=filename)
        
        if result:
            print(f"Audio generated successfully and saved as {filename}")
        else:
            print("Failed to generate audio")
        
        count += 1

def file_mode(input_file):
    """Process text from a file and generate audio."""
    try:
        with open(input_file, 'r') as f:
            if input_file.endswith('.json'):
                # JSON file with specific structure
                data = json.load(f)
                text = data.get("input", {}).get("text", "")
                seed = data.get("input", {}).get("seed")
                use_compile = data.get("input", {}).get("use_compile", True)
            else:
                # Plain text file
                text = f.read().strip()
                seed = None
                use_compile = True
                
        if not text:
            print("No text found in the input file")
            return
            
        print(f"Text from file: {text[:100]}...")
        
        # Auto-add speaker tag if not present
        if not text.startswith("[S1]") and not text.startswith("[S2]"):
            print("Adding default [S1] speaker tag")
            text = "[S1] " + text
        
        # Use filename as output name
        output_filename = os.path.splitext(os.path.basename(input_file))[0] + ".wav"
        
        # Generate audio
        result = generate_audio(text, seed, use_compile, output_filename)
        
        if result:
            print(f"Audio generated successfully and saved as {output_filename}")
        else:
            print("Failed to generate audio")
    
    except Exception as e:
        print(f"Error processing file: {e}")

def main():
    """Main function to run the script in different modes."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate speech using Nari Dia TTS model")
    parser.add_argument("--text", "-t", help="Text to convert to speech")
    parser.add_argument("--file", "-f", help="Input file containing text or JSON input")
    parser.add_argument("--output", "-o", help="Output filename (default: generated based on input)")
    parser.add_argument("--seed", "-s", type=int, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # First check if the endpoint is healthy
    print("Checking endpoint health...")
    if not check_endpoint_health():
        print("Endpoint health check failed. Please check your endpoint ID and API key.")
        return
    
    # Show output directory information
    abs_output_dir = os.path.abspath(OUTPUT_DIR)
    print(f"Audio files will be saved to: {abs_output_dir}")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if args.text:
        # Direct text input mode
        text = args.text
        output_filename = args.output or "generated_audio.wav"
        
        # Auto-add speaker tag if not present
        if not text.startswith("[S1]") and not text.startswith("[S2]"):
            print("Adding default [S1] speaker tag")
            text = "[S1] " + text
        
        # Generate audio
        result = generate_audio(text, args.seed, True, output_filename)
        
        if result:
            print(f"Audio generated successfully and saved as {output_filename}")
        else:
            print("Failed to generate audio")
    
    elif args.file:
        # File input mode
        file_mode(args.file)
    
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()