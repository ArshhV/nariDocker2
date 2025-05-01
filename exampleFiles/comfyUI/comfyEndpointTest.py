import requests
import json
import os
import time
import base64
from dotenv import load_dotenv

# Load API key from your .env.local file
load_dotenv('.env.local')
api_key = os.getenv('RUNPOD_API_KEY')

# RunPod endpoint ID
endpoint_id = "5lq23g82tx2u2k"

# Define API URLs
health_url = f"https://api.runpod.ai/v2/{endpoint_id}/health"
run_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/"

# Define path for output files
# You can set this to an absolute path to ensure files are saved in a known location
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'public')

def check_endpoint_health():
    """Check if the RunPod endpoint is healthy."""
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(health_url, headers=headers)
        print(f"Health Check - Status Code: {response.status_code}")
        print(f"Health Check - Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed with error: {e}")
        return False

def save_image(url, filename="generated_image.png"):
    """Save an image from a URL to a local file"""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Ensure public directory exists
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Save to the public folder
            public_path = os.path.join(OUTPUT_DIR, filename)
            with open(public_path, "wb") as f:
                f.write(response.content)
            print(f"Image saved as {public_path}")
            return True
        else:
            print(f"Failed to download image: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def save_base64_image(base64_string, filename="generated_image.png"):
    """Save a base64-encoded image to a local file"""
    try:
        # Ensure public directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Save to the public folder
        public_path = os.path.join(OUTPUT_DIR, filename)
        
        # Handle different base64 formats
        # If the string starts with data URI scheme like "data:image/png;base64,"
        if isinstance(base64_string, str) and base64_string.startswith('data:'):
            # Extract the actual base64 data after the comma
            base64_data = base64_string.split(',', 1)[1]
        else:
            base64_data = base64_string
            
        # Make sure we're working with a string
        if not isinstance(base64_data, str):
            base64_data = str(base64_data)
            
        # Remove any whitespace that might be in the string
        base64_data = base64_data.strip()
            
        try:
            # Decode the base64 string and save as image
            image_data = base64.b64decode(base64_data)
            
            # Check the file signature (magic bytes) to determine the correct file extension
            # This helps ensure we save with the correct format
            if image_data.startswith(b'\xFF\xD8\xFF'):  # JPEG signature
                if not filename.lower().endswith(('.jpg', '.jpeg')):
                    filename = os.path.splitext(filename)[0] + '.jpg'
                    public_path = os.path.join(OUTPUT_DIR, filename)
                print("Detected JPEG image format")
            elif image_data.startswith(b'\x89PNG\r\n\x1A\n'):  # PNG signature
                if not filename.lower().endswith('.png'):
                    filename = os.path.splitext(filename)[0] + '.png'
                    public_path = os.path.join(OUTPUT_DIR, filename)
                print("Detected PNG image format")
            elif image_data.startswith(b'GIF87a') or image_data.startswith(b'GIF89a'):  # GIF signature
                if not filename.lower().endswith('.gif'):
                    filename = os.path.splitext(filename)[0] + '.gif'
                    public_path = os.path.join(OUTPUT_DIR, filename)
                print("Detected GIF image format")
            elif image_data.startswith(b'RIFF') and image_data[8:12] == b'WEBP':  # WEBP signature
                if not filename.lower().endswith('.webp'):
                    filename = os.path.splitext(filename)[0] + '.webp'
                    public_path = os.path.join(OUTPUT_DIR, filename)
                print("Detected WEBP image format")
            
            # Write the data to file
            with open(public_path, "wb") as f:
                f.write(image_data)
            
            # Print the absolute path for easier debugging
            abs_path = os.path.abspath(public_path)
            print(f"Base64 image saved as {public_path}")
            print(f"Absolute path: {abs_path}")
            return True
        except Exception as decode_error:
            print(f"Error decoding base64 data: {decode_error}")
            
            # As a fallback, try to save the raw data
            print("Attempting to save raw data...")
            if isinstance(base64_string, str):
                with open(public_path, "w") as f:
                    f.write(base64_string)
            else:
                with open(public_path, "wb") as f:
                    f.write(base64_string)
            
            print(f"Raw data saved to {public_path}")
            return True
    except Exception as e:
        print(f"Error saving base64 image: {e}")
        return False

def save_video(video_data, filename="generated_video.mp4"):
    """Save video data to a local file"""
    try:
        # Ensure public directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Save to the public folder
        public_path = os.path.join(OUTPUT_DIR, filename)
        
        # Write the data to file
        with open(public_path, "wb") as f:
            if isinstance(video_data, str):
                # If it's a base64 string
                f.write(base64.b64decode(video_data))
            else:
                # If it's binary data
                f.write(video_data)
        
        print(f"Video saved as {public_path}")
        return True
    except Exception as e:
        print(f"Error saving video: {e}")
        return False

def generate_image(prompt="cute anime girl with massive fluffy fennec ears and a big fluffy tail blonde messy long hair blue eyes wearing a maid outfit with a long black gold leaf pattern dress and a white apron mouth open placing a fancy black forest cake with candles on top of a dinner table of an old dark Victorian mansion lit by candlelight with a bright window to the foggy forest and very expensive stuff everywhere there are paintings on the walls", 
                  negative_prompt="",
                  output_filename="generated_image.png"):
    """
    Generate content using the FLUX workflow format for RunPod
    """
    # The updated workflow format - FLUX workflow
    workflow = {
      "6": {
        "inputs": {
          "text": prompt,
          "clip": [
            "30",
            1
          ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
          "title": "CLIP Text Encode (Positive Prompt)"
        }
      },
      "8": {
        "inputs": {
          "samples": [
            "31",
            0
          ],
          "vae": [
            "30",
            2
          ]
        },
        "class_type": "VAEDecode",
        "_meta": {
          "title": "VAE Decode"
        }
      },
      "9": {
        "inputs": {
          "filename_prefix": "ComfyUI",
          "images": [
            "8",
            0
          ]
        },
        "class_type": "SaveImage",
        "_meta": {
          "title": "Save Image"
        }
      },
      "27": {
        "inputs": {
          "width": 1024,
          "height": 1024,
          "batch_size": 1
        },
        "class_type": "EmptySD3LatentImage",
        "_meta": {
          "title": "EmptySD3LatentImage"
        }
      },
      "30": {
        "inputs": {
          "ckpt_name": "flux1-dev-fp8.safetensors"
        },
        "class_type": "CheckpointLoaderSimple",
        "_meta": {
          "title": "Load Checkpoint"
        }
      },
      "31": {
        "inputs": {
          "seed": 237953136648084,
          "steps": 20,
          "cfg": 1,
          "sampler_name": "euler",
          "scheduler": "simple",
          "denoise": 1,
          "model": [
            "30",
            0
          ],
          "positive": [
            "35",
            0
          ],
          "negative": [
            "33",
            0
          ],
          "latent_image": [
            "27",
            0
          ]
        },
        "class_type": "KSampler",
        "_meta": {
          "title": "KSampler"
        }
      },
      "33": {
        "inputs": {
          "text": negative_prompt,
          "clip": [
            "30",
            1
          ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
          "title": "CLIP Text Encode (Negative Prompt)"
        }
      },
      "35": {
        "inputs": {
          "guidance": 3.5,
          "conditioning": [
            "6",
            0
          ]
        },
        "class_type": "FluxGuidance",
        "_meta": {
          "title": "FluxGuidance"
        }
      }
    }
    
    # Create the request payload
    payload = {
      "input": {
        "workflow": workflow
      }
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    print("Sending request to RunPod FLUX endpoint...")
    response = requests.post(run_url, headers=headers, json=payload)
    print(f"Response Status Code: {response.status_code}")
    
    if response.status_code == 200:
        job_id = response.json().get("id")
        print(f"Job submitted with ID: {job_id}")
        
        # Poll for job completion
        while True:
            status_response = requests.get(f"{status_url}{job_id}", headers=headers)
            status_data = status_response.json()
            
            status = status_data.get("status")
            print(f"Current status: {status}")
            
            if status == "COMPLETED":
                output = status_data.get("output")
                print(f"Job completed successfully!")
                print(f"Output type: {type(output)}")
                print(f"Full response structure: {json.dumps(status_data, indent=2)}")
                
                # Try to find image data in the response, no matter where it is
                try:
                    if output:
                        print(f"Output keys: {output.keys() if isinstance(output, dict) else 'Not a dictionary'}")
                        
                        # Check for base64-encoded image in "message" field
                        if isinstance(output, dict) and 'message' in output:
                            print("Found base64-encoded data in response 'message' field")
                            save_base64_image(output['message'], output_filename)
                            return output
                        
                        # Check for images array
                        if isinstance(output, dict) and 'images' in output and isinstance(output['images'], list) and output['images']:
                            print(f"Found images array with {len(output['images'])} items")
                            for i, img_data in enumerate(output['images']):
                                if isinstance(img_data, str) and (img_data.startswith('http') or img_data.startswith('https')):
                                    # It's a URL, download it
                                    print(f"Image {i} is a URL")
                                    save_image(img_data, output_filename)
                                else:
                                    # Assume it's base64 data
                                    print(f"Image {i} appears to be base64 data")
                                    save_base64_image(img_data, output_filename)
                            return output
                        
                        # Check for 'image' field
                        if isinstance(output, dict) and 'image' in output:
                            print("Found 'image' field in output")
                            save_base64_image(output['image'], output_filename)
                            return output
                        
                        # Check for node output fields used by ComfyUI
                        if isinstance(output, dict) and 'node_outputs' in output:
                            node_outputs = output['node_outputs']
                            print(f"Found node_outputs: {node_outputs.keys() if isinstance(node_outputs, dict) else 'Not a dictionary'}")
                            
                            # SaveImage node usually has output in node 9
                            if isinstance(node_outputs, dict) and '9' in node_outputs:
                                print("Found output from SaveImage node (9)")
                                save_data = node_outputs['9']
                                
                                if isinstance(save_data, list) and len(save_data) > 0:
                                    for i, img_data in enumerate(save_data):
                                        if isinstance(img_data, str):
                                            save_base64_image(img_data, output_filename)
                                        elif isinstance(img_data, dict) and 'filename' in img_data:
                                            print(f"SaveImage returned filename: {img_data['filename']}")
                                            # If there's actual image data
                                            if 'data' in img_data:
                                                save_base64_image(img_data['data'], output_filename)
                                        elif isinstance(img_data, list):
                                            for j, inner_img in enumerate(img_data):
                                                save_base64_image(inner_img, output_filename)
                                else:
                                    save_base64_image(save_data, output_filename)
                                return output
                                
                            # VAEDecode node usually has output in node 8
                            if isinstance(node_outputs, dict) and '8' in node_outputs:
                                print("Found output from VAEDecode node (8)")
                                vae_data = node_outputs['8']
                                
                                if isinstance(vae_data, list) and len(vae_data) > 0:
                                    for i, img_data in enumerate(vae_data):
                                        save_base64_image(img_data, output_filename)
                                else:
                                    save_base64_image(vae_data, output_filename)
                                return output
                                
                        # If we have a simple string, try to decode it as image
                        if isinstance(output, str):
                            print("Output is a direct string, trying to decode as image")
                            save_base64_image(output, output_filename)
                            return output
                            
                        # If output is a dictionary with a single value, try that value
                        if isinstance(output, dict) and len(output) == 1:
                            key = list(output.keys())[0]
                            value = output[key]
                            print(f"Output is a dictionary with a single key '{key}', trying its value")
                            if isinstance(value, str):
                                save_base64_image(value, output_filename)
                            elif isinstance(value, list) and len(value) > 0:
                                for i, item in enumerate(value):
                                    save_base64_image(item, output_filename)
                            return output
                            
                        # As a last resort, try saving the entire output
                        print("No standard image format found, saving entire output as file")
                        with open(os.path.join(OUTPUT_DIR, "output.json"), "w") as f:
                            json.dump(output, f, indent=2)
                        
                        print("No standard image fields found in output. Output saved to output.json")
                        print(json.dumps(output, indent=2))
                        
                except Exception as processing_error:
                    print(f"Error processing output: {processing_error}")
                    # Save the raw output for debugging
                    try:
                        with open(os.path.join(OUTPUT_DIR, "raw_output.json"), "w") as f:
                            json.dump(status_data, f, indent=2)
                        print(f"Raw output saved to {os.path.join(OUTPUT_DIR, 'raw_output.json')}")
                    except:
                        pass
                
                return output
            
            elif status in ["FAILED", "ERROR"]:
                error_msg = status_data.get('error', 'Unknown error')
                print(f"Job failed with error: {error_msg}")
                return None
            
            # Wait 5 seconds before checking again
            time.sleep(5)
    else:
        print(f"Failed to submit job: {response.text}")
        return None

def load_prompts_from_file(filename="message.txt"):
    """Load prompts from the message.txt file"""
    try:
        # Use hardcoded path to message.txt in the src/utils directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        message_path = os.path.join(script_dir, "message.txt")
        print(f"Looking for message.txt at: {message_path}")
        
        with open(message_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File {message_path} not found")
        return []
    except json.JSONDecodeError:
        print(f"Error: File {message_path} contains invalid JSON")
        return []
    except Exception as e:
        print(f"Error loading prompts from file: {e}")
        return []

def main():
    # First check if the endpoint is healthy
    print("Checking endpoint health...")
    if not check_endpoint_health():
        print("Endpoint health check failed. Please check your endpoint ID and API key.")
        return
    
    # Show output directory information
    abs_output_dir = os.path.abspath(OUTPUT_DIR)
    print(f"Images will be saved to: {abs_output_dir}")
    
    # Load prompts from file
    prompts_data = load_prompts_from_file()
    
    if not prompts_data:
        print("No prompts found in message.txt file or file couldn't be loaded.")
        use_default = input("Use default workflow instead? (y/n): ").lower() == 'y'
        
        if use_default:
            print("Using default settings.")
            # Generate the content with default prompts
            result = generate_image()
            
            if result:
                print("Content generation successful!")
                print(f"Check {abs_output_dir} for your generated images")
            else:
                print("Content generation failed.")
        return
    
    # Process each prompt
    print(f"Found {len(prompts_data)} prompts to process")
    
    for item in prompts_data:
        beat_number = item.get('beat_no')
        positive_prompt = item.get('positive_prompt')
        
        if not positive_prompt:
            print(f"Skipping beat {beat_number}: No positive prompt found")
            continue
        
        print(f"\nProcessing beat {beat_number}...")
        print(f"Positive prompt: {positive_prompt[:100]}...")
        
        # Generate image with this prompt
        output_filename = f"beat{beat_number}.png"
        print(f"Generating image: {output_filename}")
        
        result = generate_image(
            prompt=positive_prompt, 
            negative_prompt="", 
            output_filename=output_filename
        )
        
        if result:
            print(f"Successfully generated image for beat {beat_number}")
        else:
            print(f"Failed to generate image for beat {beat_number}")
    
    print("\nAll prompts processed!")
    print(f"Check {abs_output_dir} for your generated images")

if __name__ == "__main__":
    main()

