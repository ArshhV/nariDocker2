#!/usr/bin/env python3

import runpod
from deepseek_vl2 import DeepSeekVL2

# Global model loaded once on cold start
MODEL = DeepSeekVL2.load("deepseek-ai/deepseek-vl2-small")

def handler(event, context):
    """
    RunPod handler for DeepSeek-VL2 inference
    """
    runpod.log("Handler start - Received request")
    try:
        # Extract inputs
        input_data = event["input"]
        prompt = input_data.get("prompt")
        images = input_data.get("images", [])
        system = input_data.get("system")
        temperature = input_data.get("temperature", 0.2)
        max_tokens = input_data.get("max_tokens", 1000)

        if not prompt or not images:
            return {"error": "Missing required inputs: prompt and images"}

        # Run inference
        runpod.log("Running inference")
        result = MODEL.infer(
            prompt=prompt,
            images=images,
            system=system,
            temperature=temperature,
            max_new_tokens=max_tokens
        )
        
        runpod.log("Inference complete")
        return {"output": result}

    except Exception as e:
        runpod.log(f"Error in handler: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})