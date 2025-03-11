import base64
import tempfile
import ollama
import sys
import os

def encode_image(image_path):
    """Convert an image to base64 encoding."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_story(image_paths, story_context, story_style):
    """Generate a short story using LLaVA with given images and context."""
    
    if not image_paths:
        print("Error: No images provided!")
        return

    # Prepare vision prompt
    vision_prompt = [
        {"type": "text", "text": f"Create a {story_style.lower()} story based on these images and context: {story_context}"}
    ]

    # Convert images to base64 and add to the prompt
    for path in image_paths:
        if os.path.exists(path):
            vision_prompt.append({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{encode_image(path)}"
            })
        else:
            print(f"Warning: Image not found - {path}")

    # Run LLaVA model
    try:
        response = ollama.chat(model="llava", messages=vision_prompt)
        return response['message']['content']
    except Exception as e:
        return f"Error generating story: {str(e)}"

if __name__ == "__main__":
    # Example usage: python script.py "context text" "nostalgic" image1.jpg image2.jpg
    if len(sys.argv) < 4:
        print("Usage: python script.py '<context>' '<style>' <image1> <image2> ...")
        sys.exit(1)

    story_context = sys.argv[1]
    story_style = sys.argv[2]
    image_paths = sys.argv[3:]

    # Generate and print story
    story = generate_story(image_paths, story_context, story_style)
    print("\nGenerated Story:\n")
    print(story)
