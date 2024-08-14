import os
import shutil
import time
from flask import Flask, render_template, request, jsonify
from gradio_client import Client

app = Flask(__name__)

# Path to save the generated images
IMAGE_SAVE_PATH = 'static/images/'

# Ensure the directory exists
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

# Initialize the Gradio client with the API endpoint
client = Client("black-forest-labs/FLUX.1-schnell")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        # Call the image generation API with the user's input
        image_path, _ = call_image_generation_api(prompt)
        return render_template('index.html', image_url=image_path)
    return render_template('index.html', image_url=None)

def call_image_generation_api(prompt):
    # Use the gradio_client to make the API request
    result = client.predict(
        prompt=prompt,
        seed=0,  # Default seed value
        randomize_seed=True,
        width=1024,
        height=1024,
        num_inference_steps=4,
        api_name="/infer"
    )

    # The API returns a file path to the generated image
    image_filepath = result[0]

    # Generate a unique filename based on timestamp
    image_filename = f"generated_{int(time.time())}_{os.path.basename(image_filepath)}"

    # Define the save path in the static/images directory
    save_path = os.path.join(IMAGE_SAVE_PATH, image_filename)

    # Copy the file to the static directory
    shutil.copy(image_filepath, save_path)

    # Return the relative path for the image to be used in the frontend
    return f'{save_path}', result[1]

@app.route('/history', methods=['GET'])
def get_history():
    # Get all images in the IMAGE_SAVE_PATH directory
    images = [f'static/images/{img}' for img in os.listdir(IMAGE_SAVE_PATH) if img.endswith('.webp')]
    return jsonify(images)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81)
