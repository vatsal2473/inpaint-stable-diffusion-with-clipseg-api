from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from flask_cors import CORS, cross_origin
from src import load_models, helper_functions, prompt_segment, inpaint
import requests
import base64
from PIL import Image
import io

app = Flask(__name__)
cors = CORS(app)

# Load models
pipe = load_models.load_stable_diffusion_model()
clipseg_model = load_models.load_clipseg()

def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = base64.encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

@app.route('/clipseg-inpaint', methods=['POST'])
def clipseg_inpaint():
    img_url = request.form['img_url']
    seg_prompt = request.form['seg_prompt']
    prompt = request.form['prompt']
    sample = request.form.get('sample', False)

    # Download image
    img_path = 'img.png'
    helper_functions.get_image_from_url(img_url, img_path)

    try:
        mask_filename = prompt_segment.prompt_based_segmentation(img_path, seg_prompt, clipseg_model)
        images = inpaint.stable_diffusion_inpaint(prompt, pipe, img_path, mask_filename)
        

        if sample == True:
            li = []
            for i, img in enumerate(images):
                name = 'output/image{0}.png'.format(i)
                img.save(name)
                # convert to base64

                encoded_img = get_response_image(name)
        
                li.append(encoded_img)
            
            res = {}
            res['base64'] = li
            return res

        else:

            li = []
            for i, img in enumerate(images):
                name = 'output/image{0}.png'.format(i)
                img.save(name)
            # grid = helper_functions.image_grid(images, 1, 3)
            # grid.save('output/grid.png')

                files = {
                    'file': open(name, 'rb'),
                }

                response = requests.post('https://tmpfiles.org/api/v1/upload', files=files)
                li.append(response.json()['data']['url'])

            res = {}
            res['url'] = li
            return res

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run()