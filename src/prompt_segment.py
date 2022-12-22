from torchvision import transforms
from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt
import cv2

def prompt_based_segmentation(img_path, seg_prompt, clipseg_model):

    with Image.open('img.png') as img:
        input_image = img.convert('RGB')
    # input_image = Image.open(img_path).convert('RGB')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((512, 512)),
    ])

    img = transform(input_image).unsqueeze(0)

    prompts = [seg_prompt]
    # predict

    with torch.no_grad():
        preds = clipseg_model(img.repeat(len(prompts),1,1,1), prompts)[0]

    filename = "mask.png"
    plt.imsave(filename,torch.sigmoid(preds[0][0]))

    img = cv2.imread(filename)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

    cv2.imwrite(filename, bw_image)
    cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)
    # Image.fromarray(bw_image).save(filename)

    return filename