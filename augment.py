import os
import json
import random
import nltk
from PIL import Image, ImageEnhance
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')

# Function to augment transparent images
def augment_transparent_image(image):
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    alpha = image.getchannel('A')
    angle = random.uniform(-60, 60)
    image = image.rotate(angle, resample=Image.BICUBIC, expand=True)
    alpha = alpha.rotate(angle, resample=Image.BICUBIC, expand=True)

    width_shift = random.uniform(-0.4, 0.4) * image.size[0]
    height_shift = random.uniform(-0.4, 0.4) * image.size[1]
    image = Image.Image.transform(
        image,
        (image.size[0], image.size[1]),
        Image.AFFINE,
        (1, 0, width_shift, 0, 1, height_shift),
        resample=Image.BICUBIC,
    )
    alpha = Image.Image.transform(
        alpha,
        (alpha.size[0], alpha.size[1]),
        Image.AFFINE,
        (1, 0, width_shift, 0, 1, height_shift),
        resample=Image.BICUBIC,
    )

    image.putalpha(alpha)
    zoom_factor = random.uniform(0.7, 1.5)
    w, h = image.size
    image = image.resize((int(w * zoom_factor), int(h * zoom_factor)), Image.LANCZOS)
    enhancer = ImageEnhance.Brightness(image)
    brightness_factor = random.uniform(0.7, 1.3)
    image = enhancer.enhance(brightness_factor)
    image = image.resize((150, 150), Image.Resampling.LANCZOS)
    return image

# Function to generate a new caption using synonyms
def generate_caption(caption):
    tokens = word_tokenize(caption)
    new_caption = []

    for token in tokens:
        # Get possible synonyms from WordNet
        synonyms = set()
        for syn in wn.synsets(token):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        
        # Remove the original word from synonyms if it's present
        if token in synonyms:
            synonyms.remove(token)
        
        # Randomly pick a synonym or keep the original word
        if synonyms:
            new_token = random.choice(list(synonyms))
        else:
            new_token = token
        
        new_caption.append(new_token)

    # Join the tokens back into a sentence
    return " ".join(new_caption)

# Path to the JSON file
json_file_path = '/projectnb/cs505aw/students/tsmuskin/Specialist-Diffusion-main/data/emoji/source_list.json'

# Load the JSON dictionary
with open(json_file_path, 'r') as f:
    img_dict = json.load(f)

# Directory to save augmented images
save_dir = '/projectnb/cs505aw/students/tsmuskin/Specialist-Diffusion-main/data/emoji/emoji-images'
os.makedirs(save_dir, exist_ok=True)

new_images = {}
# Process each image in the JSON dictionary
for key, details in img_dict.items():
    try:
        # Extract the image path from the nested dictionary
        img_path = details.get('image_path')
        if not img_path:
            print(f"Missing image_path for key: {key}")
            continue

        # Load the image
        img = Image.open(img_path)
        
        # augmented_images = []
        # Generate augmented images
        for i in range(10):  # Adjust the range for more/less augmentations
            augmented_img = augment_transparent_image(img)
            
            # Generate a unique output name
            img_name = os.path.basename(img_path)
            output_name = f"{os.path.splitext(img_name)[0]}_{key}_aug_{i}.png"
            output_path = os.path.join(save_dir, output_name)
            
            # Save the augmented image
            augmented_img.save(output_path, format="PNG")
            print(f"Saved augmented image: {output_path}")

        
            # Add the augmented images to the source_list
            new_key = f"{os.path.splitext(img_name)[0]}_{key}_aug_{i}.png"
            new_caption = generate_caption(details["caption"])
            new_images[new_key] = {
                "image_path": output_path,
                "caption": new_caption
            }
    
    except Exception as e:
        print(f"Error processing {details}: {e}")

# Update the original dictionary with augmented images
img_dict.update(new_images)

# Save the updated source_list to the JSON file
with open(json_file_path, 'w') as f:
    json.dump(img_dict, f, indent=4)
    print("Updated source list with augmented images.")

