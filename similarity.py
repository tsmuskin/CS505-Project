import torch
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Example captions
captions = ["grinning face with big eyes", "skull and crossbones", "crying cat", "see-no-evil monkey", "red heart", "shooting star", " eyes", "woman's face" "rock hand symbol", "a man artist", "a man running", "person mountain biking", "two women holding hands.", "family: man, woman, girl, girl.", "fox.", "palm tree.", "strawberry", "popcorn", "bottle with popping cork.",  "globe showing Europe-Africa.", "desert island.", "bank.", "sunrise.", "fire engine.", "motorway", "stopwatch.", "clock with hands at eight-thirty.", "crescent moon.", "umbrella with rain drops.", "party popper.", "video game controller.", "ballet shoes.", "musical notes.", "laptop."  ]
inputs = processor(text=captions, return_tensors="pt", padding=True)
outputs = model.get_text_features(**inputs)

# Save precomputed embeddings
torch.save(outputs, "similars/emoji-similars.pt")
