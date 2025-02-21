import torch
from diffusers import StableDiffusionPipeline
from google.colab import files

# Check if CUDA is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_id = "runwayml/stable-diffusion-v1-5"
# Use the device variable to specify where to load the model
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
pipe.safety_checker = None

prompt = "Beautiful Woman, 20 Years old, piercing green eyes, long curly brown hair, flawless skin, wet body dripping, intimate sexy pose"
variations = [
  "With slightly different lighting",
  "and expression changed slightly",
  "with hair styled differently",
  "wearing alternative outfit" ,
  "in a different pose" ,
  "legs spread wide open against velvet curtains",
  "sitting on plush couch with legs opened",
  "standing in doorway with breast exposed seductive  smile",
  "lying on satin sheets finger pussy soft gaze",
  "posing naked in a stunning luxurious window an erotic backdrop"
]

for i, variation in enumerate(variations):
  variation_prompt = prompt + ", " + variation
  # Use the device variable to specify where to run the model
  with torch.autocast(device):
    image = pipe(variation_prompt).images[0]
  image.save(f"elegant_woman_variation_{i}.png")
  files.download(f"elegant_woman_variation_{i}.png")
  print(f"Image variation {i} generated, saved, and downloaded.")

i = 0
while True:
    variation_prompt = prompt + ", " + variations[i % len(variations)]
    # Use the device variable to specify where to run the model
    with torch.autocast(device):
        image = pipe(variation_prompt).images[0]
    # Don't save or display images
    # image.save(f"elegant_woman_variation_{i}.png")
    # files.download(f"elegant_woman_variation_{i}.png")
    print(f"Image variation {i} generated.")
    i += 1
