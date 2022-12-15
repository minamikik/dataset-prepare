import torch
from transformers import CLIPModel, CLIPProcessor
from src.aesthetic.aesthetic import image_embeddings, Classifier

aesthetic_path = 'models/aesthetic/aes-B32-v0.pth'
clip_name = 'openai/clip-vit-base-patch32'
url = 'https://cdn.donmai.us/original/16/67/__klein_moretti_lord_of_the_mysteries_drawn_by_ji26725339__1667415282975e8f8c574ca26d83e3be.jpg'

clipprocessor = CLIPProcessor.from_pretrained(clip_name)
clipmodel = CLIPModel.from_pretrained(clip_name).to('cuda').eval()

aes_model = Classifier(512, 256, 1)
aes_model.load_state_dict(torch.load(aesthetic_path))
torch.set_default_tensor_type('torch.cuda.FloatTensor')

image_embeds = image_embeddings(url, clipmodel, clipprocessor)
prediction = aes_model(torch.from_numpy(image_embeds).float().to('cuda'))
print(f'Prediction: {prediction.item()}')