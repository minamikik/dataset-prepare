from PIL import Image
from clip_interrogator import Interrogator, Config
image = Image.open("../dataset/test/b.png").convert('RGB')
ci = Interrogator(Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k"))
print(ci.interrogate(image))



