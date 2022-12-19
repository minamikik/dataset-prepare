:: python .\tagging_image.py --source ../dataset/test/Normal --model "ViT-H-14/laion2b_s32b_b79k"
:: python .\tagging_image.py --source ../dataset/test/Normal --model "xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k"
:: python .\tagging_image.py --source ../dataset/test/Normal --model "ViT-L-14/openai"
:: python .\tagging_image.py --source ../dataset/test/Normal --model "deepdanbooru"

:: python .\tagging_image.py --source ../dataset/test/Alt --model "ViT-H-14/laion2b_s32b_b79k" --alt_flavors "./alt_caption_data/high_tech_flavors.txt"
:: python .\tagging_image.py --source ../dataset/test/Alt --model "xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k" --alt_flavors "./alt_caption_data/high_tech_flavors.txt"
:: python .\tagging_image.py --source ../dataset/test/Alt --model "ViT-L-14/openai" --alt_flavors "./alt_caption_data/high_tech_flavors.txt"
:: python .\tagging_image.py --source ../dataset/test/Alt --model "deepdanbooru"

:: python .\tagging_image.py --source ../dataset/test/person/a --model "ViT-H-14/laion2b_s32b_b79k"
:: python .\tagging_image.py --source ../dataset/test/person/a --model "xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k"
:: python .\tagging_image.py --source "../dataset/source/japanese beauty/girl/Kanna Hashimoto" --model "ViT-H-14/laion2b_s32b_b79k" --person --force
:: python .\tagging_image.py --source ../dataset/test/person/b --model "xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k" --person "./alt_caption_data/person.txt" --age 20 --force

:: python .\tagging_image.py --source "../dataset/source/japanese beauty/girl/Kanna Hashimoto" --model "ViT-H-14/laion2b_s32b_b79k" --person --force
python .\tagging_image.py --source "../dataset\source\nsfw\a japanese woman having sex in missionary position" --model "ViT-H-14/laion2b_s32b_b79k" --force

