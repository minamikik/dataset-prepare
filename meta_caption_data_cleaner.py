
import os
import json
from typing import List


source_dir = 'captions_data\source'
output_dir = 'captions_data\output'
os.makedirs(output_dir, exist_ok=True)

def create_persons_list() -> None:
    utils_colors = _load_list(os.path.join(source_dir, 'utils_colors.txt'))

    hairs =[]
    hairs_length = ['short', 'long']
    hairs_colors = _load_list(os.path.join(source_dir, 'hairs_colors.txt'))
    hairs_styles_seed = _load_list(os.path.join(source_dir, 'hairs_styles_seeds.txt'))

    hairs.extend([f"{l} hair" for l in hairs_length])
    hairs.extend([f"{s} hair" for s in hairs_styles_seed])
    hairs.extend([f"{c} hair" for c in hairs_colors])
    hairs.extend([f"{l} {s} hair" for l in hairs_length for s in hairs_styles_seed])
    hairs.extend([f"{c} {l} hair" for l in hairs_length for c in hairs_colors])
    hairs.extend([f"{l} {c} hair" for l in hairs_length for c in hairs_colors])
    hairs.extend([f"{c} {s} hair" for s in hairs_styles_seed for c in hairs_colors])
    hairs.extend([f"{s} {c} hair" for s in hairs_styles_seed for c in hairs_colors])
    hairs.extend([f"{c} {l} {s} hair" for s in hairs_styles_seed for c in hairs_colors for l in hairs_length])
    hairs.sort()

    hairs_styles = []
    raw_hairs_styles = _combine_lists(hairs_styles_seed, hairs_styles_seed)
    hairs_styles.extend([f"with {r} hair cut" for r in raw_hairs_styles])
    hairs_styles.extend([f"with {r} hair style" for r in raw_hairs_styles]) 


#    _output_list(hairs_colors, 'new_hairs_colors.txt')
#    _output_list(hairs_styles_seed, 'new_hairs_styles_seeds.txt')
    _output_list(hairs_styles, 'new_hairs_styles.txt')
    _output_list(hairs, 'new_hairs.txt')



    persons_adjectives = _load_list(os.path.join(source_dir, 'persons_adjectives.txt'))
    person_type = ['girl', 'woman', 'boy', 'man', 'person']
    persons_type = ['girls', 'women', 'boys', 'men', 'people']
    persons = []
    persons.extend([f"a {a} {p}" for a in persons_adjectives for p in person_type])
    persons.extend([f"{a} {p}" for a in persons_adjectives for p in persons_type])
    persons.extend([f"a {a} {p} with {uc} eyes" for a in persons_adjectives for p in person_type for uc in utils_colors])

    _output_list(persons, 'new_persons.txt')

    clothes_outfit = _load_list(os.path.join(source_dir, 'clothes_outfit.txt'))
    clothes_tops = _load_list(os.path.join(source_dir, 'clothes_tops.txt'))
    clothes_bottoms = _load_list(os.path.join(source_dir, 'clothes_bottoms.txt'))
    clothes_misc = _load_list(os.path.join(source_dir, 'clothes_misc.txt'))

    clothes = []
    clothes.extend([f"in {co}" for co in clothes_outfit])
    clothes.extend([f"in {ct}" for ct in clothes_tops])
    clothes.extend([f"in {cb}" for cb in clothes_bottoms])
    clothes.extend([f"with {cm}" for cm in clothes_misc])
    clothes.extend([f"in {uc} {co}" for uc in utils_colors for co in clothes_outfit])
    clothes.extend([f"in {uc} {ct}" for uc in utils_colors for ct in clothes_tops])
    clothes.extend([f"in {uc} {cb}" for uc in utils_colors for cb in clothes_bottoms])
    clothes.extend([f"with {uc} {cm}" for uc in utils_colors for cm in clothes_misc])
    clothes.extend([f"in {co} and {ct}" for co in clothes_outfit for ct in clothes_tops])
    clothes.extend([f"in {co} and {cb}" for co in clothes_outfit for cb in clothes_bottoms])
    clothes.extend([f"in {ct} and {cb}" for ct in clothes_tops for cb in clothes_bottoms])
    clothes.extend([f"in {co} and {ct} and {cb}" for co in clothes_outfit for ct in clothes_tops for cb in clothes_bottoms])
    clothes.extend([f"in {ct} and {cb} with {cm}" for ct in clothes_tops for cb in clothes_bottoms for cm in clothes_misc])

#    _output_list(persons_adjectives, 'new_persons_adjectives.txt')
#    _output_list(clothes_outfit, 'new_clothes_outfit.txt')
#    _output_list(clothes_tops, 'new_clothes_tops.txt')
#    _output_list(clothes_bottoms, 'new_clothes_bottoms.txt')
#    _output_list(clothes_misc, 'new_clothes_misc.txt')
    _output_list(clothes, 'new_clothes.txt')

    photos = []
    photos_photographers = _load_list(os.path.join(source_dir, 'photos_photographers.txt'))
    photos_cameras = _load_list(os.path.join(source_dir, 'photos_cameras.txt'))
    photos_lenses = _load_list(os.path.join(source_dir, 'photos_lenses.txt'))
 

#    _output_list(photos_photographers, 'new_photos_photographers.txt')
#    _output_list(photos_cameras, 'new_photos_cameras.txt')
#    _output_list(photos_lenses, 'new_photos_lenses.txt') 

    nsfw_positions = _load_list(os.path.join(source_dir, 'nsfw_positions.txt'))
    raw_nsfw_actions = _load_list(os.path.join(source_dir, 'nsfw_actions.txt'))
    nsfw_seme = [
        ['boy', 'he', 'him'],
        ['boys', 'they', 'their'],
        ['man', 'he', 'him'],
        ['men', 'they', 'their'],
        ['people', 'they', 'their']
    ]
    nsfw_uke = [
        ['girl', 'she', 'her'],
        ['girls', 'they', 'their'],
        ['woman', 'she', 'her'],
        ['women', 'they', 'their'],
        ['lady', 'she', 'her'],
        ['ladies', 'they', 'their']
    ]
    nsfw_p = ['penis', 'phallus', 'dick', 'prick', 'cock', 'wang']
    nsfw_actions = []
    nsfw_actions.extend([f"{na.replace('[seme]', ns[0]).replace('[seme2]', ns[1]).replace('[seme3]', ns[2]).replace('[uke]', nu[0]).replace('[uke2]', nu[1]).replace('[uke3]', nu[2]).replace('[seme+]', np)}" for na in raw_nsfw_actions for nu in nsfw_uke for ns in nsfw_seme for np in nsfw_p])



    _output_list(nsfw_positions, 'new_nsfw_positions.txt')
    _output_list(nsfw_actions, 'new_nsfw_actions.txt')


def _output_list(list, file_name):
    with open(os.path.join(output_dir, file_name), 'w', encoding='utf-8', errors='replace') as f:
        for item in list:
            f.write(f"{item}\n")


def _combine_lists(list1, list2):
    combined_list = []
    for i in list1:
        for j in list2:
            if not i == j:
                combined_text = i + ' ' + j
                combined_list.append(combined_text)
    return combined_list


def _load_list(file_path) -> List[str]:
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        items = [line.strip() for line in f.readlines()]
        items = set(items)
        items = list(items)
        items.sort()
    return items


def main():
    create_persons_list()

if __name__ == '__main__':
    main()