from PIL import Image
import os
from tqdm import tqdm

'''source'''
rgb_dir = '/data/NeRF/100imgperclass-400x400'
output_dir = '/data/NeRF/100imgperclass-100x100'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for root, dirs, files, in os.walk(rgb_dir):
    for file in files:
        if '.png' in file:
            os.makedirs(root.replace(rgb_dir, output_dir), exist_ok=True)
            file_path = os.path.join(root, file)
            rgb_im = Image.open(file_path)
            rgb_im = rgb_im.resize((100, 100))
            rgb_im.save(file_path.replace(rgb_dir, output_dir))

# imgs= sorted(os.listdir(rgb_dir))
#
# for i, img_path in tqdm(enumerate(imgs)):
#     if i >= 50: break
#     im = Image.open(os.path.join(rgb_dir, img_path))
#     rgb_im = im.convert('RGB')
#     rgb_im = rgb_im.resize((100, 100))
#     rgb_im.save(os.path.join(output_dir, img_path))
