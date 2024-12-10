import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

depth_mask = "/mydata/data/hyunsoo/co3d_sample/apple/12_90_489/depth_masks/frame000001.png"
depth_image = "/mydata/data/hyunsoo/co3d_sample/apple/12_90_489/depths/frame000001.jpg.geometric.png"
depth_mask = Image.open(depth_mask)
depth_image = Image.open(depth_image)

depth_mask = np.array(depth_mask)
depth_image = np.array(depth_image)
depth_image = depth_image * depth_mask

depth_image_show = Image.fromarray(depth_image)
plt.imshow(depth_image_show)

print(max(depth_image.flatten()))
print(min(depth_image.flatten())) 
