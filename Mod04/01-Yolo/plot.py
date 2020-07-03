from PIL import Image
import matplotlib.pyplot as plt

img_path = '../../data/object_detection/000863.jpg'

plt.figure()
image = Image.open(img_path)
#fig = plt.figure(figsize=(12, 5))
#fig = plt.figure()
fig, ax = plt.subplots(1, figsize=(10,5))
ax.imshow(image)
plt.show()
    
plt.imshow(image)
#plt.title('2')
plt.show()
