import cv2
import matplotlib.pyplot as plt

image = cv2.imread('./Dataset/Chica.png', 1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_xyz = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)

image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

fig = plt.figure(figsize=(10, 20))
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(image_xyz)
ax1.set_title('XYZ')
ax1.set_xticks([])
ax1.set_yticks([])
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(image_lab)
ax2.set_title('L*a*b')
ax2.set_xticks([])
ax2.set_yticks([])
plt.show()