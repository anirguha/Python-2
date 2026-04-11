import pygame
import numpy as np
import matplotlib.pyplot as plt

# Initialize PyGame and the display
pygame.init()
width, height = 600, 600
screen = pygame.Surface((width, height))  # Off-screen drawing surface

# Colors
def color_gradient(start_color, end_color, steps):
    return [
        (
            int(start_color[0] + (end_color[0] - start_color[0]) * i / (steps - 1)),
            int(start_color[1] + (end_color[1] - start_color[1]) * i / (steps - 1)),
            int(start_color[2] + (end_color[2] - start_color[2]) * i / (steps - 1))
        )
        for i in range(steps)
    ]

# Drawing parameters
num_squares = 20
max_size = 500
min_size = 50
center = width // 2, height // 2
colors = color_gradient((255, 0, 0), (0, 0, 255), num_squares)

# Draw on off-screen surface
screen.fill((255, 255, 255))  # White background
for i in range(num_squares):
    size = max_size - i * (max_size - min_size) // (num_squares - 1)
    rect = pygame.Rect(0, 0, size, size)
    rect.center = center
    pygame.draw.rect(screen, colors[i], rect, width=3)

# Convert PyGame surface to NumPy array and display with matplotlib
image = pygame.surfarray.array3d(screen)
image = np.transpose(image, (1, 0, 2))  # Convert from (width, height, channel) to (height, width, channel)

plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis('off')
plt.title('Concentric Squares (PyGame Rendered in Jupyter)')
plt.show()
