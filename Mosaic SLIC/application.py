import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, feature
from skimage.segmentation import slic, mark_boundaries
from scipy.ndimage import gaussian_filter
from tkinter import Tk, Label, Button, filedialog

def create_mosaic(image_path, n_segments=100, compactness=10, apply_blur=False, apply_edges=False, apply_boundaries=False):
    # Load the image
    image = io.imread(image_path)
    if image.shape[-1] == 4:
        image = color.rgba2rgb(image)  # Convert RGBA to RGB

    # Perform SLIC segmentation
    segments = slic(image, n_segments=n_segments, compactness=compactness, start_label=1)

    # Calculate the average color for each segment
    mosaic = np.zeros_like(image)
    for segment_label in np.unique(segments):
        mask = segments == segment_label
        mean_color = image[mask].mean(axis=0)
        mosaic[mask] = mean_color

    # Apply Gaussian blur if selected
    if apply_blur:
        mosaic = gaussian_filter(mosaic, sigma=1)
        
    # Apply edge detection if selected
    if apply_edges:
        edges = feature.canny(color.rgb2gray(image))
        mosaic[edges] = [0, 0, 0]  # Set edges to black

    # Mark boundaries if selected
    if apply_boundaries:
        mosaic = mark_boundaries(mosaic, segments, color=(0, 0, 0), mode='outer')
        
    return mosaic

def main():
    root = Tk()
    root.title("Mosaic Image Creator")
    root.geometry("300x150")
    
    def browse_file():
        image_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not image_path:
            return
            
        root.withdraw()  # Hide the window while showing the plot
        process_image(image_path)
        root.destroy()  # Close the window after showing the plot
    
    # Create and pack widgets
    label = Label(root, text="Select an image to create mosaic variants", wraplength=250, pady=20)
    label.pack()
    
    browse_button = Button(root, text="Browse Image", command=browse_file)
    browse_button.pack(pady=20)
    
    root.mainloop()
    
def process_image(image_path):
    
    segment_values = [100, 200, 300]

    # Display the original image
    original_image = io.imread(image_path)
    fig, ax = plt.subplots(1, len(segment_values) + 1, figsize=(18, 6))
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    # Loop over n_segments values and create mosaic images
    for i, n_segments in enumerate(segment_values):
        mosaic_image = create_mosaic(image_path, n_segments=n_segments, compactness=30, apply_blur=False, apply_edges=False, apply_boundaries=True)
        ax[i + 1].imshow(mosaic_image)
        ax[i + 1].set_title(f"Mosaic with {n_segments} Segments")
        ax[i + 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()