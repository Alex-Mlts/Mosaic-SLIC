import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color, feature
from skimage.segmentation import slic, mark_boundaries
from scipy.ndimage import gaussian_filter

from tkinter import Tk, Label, Button, filedialog

def image2mosaic(img_path, num = 100, compact = 10, blur = False, edges = False, boundaries = False):
    
    img = io.imread(img_path)
    
    if img.shape[-1] == 4:
        img = color.rgba2rgb(img)
    
    segments = slic(img, n_segments=num, compactness=compact, start_label=1)
    
    mosaic = np.zeros_like(img)
    
    for label in np.unique(segments):
        mask = segments == label
        mean_color = img[mask].mean(axis=0)
        mosaic[mask] = mean_color
        
    if blur:
        mosaic = gaussian_filter(mosaic, sigma=1)
        
    if edges:
        edges = feature.canny(color.rgb2gray(img))
        mosaic[edges] = [0, 0, 0]
        
    if boundaries:
        mosaic = mark_boundaries(mosaic, segments, color=(0, 0, 0), mode='outer')
        
    return mosaic

def main():
    
    root = Tk()
    root.title("Mosaic Image Creator")
    root.geometry("300x150")
    
    def browse_file():
        img_path = filedialog.askopenfilename(
            title="Select an Image"
        )
        
        if not img_path:
            return
        
        root.withdraw()
        process(img_path)
        root.destroy()
        
    label = Label(root, text="Select an image to create mosaic variants", wraplength=250, pady=20)
    label.pack()
    
    browse_button = Button(root, text="Browse Image", command=browse_file)
    browse_button.pack(pady=20)
    
    root.mainloop()
    
def process(img_path):

    superpixels = [100, 200, 300, 500]
            
    original = io.imread(img_path)
    fig, axes = plt.subplots(1, len(superpixels) + 1, figsize=(20, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
            
    for i, num in enumerate(superpixels):
        mosaic = image2mosaic(img_path, num=num, compact=10, blur=False, edges=False, boundaries=True)
        axes[i + 1].imshow(mosaic)
        axes[i + 1].set_title(f"{num} Superpixels")
        axes[i + 1].axis('off')
            
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()