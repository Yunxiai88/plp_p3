import os
import cv2
import argparse
import numpy as np
from PIL import Image
from skimage import transform

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    clahe = cv2.createCLAHE()
    
    lab[...,0] = clahe.apply(lab[...,0])
    
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_skimage_resize(image):
    small = transform.resize(
        image=image,
        output_shape = (756, 1344, image.shape[2]))
    print(small.shape)
    
    return (small*255).astype(np.uint8)

def apply_image_resize(IMAGE_DIR, fileName):
    image = Image.open(os.path.join(IMAGE_DIR, fileName))
    
    width, height = image.size
    
    #change size
    return image.resize((1344, 756))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="./", type=str, required=False, help="image path")
    
    args = parser.parse_args()
    print("args:\n" + args.__repr__())
    
    file_path = args.path
    
    for file in os.listdir(file_path):
        if not os.path.exists(file_path + '\\output'):
            os.makedirs(file_path + '\\output')
        
        if file == "output":
            continue
        
        image_file = os.path.join(file_path, file)
        print('processing: ' + image_file)
        
        out_file = os.path.join(file_path + "\\output", os.path.basename(file))
        
        # option 1
        input = cv2.imread(image_file)
        out = apply_skimage_resize(input)
        cv2.imwrite(out_file, out)
        
        # option 2
        #out = apply_image_resize(file_path, file)
        #out.save(out_file)