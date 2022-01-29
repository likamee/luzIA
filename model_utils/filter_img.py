from PIL import Image, ImageFilter, ImageEnhance
import cv2
import random
import shutil
import os
import numpy as np
import skimage
import skimage.feature
import skimage.viewer
from skimage.filters import gaussian
import matplotlib.pyplot as plt

def process_filters(SOURCE, FILENAME, METHOD, PATO = ''):
    TYPE = 'alteradas' if PATO else 'normais'
    FOLDER = PATO if PATO else '/'+TYPE

    """ if METHOD == 'raw':
        shutil.copy(os.path.join(SOURCE+TYPE+PATO, FILENAME), SOURCE+'algo'+FOLDER)
    else:       """

    #IMAGEGS = cv2.imread(os.path.join(SOURCE+TYPE+PATO, FILENAME), 0)
    IMAGE = cv2.imread(os.path.join(SOURCE+TYPE+PATO, FILENAME))
    #IMAGEGS = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2GRAY)
    IMAGEGS = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2GRAY)
    if check_bright(IMAGEGS): #or check_blurry(IMAGEGS):
        cv2.imwrite(SOURCE+'excluidas/'+FILENAME, IMAGE)
        return False
    IMAGE = globals()[METHOD](IMAGEGS)
    #IMAGE = globals()['unsharp'](IMAGE)
    
    cv2.imwrite(SOURCE+'algo'+FOLDER+'/'+FILENAME, IMAGE)
    #IMAGE.save(SOURCE+'algo'+FOLDER+'/'+FILENAME)

    return True
    #IMAGE.save(SOURCE+'algo/normais/'+FILENAME)
    #shutil.copy(os.path.join(SOURCE+'normais', FILENAME), SOURCE+'algo/normais')
   



def check_blurry(IMAGE):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    THRESHOLD = 150
    fm = cv2.Laplacian(IMAGE, cv2.CV_64F).var()
    return fm < THRESHOLD

def check_bright(IMAGE):
    THRESHOLDS = [145, 15] # 135 35
    is_light = np.mean(IMAGE) > THRESHOLDS[0]
    is_dark = np.mean(IMAGE) < THRESHOLDS[1]
    return is_light or is_dark

def combo_filter(IMAGE):
    R, IMAGE, B = cv2.split(IMAGE)
    CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #2 #8
    IMAGE = CLAHE.apply(IMAGE)
    IMAGE = cv2.bitwise_not(IMAGE)
    return IMAGE

def reddit_constrast(IMAGE):
    clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8,8))
    lab = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2,a,b))  # merge channels
    IMAGE = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return IMAGE

def gaussian_blur(IMAGE):
    #image = np.array(img)
    image_blur = cv2.GaussianBlur(IMAGE,(65,65),10)
    # new_image = cv2.subtract(img,image_blur).astype('float32') # WRONG, the result is not stored in float32 directly
    new_image = cv2.subtract(IMAGE,image_blur, dtype=cv2.CV_32F)
    IMAGE = cv2.normalize(new_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return IMAGE

def hsv(IMAGE):
    IMAGE = cv2.cvtColor(IMAGE, cv2.COLOR_RGB2HSV)
    return IMAGE

def canny(IMAGE):
    image = skimage.feature.canny(
        image=IMAGE,
        sigma=3,
        low_threshold=2,
        high_threshold=10,
    )
    image = image * 255
    return image

def contrast(IMAGE):
    image = Image.fromarray(IMAGE.astype('uint8'))
    converter = ImageEnhance.Contrast(image)
    image = converter.enhance(1.5)
    return np.array(image)

def saturation(IMAGE):
    image = Image.fromarray(IMAGE.astype('uint8'))
    converter = ImageEnhance.Color(image)
    image = converter.enhance(1.5)
    return np.array(image)

def sharpness(IMAGE):
    image = Image.fromarray(IMAGE.astype('uint8'))
    converter = ImageEnhance.Sharpness(image)
    image = converter.enhance(1.5)
    return np.array(image)

def brightness(IMAGE):
    image = Image.fromarray(IMAGE.astype('uint8'))
    converter = ImageEnhance.Brightness(image)
    image = converter.enhance(1.5)
    return np.array(image)


def conservative_smoothing(IMAGE):
    filter_size = 9    
    
    data = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2GRAY)
    new_image = data.copy()

    temp = []
    indexer = filter_size // 2
    new_image = data.copy()
    nrow, ncol = data.shape
    for i in range(nrow):
        for j in range(ncol):
            for k in range(i-indexer, i+indexer+1):
                for m in range(j-indexer, j+indexer+1):
                    if (k > -1) and (k < nrow):
                        if (m > -1) and (m < ncol):
                            temp.append(data[k,m])
            temp.remove(data[i,j])
            max_value = max(temp)
            min_value = min(temp)
            if data[i,j] > max_value:
                new_image[i,j] = max_value
            elif data[i,j] < min_value:
                new_image[i,j] = min_value
            temp =[]
    
    return new_image

def crimmins(IMAGE):    
    data = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2GRAY)
    new_image = data.copy()


    nrow = len(data)
    ncol = len(data[0])
    
    # Dark pixel adjustment
    
    # First Step
    # N-S
    for i in range(1, nrow):
        for j in range(ncol):
            if data[i-1,j] >= (data[i,j] + 2):
                new_image[i,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(ncol-1):
            if data[i,j+1] >= (data[i,j] + 2):
                new_image[i,j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow):
        for j in range(1, ncol):
            if data[i-1,j-1] >= (data[i,j] + 2):
                new_image[i,j] += 1
    data = new_image
    #NE-SW
    for i in range(1, nrow):
        for j in range(ncol-1):
            if data[i-1,j+1] >= (data[i,j] + 2):
                new_image[i,j] += 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1, nrow-1):
        for j in range(ncol):
            if (data[i-1,j] > data[i,j]) and (data[i,j] <= data[i+1,j]):
                new_image[i,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol-1):
            if (data[i,j+1] > data[i,j]) and (data[i,j] <= data[i,j-1]):
                new_image[i,j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow-1):
        for j in range(1, ncol-1):
            if (data[i-1,j-1] > data[i,j]) and (data[i,j] <= data[i+1,j+1]):
                new_image[i,j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow-1):
        for j in range(1, ncol-1):
            if (data[i-1,j+1] > data[i,j]) and (data[i,j] <= data[i+1,j-1]):
                new_image[i,j] += 1
    data = new_image
    #Third Step
    # N-S
    for i in range(1, nrow-1):
        for j in range(ncol):
            if (data[i+1,j] > data[i,j]) and (data[i,j] <= data[i-1,j]):
                new_image[i,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol-1):
            if (data[i,j-1] > data[i,j]) and (data[i,j] <= data[i,j+1]):
                new_image[i,j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow-1):
        for j in range(1, ncol-1):
            if (data[i+1,j+1] > data[i,j]) and (data[i,j] <= data[i-1,j-1]):
                new_image[i,j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow-1):
        for j in range(1, ncol-1):
            if (data[i+1,j-1] > data[i,j]) and (data[i,j] <= data[i-1,j+1]):
                new_image[i,j] += 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(nrow-1):
        for j in range(ncol):
            if (data[i+1,j] >= (data[i,j]+2)):
                new_image[i,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1,ncol):
            if (data[i,j-1] >= (data[i,j]+2)):
                new_image[i,j] += 1
    data = new_image
    # NW-SE
    for i in range(nrow-1):
        for j in range(ncol-1):
            if (data[i+1,j+1] >= (data[i,j]+2)):
                new_image[i,j] += 1
    data = new_image
    # NE-SW
    for i in range(nrow-1):
        for j in range(1,ncol):
            if (data[i+1,j-1] >= (data[i,j]+2)):
                new_image[i,j] += 1
    data = new_image
    
    # Light pixel adjustment
    
    # First Step
    # N-S
    for i in range(1,nrow):
        for j in range(ncol):
            if (data[i-1,j] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(ncol-1):
            if (data[i,j+1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # NW-SE
    for i in range(1,nrow):
        for j in range(1,ncol):
            if (data[i-1,j-1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # NE-SW
    for i in range(1,nrow):
        for j in range(ncol-1):
            if (data[i-1,j+1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1,nrow-1):
        for j in range(ncol):
            if (data[i-1,j] < data[i,j]) and (data[i,j] >= data[i+1,j]):
                new_image[i,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol-1):
            if (data[i,j+1] < data[i,j]) and (data[i,j] >= data[i,j-1]):
                new_image[i,j] -= 1
    data = new_image
    # NW-SE
    for i in range(1,nrow-1):
        for j in range(1,ncol-1):
            if (data[i-1,j-1] < data[i,j]) and (data[i,j] >= data[i+1,j+1]):
                new_image[i,j] -= 1
    data = new_image
    # NE-SW
    for i in range(1,nrow-1):
        for j in range(1,ncol-1):
            if (data[i-1,j+1] < data[i,j]) and (data[i,j] >= data[i+1,j-1]):
                new_image[i,j] -= 1
    data = new_image
    # Third Step
    # N-S
    for i in range(1,nrow-1):
        for j in range(ncol):
            if (data[i+1,j] < data[i,j]) and (data[i,j] >= data[i-1,j]):
                new_image[i,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1,ncol-1):
            if (data[i,j-1] < data[i,j]) and (data[i,j] >= data[i,j+1]):
                new_image[i,j] -= 1
    data = new_image
    # NW-SE
    for i in range(1,nrow-1):
        for j in range(1,ncol-1):
            if (data[i+1,j+1] < data[i,j]) and (data[i,j] >= data[i-1,j-1]):
                new_image[i,j] -= 1
    data = new_image
    # NE-SW
    for i in range(1,nrow-1):
        for j in range(1,ncol-1):
            if (data[i+1,j-1] < data[i,j]) and (data[i,j] >= data[i-1,j+1]):
                new_image[i,j] -= 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(nrow-1):
        for j in range(ncol):
            if (data[i+1,j] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1,ncol):
            if (data[i,j-1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # NW-SE
    for i in range(nrow-1):
        for j in range(ncol-1):
            if (data[i+1,j+1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # NE-SW
    for i in range(nrow-1):
        for j in range(1,ncol):
            if (data[i+1,j-1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image

    return data

def unsharp(IMAGE):    
    image = Image.fromarray(IMAGE.astype('uint8'))
    new_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))

    return new_image
    
def raw(IMAGE):
    #IMAGE = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2RGB)
    return IMAGE

def low_pass(IMAGE):
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # convert to HSV
    image2 = cv2.cvtColor(IMAGE, cv2.COLOR_RGB2GRAY)
    
    dft = cv2.dft(np.float32(image2),flags = cv2.DFT_COMPLEX_OUTPUT)

    # shift the zero-frequncy component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)
    rows, cols = image2.shape
    crow,ccol = rows//2 , cols//2

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1

    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

    return img_back
    
def high_pass(IMAGE):
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # convert to HSV
    image2 = cv2.cvtColor(IMAGE, cv2.COLOR_RGB2GRAY)
    
    dft = cv2.dft(np.float32(image2),flags = cv2.DFT_COMPLEX_OUTPUT)

    # shift the zero-frequncy component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)
    rows, cols = image2.shape
    crow,ccol = rows//2 , cols//2

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1

    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

    return img_back

#for FILENAME in random.sample(FILESN, len(FILESN)):
def apply_filter(METHOD,PATO, SOURCE, FILESN, FILESP, IMGN, PROP, H_RESO, L_RESO, TYPEIMG):
    COUNT = 0
    RESO = H_RESO if TYPEIMG == 'h' else L_RESO
    for FILENAME in FILESN:
        if any(list(map(lambda x: x in FILENAME, RESO))):
            if not process_filters(SOURCE, FILENAME, METHOD):
                continue
            COUNT = COUNT + 1
            if COUNT == IMGN:
                break

    x = 0

    NNORM = COUNT
    IMGP = int(NNORM//PROP) if len(FILESP) > NNORM//PROP else len(FILESP)
    COUNT = 0
    for FILENAME in FILESP:
        if any(list(map(lambda x: x in FILENAME, RESO))):
            if not process_filters(SOURCE, FILENAME, METHOD, '/'+PATO):
                continue
            COUNT = COUNT + 1
            if COUNT == IMGP:
                break
        
    
"""if ("20sus" not in FILENAME and "021sus" not in FILENAME and "60sus" not in FILENAME and "70sus" not in FILENAME and "80sus" not in FILENAME):
    continue"""