import cv2  # openCV
import numpy as np


def imgREAD(imgName, param):
    # imgName = fileName - define in config.py
    # param: 0: nothing, 1 print matrix, 2 show image
    # img = cv2.imread(config.imageToRead)
    img = cv2.imread(imgName)

    print("READ File: " + str(imgName))

    print(type(img))
    # <class 'numpy.ndarray'>

    print(img.shape)
    # (B, G, R)

    # print(img.dtype)
    # uint8	print (img)

    if param == 1:
        print(img)
    if param == 2:
        imgSHOW(imgName)

    return img


def imgC2G(imgName, grayName, param):
    # imgName = fileName - define in config.py
    # param: 0: nothing, 1 print matrix, 2 show image
    # img = cv2.imread(config.imageToRead)
    img = cv2.imread(imgName)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("GRAY File: " + str(grayName))

    print(type(gray))
    # <class 'numpy.ndarray'>

    print(gray.shape)
    # (B, G, R)

    # print(img.dtype)
    # uint8	print (img)

    cv2.imwrite(grayName, gray)

    if param == 1:
        print(gray)
    if param == 2:
        imgSHOW(gray)

    return gray


def imgWRITE(imgName, img, param):
    # imgName = fileName - define in config.py
    # param: 0: nothing, 1 print matrix, 2 show image
    # img = cv2.imwrite(config.imageSpec)

    cv2.imwrite(imgName, img)

    print("WRITE File: " + str(imgName))

    print(type(img))
    # <class 'numpy.ndarray'>

    print(img.shape)
    # (225, 400, 3)

    # print(img.dtype)
    # uint8	print (img)
    if param == 1:
        print(img)
    if param == 2:
        imgSHOW(imgName)
    return img


def imgSHOW(imgName):
    # importing Image class from PIL package
    from PIL import Image
    # creating a object
    im = Image.open(imgName)
    im.show()
    return (1)


def imgSORT(img, numberBlock=0, param=0):
    #	print(img1.shape)

    nb = numberBlock
    para = param
    height = img.shape[0]
    width = img.shape[1]
    nzahl = int(height / nb)
    #	print("pic height:" + str(height))
    #	print("pic width:" + str(width))
    #	print("block:" + str(nb))
    #	print("Param:" + str(para))
    #   ---------------------------------
    #   preset ordner
    #	os.mkdir("./blocks/")
    imageBlock = np.zeros_like(img).astype(int)
    nf = 0
    nbK = 0
    nbK1 = nb
    nbI = 0
    nbI1 = nb

    for ri in range(nzahl):
        for r in range(nzahl):
            for i in range(nbI, nbI1):
                for k in range(nbK, nbK1):
                    imageBlock[i][k] = img[i][k]
            cv2.imwrite('./blocks/block' + str(nf) + '.png', imageBlock)
            nf = nf + 1
            imageBlock = np.zeros_like(img).astype(int)
            nbK = nbK + nb
            nbK1 = nbK1 + nb
        nbI = nbI + nb
        nbI1 = nbI1 + nb
        nbK = 0
        nbK1 = nb
    return imageBlock


def animator(png_dir="", modus=""):
    # GIF animation generator
    import os
    import imageio
    #	png_dir = config.dirBlocks
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave('./img_out/movie' + modus + '.gif', images)
    return True
