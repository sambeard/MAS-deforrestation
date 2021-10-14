import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt


#------R G B------
#old 51 
#young 53
#wastleand 36
#farmland 67
#river blue

patches = np.array([[85, 89, 66, 51], #old 0
           [34, 177, 76, 53], #young 1
           [0, 162, 232, 96], #river 2
           [239, 228, 176, 36], #wasteland 3
           [127, 127, 127, 5], #city 4
           [255, 242, 0, 67]], np.float32) #farmland 5

patch_name = ["old forest", 
            "young forest",
            "river",
            "wasteland",
            "city",
            "farmland"]

def calculate_min_distance(color):
    min_l2 = 100000
    for i in range(0, 6):
        # print(patches[i][0])
        l2_r = np.square(color[2] - patches[i][0])
        l2_g = np.square(color[1] - patches[i][1])
        l2_b = np.square(color[0] - patches[i][2])

        l2 = np.sqrt(sum((l2_r, l2_b, l2_g)))
        if l2 <= min_l2:
            min_l2  = l2
            patch_type = i
    return patch_type


im = cv.resize(cv.imread("world_bmp.bmp"),(200,200))
w, h ,_= im.shape
print("width: ", w, "height: ", h )

Z = im.reshape((-1,3))
Z = np.float32(Z)

K = 10
criteria =  (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center  = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((im.shape))

print(Z.shape)
print(label.shape)
print(res.shape)
print(res2.shape)

cv.imshow('res2', res2)
cv.waitKey(0)
cv.destroyAllWindows()

original_stdout = sys.stdout # Save a reference to the original standard output

#"PATCHES"
#"pxcor","pycor","pcolor","plabel","plabel-color","ptype","maturity","altitude"
#"0","200","51","""""","9.9","""old forest""","0","280"



cp_label = np.array(label)
cp_label = cp_label.reshape(-1)
patch_names = []

for i in range (0, 10):
    x = np.array(np.where(cp_label == i), np.int16)
    x = x.reshape(-1)
    color = res[x[0]]
    patch_type = calculate_min_distance(np.float32(color))
    cp_label = np.where(cp_label == i, patch_type, cp_label)

with open('patches.csv', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print("\"PATCHES\"\n\"pxcor\",\"pycor\",\"pcolor\",\"plabel\",\"plabel-color\",\"ptype\",\"maturity\",\"altitude\"")
    for pycor in range (0, h):
        for pxcor in range(0, w):
            pcolor = patches[cp_label[pxcor]][3]
            plabel = patch_name[cp_label[pxcor]]
            print("\"%d\",\"%d\",\"%d\",\"\"\"\"\"\",\"9.9\",\"\"\"%s\"\"\",\"0\""% (pxcor, pycor, pcolor, plabel))
    sys.stdout = original_stdout # Reset the standard output to its original value







