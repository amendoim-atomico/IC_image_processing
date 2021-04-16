import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.gridspec import GridSpec

#load image, must be GRAY_SCALE

img = cv2.imread("brain.jpg",0)

height,width = img.shape

#variables:

#rect region to be truncated

#rectangle coordinates, top left (a1,a2) -> (b1,b2) bottom right
#coord = (a1,a2,b1,b2)
coord = (200,250,300,400)

#width of the rect
width_rect = 40

#end variables

x1=coord[0]
y1=coord[1]

x2=coord[2]
y2=coord[3]


fourier = np.fft.fft2(img)

#fourier needs a shift to move the low and high frequencies of the k space
fourier_shift = np.fft.fftshift(fourier)

#log is only for display result
spectrum = np.log(abs(fourier_shift))

#go all over the choosen area
for i in range(x1,x2):
    for j in range(y1,y2):
        #pick area
        if not (i>=x1+width_rect and i<=x2-width_rect and j>=y1+width_rect and j<=y2-width_rect):
            fourier_shift[j][i] = np.complex(0,0)


spectrum2 = np.log(abs(fourier_shift))

img2 = abs(np.fft.ifft2(fourier_shift))


#Create figure and GRID

fig = plt.figure(figsize=(10,40))
gs = GridSpec(1,4)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[0,3])


#original image
ax1.imshow(img,cmap='gray')
ax1.set_title("original image", fontsize=10)

#k-space from original image
ax2.imshow(spectrum,cmap='gray')
ax2.set_title("k-space original", fontsize=10)

#truncated k-space
ax3.imshow(spectrum2,cmap='gray')
ax3.set_title("k-space truncated",fontsize=10)

#image from truncated k-space
ax4.imshow(img2,cmap='gray')
ax4.set_title("image from truncated k-space", fontsize=10)


plt.show()

