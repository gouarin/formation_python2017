#
"""
RGB color image
"""
from scipy import misc
import numpy as np
import image as img

class rgb_image(img.image):
    """
    """
    def __init__(self, w = 0, h = 0, fileName = None, from_array = None):
        """
        Construction d'une image de width x height pixels contenant trois channels R,G et B
        """
        if None == fileName :
            img.image.__init__(self,w,h,3, from_array)
        else:
            data = misc.imread(fileName)
            img.image.__init__(self,data.shape[0],data.shape[1],3)
            self.__pixels__ = data[:,:,0:3]/255.


if __name__ == '__main__':
    img = rgb_image(fileName = 'lena_rgb.png')
    img.pixels = np.ones((img.height,img.width,3),np.double) - img.pixels
    img.show()
    img.save("inverse_rgb_lena.png")
    
