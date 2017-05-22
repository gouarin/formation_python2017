#
"""
Gray level image
"""
from scipy import misc
import numpy as np
import image as img

class gray_image(img.image):
    """
    """
    def __init__(self, w = 0, h = 0, fileName = None):
        """
        Construction d'une image de width x height pixels
        """
        if None == fileName :
            img.image.__init__(self,w,h)
        else:
            data = misc.imread(fileName)
            img.image.__init__(self,data.shape[0],data.shape[1])
            self.__pixels__ = data/255.
            
        
if __name__ == '__main__':
    img = gray_image(fileName='lena.png')
    img.pixels = np.ones((img.height,img.width),np.double) - img.pixels
    img.show()
    
