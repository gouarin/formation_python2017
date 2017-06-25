#
"""
"""
from scipy import misc
import numpy as np

class image(object):
    """
    """
    def __init__(self,w = 0, h = 0, depth = 1, from_array=None):
        """
        Cree une image de w x h pixels et de profondeur depth
        """
        if from_array is None:
            if depth > 1 :
                self.__pixels__ = np.zeros((h,w,depth),np.double)
            else:
                self.__pixels__ = np.zeros((h,w),np.double)
        else:
            self.__pixels__ = from_array

    def show(self) :
        """Affiche l'image a l'ecran"""
        misc.imshow(255*self.__pixels__)


    def save(self, fileName):
        """
        Sauvegarde l'image en format png
        """
        misc.imsave(fileName, 255*self.__pixels__)

    @property
    def pixels(self):
        """Retourne le tableau des pixels"""
        return self.__pixels__
        
    @pixels.setter
    def pixels(self, value):
        """
        Change le tableau des pixels
        """
        self.__pixels__ = value
    
    @property
    def  width(self):
        """Retourne la largeur de l'image"""
        return self.__pixels__.shape[1]

    @property
    def height(self):
        """Retourne la hauteur de l'image"""
        return self.__pixels__.shape[0]
