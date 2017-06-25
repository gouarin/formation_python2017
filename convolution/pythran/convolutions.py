import abc
import numpy as np
from kernels import *

class convolution:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def convolve(self, image):
        """
        Applique la convolution sur une image ( grise ou en couleur )
        """
        return


class mean(convolution):
    """
    Definie une convolution faisant une moyenne des pixels voisins d'un pixel donne
    ( stencil de 3x3 )
    """
    def convolve(self, image):
        fun = globals()['convolve_mean{}'.format(len(image.pixels.shape))]
        return type(image)(from_array=fun(image.pixels))

class laplacien(convolution):
    """
    Definie l'operateur laplacien comme convolution : permet de detecter les bords dans une image
    """
    def convolve( self, image ):
        fun = globals()['convolve_laplacien{}'.format(len(image.pixels.shape))]
        return type(image)(from_array=fun(image.pixels))

class convolv_matrix(convolution):
    """
    Convolution generale avec une taille de stencil quelconque. Permet de definir tous les stencils que l'on souhaite !
    """
    def __init__(self, convolution_array ):
        self.__convolution_array__ = convolution_array

    def convolve( self, image ) :
        fun = globals()['convolve_matrix{}'.format(len(image.pixels.shape))]
        return type(image)(from_array=fun(image.pixels, self.__convolution_array__))

if __name__ == '__main__':
    import perf
    import sys
    from scipy.misc.pilutil import imread
    import gray_image as gimage
    import rgb_image as cimage
    img = gimage.gray_image(fileName='../data/lena.png')
    img_skel = gimage.gray_image(fileName='../data/squelette.png')
    cimg = cimage.rgb_image(fileName='../data/lena_rgb.png')

    show = len(sys.argv) > 1 and sys.argv[1] == "show"
    if not show:
        runner = perf.Runner()

    lapl = laplacien()
    if show:
        output_img = lapl.convolve(img)
        output_img.show()
    else:
        runner.bench_func('laplacien/grayscale', lambda: lapl.convolve(img))

    if show:
        output_img = lapl.convolve(cimg)
        output_img.show()
    else:
        runner.bench_func('laplacien/color', lambda: lapl.convolve(cimg))

    m = mean()
    if show:
        output_img = m.convolve(cimg)
        output_img.show()
    else:
        runner.bench_func('mean/color', lambda: m.convolve(cimg))

    # Matrice de convolution pour detection de bord amelioree :
    convol = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],np.double)
    f = convolv_matrix(convol)
    if show:
        output_img = f.convolve(img)
        output_img.show()
    else:
        runner.bench_func('border/grayscale', lambda: f.convolve(img))

    # Matrice de convolution pour preciser les contours d'une image ( sharpen )
    convol = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.double)
    f = convolv_matrix(convol)
    if show:
        output_img = f.convolve(img_skel)
        output_img.show()
    else:
        runner.bench_func('sharpen/grayscale', lambda: f.convolve(img_skel))
    # Matrice de convolution pour faire du Gaussian blur avec un stencil de 5x5
    convol = (1./256.)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]],np.double)
    f = convolv_matrix(convol)
    if show:
        output_img = f.convolve(cimg)
        output_img.show()
    else:
        runner.bench_func('blur/grayscale', lambda: f.convolve(img_skel))
