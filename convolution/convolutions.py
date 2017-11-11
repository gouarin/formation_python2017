import abc
import numpy as np

class convolution:
    __metaclass__ = abc.ABCMeta

    def __init__(self, vectorize):
        if vectorize:
            self.fun = lambda _: 'convolve_' + self.__class__.__name__
        else:
            self.fun = lambda im: 'convolve_{}{}'.format(self.__class__.__name__, len(im.pixels.shape))

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
        fun = getattr(kernels, self.fun(image))
        return type(image)(from_array=fun(image.pixels))

class laplacien(convolution):
    """
    Definie l'operateur laplacien comme convolution : permet de detecter les bords dans une image
    """
    def convolve( self, image ):
        fun = getattr(kernels, self.fun(image))
        return type(image)(from_array=fun(image.pixels))

class matrix(convolution):
    """
    Convolution generale avec une taille de stencil quelconque. Permet de definir tous les stencils que l'on souhaite !
    """
    def __init__(self, convolution_array, vectorize):
        super(matrix, self).__init__(vectorize)
        self.__convolution_array__ = convolution_array

    def convolve( self, image ) :
        fun = getattr(kernels, self.fun(image))
        return type(image)(from_array=fun(image.pixels, self.__convolution_array__))

def simple_bench(title, function, n):
    from time import time
    function() # warmup
    timings = []
    for i in range(n):
        start = time()
        function()
        end = time()
        timings.append(end -start)
    #print('[{}]: min: {}s, max: {}s, median: {}s, average: {}s'.format(title, min(timings), max(timings), sorted(timings)[n//2], sum(timings)/n))
    print('[{}]: {}s'.format(title, sorted(timings)[n//2]))

if __name__ == '__main__':
    import sys
    from scipy.misc.pilutil import imread
    import gray_image as gimage
    import rgb_image as cimage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('engine', help='target engine')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--vectorize', action='store_true')
    parser.add_argument('--count', default=10, type=int)
    args = parser.parse_args()

    sys.path.append(args.engine)
    import kernels

    img = gimage.gray_image(fileName='data/lena.png')
    img_skel = gimage.gray_image(fileName='data/squelette.png')
    cimg = cimage.rgb_image(fileName='data/lena_rgb.png')

    show = args.show
    lapl = laplacien(args.vectorize)
    if show:
        output_img = lapl.convolve(img)
        output_img.show()
    else:
        simple_bench('laplacien/grayscale', lambda: lapl.convolve(img), args.count)

    if show:
        output_img = lapl.convolve(cimg)
        output_img.show()
    else:
        simple_bench('laplacien/color', lambda: lapl.convolve(cimg), args.count)

    m = mean(args.vectorize)
    if show:
        output_img = m.convolve(img)
        output_img.show()
    else:
        simple_bench('mean/grayscale', lambda: m.convolve(img), args.count)

    if show:
        output_img = m.convolve(cimg)
        output_img.show()
    else:
        simple_bench('mean/color', lambda: m.convolve(cimg), args.count)

    # Matrice de convolution pour detection de bord amelioree :
    convol = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],np.double)
    f = matrix(convol, args.vectorize)
    if show:
        output_img = f.convolve(img)
        output_img.show()
    else:
        simple_bench('border/grayscale', lambda: f.convolve(img), args.count)

    # Matrice de convolution pour preciser les contours d'une image ( sharpen )
    convol = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.double)
    f = matrix(convol, args.vectorize)
    if show:
        output_img = f.convolve(img_skel)
        output_img.show()
    else:
        simple_bench('sharpen/grayscale', lambda: f.convolve(img_skel), args.count)
    # Matrice de convolution pour faire du Gaussian blur avec un stencil de 5x5
    convol = (1./256.)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]],np.double)
    f = matrix(convol, args.vectorize)
    if show:
        output_img = f.convolve(cimg)
        output_img.show()
    else:
        simple_bench('blur/grayscale', lambda: f.convolve(img_skel), args.count)
