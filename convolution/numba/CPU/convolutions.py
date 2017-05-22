import abc
import numpy as np
import numba
from numba import int64, float64

DATA_PATH = '../../data/'

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
        out_image = type(image)(image.width-2,image.height-2)
        for i in range(1,image.width-1):
            for j in range(1,image.height-1):
                # A vectoriser pour les eleves
                out_image.pixels[i-1,j-1] = 0.25*(image.pixels[i-1,j]+image.pixels[i+1,j]+image.pixels[i,j-1]+image.pixels[i,j+1])
        return out_image

@numba.njit(parallel=False)
def convolve_impl(width, height, image_p, out_p):
    for j in range(1, height-1):
        for i in range(1, width-1):
            out_p[j-1,i-1] = (4*image_p[j,i]-image_p[j-1,i]-image_p[j+1,i]
                                            -image_p[j,i-1]-image_p[j,i+1])

# @numba.guvectorize([(int64[:], int64[:], float64[:, :], float64[:, :])], '(),(),(nx,ny)->(nx,ny)', 
#                    nopython=True, target='cuda')
# def guconvolve_impl(width, height, image_p, out_p):
#     for i in range(1, width[0]-1):
#         for j in range(1, height[0]-1):
#             out_p[i-1,j-1] = (4*image_p[i,j]-image_p[i-1,j]-image_p[i+1,j]
#                                             -image_p[i,j-1]-image_p[i,j+1])

@numba.njit(parallel=True)
def convolve_impl_color(width, height, image_p, out_p):
    for j in range(1, height-1):
        for i in range(1, width-1):
            for k in range(3):
                out_p[j,i,k] = (4*image_p[j,i,k]-image_p[j-1,i,k]-image_p[j+1,i,k]
                                            -image_p[j,i-1,k]-image_p[j,i+1,k])

class laplacien(convolution):
    """
    Definie l'operateur laplacien comme convolution : permet de detecter les bords dans une image
    """
    def convolve( self, image ):
        #out_image = type(image)(image.width-2,image.height-2)
        #print(image.width,image.height,image.pixels.shape)
        out_image = type(image)(image.width,image.height)

        if out_image.pixels.ndim == 2:
            #print(out_image.pixels.shape, image.pixels.shape, image.pixels.dtype)
            convolve_impl(image.width, image.height, image.pixels, out_image.pixels)
        if out_image.pixels.ndim == 3:
            convolve_impl_color(image.width, image.height, image.pixels, out_image.pixels)
        # On renormalise l'image :
        valmax = np.max(out_image.pixels.flat)
        valmax = max(1.,valmax)+1.E-9
        out_image.pixels *= 1./valmax
        return out_image

class convolv_matrix(convolution):
    """
    Convolution generale avec une taille de stencil quelconque. Permet de definir tous les stencils que l'on souhaite !
    """
    def __init__(self, convolution_array ):
        self.__convolution_array__ = convolution_array

    def convolve( self, image ) :
        nx     = self.__convolution_array__.shape[0]
        ny     = self.__convolution_array__.shape[1]
        half_x = nx//2
        half_y = ny//2
        out_image = type(image)(image.width-nx+1,image.height-ny+1)
        for j in range(half_y, image.height - half_y):
            for i in range(half_x, image.width - half_x):
                out_image.pixels[j-half_y,i-half_x] = 0.
                for jw in range(-half_y,ny-half_y):
                    for iw in range(-half_x,nx-half_x):
                        out_image.pixels[j-half_y,i-half_x] += self.__convolution_array__[jw+half_y,iw+half_x]*image.pixels[j+jw,i+iw]
        # On renormalise l'image en ramenant les valeurs des couleurs entre 0 et 1
        out_image.pixels = np.abs(out_image.pixels)
        valmax = np.max(out_image.pixels.flat)
        valmax = max(1.,valmax)+1.E-9
        out_image.pixels *= 1./valmax
        return out_image

if __name__ == '__main__':
    import time
    from scipy import misc
    import gray_image as gimage
    import rgb_image as cimage
    # img = gimage.gray_image(fileName=DATA_PATH + 'lena.png')
    # img_skel = gimage.gray_image(fileName=DATA_PATH + 'squelette.png')
    # cimg = cimage.rgb_image(fileName=DATA_PATH + 'lena_rgb.png')
    lapl = laplacien()
    # output_img = lapl.convolve(img)
    # start = time.time()
    # output_img = lapl.convolve(img)
    # end   = time.time()
    # print("Temps mis pour calculer laplacien de lena en gris : {} secondes".format(end-start))
    # output_img.show()
    cimg = cimage.rgb_image(fileName=DATA_PATH + 'input_0.png')
    cimg = np.zeros((1000, 1000))
    cimg_out = np.zeros((1000, 1000))
    convolve_impl(1000, 1000, cimg, cimg_out)
    #output_img = lapl.convolve(cimg)
    start = time.time()
    for i in range(100):
        convolve_impl(1000, 1000, cimg, cimg_out)
        #output_img = lapl.convolve(cimg)
    end   = time.time()
    print("Temps mis pour calculer laplacien de lena en couleur : {} secondes".format(end-start))
    # output_img.show()
    # output_img.save(fileName='lena_contour_rgb.png')
    # m = mean()
    # start = time.time()
    # output_img = m.convolve(cimg)
    # end   = time.time()
    # print("Temps mis pour calculer moyenne de lena en couleur : {} secondes".format(end-start))
    # output_img.show()

    # # Matrice de convolution pour detection de bord amelioree :
    # convol = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],np.double)
    # f = convolv_matrix(convol)
    # start = time.time()
    # output_img = f.convolve(img)
    # end   = time.time()
    # print("Temps mis pour calculer convolution contour ameliore de lena en gris : {} secondes".format(end-start))
    # output_img.show()

    # # Matrice de convolution pour preciser les contours d'une image ( sharpen )
    # convol = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.double)
    # f = convolv_matrix(convol)
    # start = time.time()
    # output_img = f.convolve(img_skel)
    # end   = time.time()
    # print("Temps mis pour calculer convolution sharpen de squelette en gris : {} secondes".format(end-start))
    # output_img.show()

    # # Matrice de convolution pour faire du Gaussian blur avec un stencil de 5x5
    # convol = (1./256.)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]],np.double)
    # f = convolv_matrix(convol)
    # start = time.time()
    # output_img = f.convolve(cimg)
    # end   = time.time()
    # print("Temps mis pour calculer convolution gaussian blur 5x5 de lena en couleur : {} secondes".format(end-start))
    # output_img.show()
