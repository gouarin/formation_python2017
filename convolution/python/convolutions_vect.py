import abc
import numpy as np

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
        out_image.pixels[:,:] = 0.25*(image.pixels[0:-2,0:-2]+image.pixels[2:,0:-2]
                                      +image.pixels[0:-2,2:]+image.pixels[2:,2:])
        return out_image

class laplacien(convolution):
    """
    Definie l'operateur laplacien comme convolution : permet de detecter les bords dans une image
    """
    def convolve( self, image ):
        out_image = type(image)(image.width-2,image.height-2)
        out_image.pixels[:,:] = np.abs(4*image.pixels[1:-1,1:-1] -
                                       image.pixels[0:-2,1:-1] - image.pixels[2:,1:-1] -
                                       image.pixels[1:-1,0:-2] - image.pixels[1:-1,2:])
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
        nx     = self.__convolution_array__.shape[1]
        ny     = self.__convolution_array__.shape[0]
        half_x = nx//2
        half_y = ny//2
        out_image = type(image)(image.width-nx+1,image.height-ny+1)
        # A la creation, les pixels sont tous noirs, donc pas besoin de remettre a zero
        # les valeurs des pixels

        h = out_image.height
        w = out_image.width
        for jw in range(0,ny):
            for iw in range(0,nx):
                out_image.pixels[:,:] += self.__convolution_array__[jw,iw]*image.pixels[jw:jw+h,iw:iw+w]
        
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
    img = gimage.gray_image(fileName='../data/lena.png')
    img_skel = gimage.gray_image(fileName='../data/squelette.png')
    cimg = cimage.rgb_image(fileName='../data/lena_rgb.png')
    lapl = laplacien()
    start = time.time()
    for i in range(10000):
        output_img = lapl.convolve(img)
    end   = time.time()
    print("Temps mis pour calculer laplacien de lena en gris : {} secondes".format(end-start))
    output_img.show()
    output_img.save(fileName='lena_contour.png')
    start = time.time()
    output_img = lapl.convolve(cimg)
    end   = time.time()
    print("Temps mis pour calculer laplacien de lena en couleur : {} secondes".format(end-start))
    output_img.show()
    m = mean()
    start = time.time()
    output_img = m.convolve(cimg)
    end   = time.time()
    print("Temps mis pour calculer moyenne de lena en couleur : {} secondes".format(end-start))
    output_img.show()

    # Matrice de convolution pour detection de bord amelioree :
    convol = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],np.double)
    f = convolv_matrix(convol)
    start = time.time()
    output_img = f.convolve(img)
    end   = time.time()
    print("Temps mis pour calculer convolution contour ameliore de lena en gris : {} secondes".format(end-start))
    output_img.show()
    output_img.save(fileName='lena_contour_upgraded.png')

    # Matrice de convolution pour preciser les contours d'une image ( sharpen )
    convol = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.double)
    f = convolv_matrix(convol)
    start = time.time()
    output_img = f.convolve(img_skel)
    end   = time.time()
    print("Temps mis pour calculer convolution sharpen de squelette en gris : {} secondes".format(end-start))
    output_img.show()

    # Matrice de convolution pour preciser les contours ameliores d'une image ( sharpen )
    convol = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]],np.double)
    f = convolv_matrix(convol)
    start = time.time()
    output_img = f.convolve(img_skel)
    end   = time.time()
    print("Temps mis pour calculer convolution sharpen de squelette en gris : {} secondes".format(end-start))
    output_img.show()

    # Matrice de convolution pour faire du Gaussian blur avec un stencil de 5x5
    convol = (1./256.)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]],np.double)
    f = convolv_matrix(convol)
    start = time.time()
    output_img = f.convolve(cimg)
    end   = time.time()
    print("Temps mis pour calculer convolution gaussian blur 5x5 de lena en couleur : {} secondes".format(end-start))
    output_img.show()

