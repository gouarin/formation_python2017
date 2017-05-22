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
        for i in range(1,image.width-1):
            for j in range(1,image.height-1):
                # A vectoriser pour les eleves
                out_image.pixels[i-1,j-1] = 0.25*(image.pixels[i-1,j]+image.pixels[i+1,j]+image.pixels[i,j-1]+image.pixels[i,j+1])
        return out_image

class laplacien(convolution):
    """
    Definie l'operateur laplacien comme convolution : permet de detecter les bords dans une image
    """
    def convolve( self, image ):
        out_image = type(image)(image.width-2,image.height-2)
        for i in range(1,image.width-1):
            for j in range(1,image.height-1):
                # A vectoriser pour les eleves
                out_image.pixels[i-1,j-1] = np.abs(4*image.pixels[i,j]-image.pixels[i-1,j]-image.pixels[i+1,j]
                                                   -image.pixels[i,j-1]-image.pixels[i,j+1])
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
    img = gimage.gray_image(fileName='../data/lena.png')
    img_skel = gimage.gray_image(fileName='../data/squelette.png')
    cimg = cimage.rgb_image(fileName='../data/lena_rgb.png')
    lapl = laplacien()
    start = time.time()
    output_img = lapl.convolve(img)
    end   = time.time()
    print("Temps mis pour calculer laplacien de lena en gris : {} secondes".format(end-start))
    output_img.show()
    start = time.time()
    output_img = lapl.convolve(cimg)
    end   = time.time()
    print("Temps mis pour calculer laplacien de lena en couleur : {} secondes".format(end-start))
    output_img.show()
    output_img.save(fileName='lena_contour_rgb.png')
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

    # Matrice de convolution pour preciser les contours d'une image ( sharpen )
    convol = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.double)
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
