import numpy as np
"""
Definit une convolution faisant une moyenne des voisins d'un pixel donne
( stencil de 3x3 )
"""
def convolve_mean2(image):
    height, width = image.shape
    out_image = np.empty((height-2,width-2))
    out_image[:, :] = 0.25*(image[:-2,1:-1]+image[2:,1:-1]+image[1:-1,:-2]+image[1:-1,2:])
    return out_image

def convolve_mean3(image):
    height, width, dim = image.shape
    out_image = np.empty((height-2,width-2, dim))
    out_image[:, :, :] = 0.25*(image[:-2,1:-1,:]+image[2:,1:-1,:]+image[1:-1,:-2,:]+image[1:-1,2:,:])
    return out_image

"""
Definie l'operateur laplacien comme convolution : permet de detecter les bords dans une image
"""
def convolve_laplacien2(image):
    height, width = image.shape
    out_image = np.empty((height-2,width-2))
    out_image[:, :] = np.abs(4*image[1:-1,1:-1]-image[:-2,1:-1]-image[2:,1:-1]
                                               -image[1:-1,:-2]-image[1:-1,2:])
    # On renormalise l'image :
    valmax = np.max(out_image)
    valmax = max(1.,valmax)+1.E-9
    out_image *= 1./valmax
    return out_image

def convolve_laplacien3(image):
    height, width, dim = image.shape
    out_image = np.empty((height-2,width-2, dim))
    out_image[:, :, :] = np.abs(4*image[1:-1,1:-1,:]-image[:-2,1:-1,:]-image[2:,1:-1,:]
                                                    -image[1:-1,:-2,:]-image[1:-1,2:,:])
    # On renormalise l'image :
    valmax = np.max(out_image)
    valmax = max(1.,valmax)+1.E-9
    out_image *= 1./valmax
    return out_image

"""
Convolution generale avec une taille de stencil quelconque. Permet de definir tous les stencils que l'on souhaite !
"""
def convolve_matrix2(image, convolution_array) :
    height, width= image.shape
    nx     = convolution_array.shape[0]
    ny     = convolution_array.shape[1]
    half_x = nx//2
    half_y = ny//2
    out_image = np.zeros((height-nx+1,width-ny+1))
    h, w = out_image.shape

    for jw in range(0,ny):
        for iw in range(0,nx):
            out_image[:,:] += convolution_array[jw,iw]*image[jw:jw+h,iw:iw+w]

    # On renormalise l'image en ramenant les valeurs des couleurs entre 0 et 1
    out_image = np.abs(out_image)
    valmax = np.max(out_image)
    valmax = max(1.,valmax)+1.E-9
    out_image *= 1./valmax
    return out_image

def convolve_matrix3(image, convolution_array) :
    height, width, dim = image.shape
    nx     = convolution_array.shape[0]
    ny     = convolution_array.shape[1]
    half_x = nx//2
    half_y = ny//2
    out_image = np.zeros((height-nx+1,width-ny+1, dim))
    h, w, d = out_image.shape

    for jw in range(0,ny):
        for iw in range(0,nx):
            out_image[:,:,:] += convolution_array[jw,iw]*image[jw:jw+h,iw:iw+w,:]

    # On renormalise l'image en ramenant les valeurs des couleurs entre 0 et 1
    out_image = np.abs(out_image)
    valmax = np.max(out_image)
    valmax = max(1.,valmax)+1.E-9
    out_image *= 1./valmax
    return out_image
