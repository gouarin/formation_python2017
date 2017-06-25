import numpy as np
import numba
import math
"""
Definit une convolution faisant une moyenne des voisins d'un pixel donne
( stencil de 3x3 )
"""
@numba.njit
def convolve_mean2(image):
    height, width = image.shape
    out_image = np.empty((height-2,width-2))
    for i in range(1,width-1):
        for j in range(1,height-1):
            out_image[i-1,j-1] = 0.25*(image[i-1,j]+image[i+1,j]+image[i,j-1]+image[i,j+1])
    return out_image

@numba.njit
def convolve_mean3(image):
    height, width, d = image.shape
    out_image = np.empty((height-2,width-2, d))
    for i in range(1,width-1):
        for j in range(1,height-1):
           for k in range(d):
            out_image[i-1,j-1,k] = 0.25*(image[i-1,j,k]+image[i+1,j,k]+image[i,j-1,k]+image[i,j+1,k])
    return out_image

"""
Definie l'operateur laplacien comme convolution : permet de detecter les bords dans une image
"""
@numba.jit
def convolve_laplacien2(image):
    valmax = 0.
    height, width = image.shape
    out_image = np.empty((height-2,width-2))
    for i in range(1,width-1):
        for j in range(1,height-1):
            # A vectoriser pour les eleves
            out_image[i-1,j-1] = np.abs(4*image[i,j]-image[i-1,j]-image[i+1,j]
                                                      -image[i,j-1]-image[i,j+1])
    # On renormalise l'image :
    valmax = np.max(out_image)
    valmax = max(1.,valmax)+1.E-9
    out_image *= 1./valmax
    return out_image

@numba.jit
def convolve_laplacien3(image):
    height, width, d = image.shape
    out_image = np.empty((height-2,width-2, d))
    for i in range(1,width-1):
        for j in range(1,height-1):
            for k in range(3):
                out_image[i-1, j-1, k] = np.abs(4*image[i,j, k]-image[i-1, j, k]-image[i+1, j, k]
                                                               -image[i, j-1, k]-image[i, j+1, k])
    # On renormalise l'image :
    valmax = np.max(out_image)
    valmax = max(1.,valmax)+1.E-9
    out_image *= 1./valmax
    return out_image

"""
Convolution generale avec une taille de stencil quelconque. Permet de definir tous les stencils que l'on souhaite !
"""
@numba.jit
def convolve_matrix2(image, convolution_array) :
    height, width= image.shape
    nx     = convolution_array.shape[0]
    ny     = convolution_array.shape[1]
    half_x = nx//2
    half_y = ny//2
    out_image = np.empty((height-nx+1,width-ny+1))
    for j in range(half_y, height - half_y):
        for i in range(half_x, width - half_x):
            tmp = 0.
            for jw in range(-half_y,ny-half_y):
                for iw in range(-half_x,nx-half_x):
                    tmp += convolution_array[jw+half_y,iw+half_x]*image[j+jw,i+iw]
            out_image[j-half_y,i-half_x] = abs(tmp)
    # On renormalise l'image en ramenant les valeurs des couleurs entre 0 et 1
    valmax = np.max(out_image)
    valmax = max(1.,valmax)+1.E-9
    out_image *= 1./valmax
    return out_image

@numba.jit
def convolve_matrix3(image, convolution_array) :
    height, width, d = image.shape
    nx     = convolution_array.shape[0]
    ny     = convolution_array.shape[1]
    half_x = nx//2
    half_y = ny//2
    out_image = np.empty((height-nx+1,width-ny+1, d))
    for j in range(half_y, height - half_y):
        for i in range(half_x, width - half_x):
            for k in range(3):
                tmp = 0.
                for jw in range(-half_y,ny-half_y):
                    for iw in range(-half_x,nx-half_x):
                        tmp += convolution_array[jw+half_y,iw+half_x]*image[j+jw, i+iw, k]
                out_image[j-half_y,i-half_x, k] = abs(tmp)
    # On renormalise l'image en ramenant les valeurs des couleurs entre 0 et 1
    valmax = np.max(out_image)
    valmax = max(1.,valmax)+1.E-9
    out_image *= 1./valmax
    return out_image

