import numpy as np
import numba
import cmath
"""
Definit une convolution faisant une moyenne des voisins d'un pixel donne
( stencil de 3x3 )
"""
#@numba.guvectorize(['(float64[:,:], int64[:], int64[:], float64[:])'], '(nx, ny),(), ()->()', target='parallel', nopython=True)
#def convolve_mean2_gu(image, i, j, out_image):
#    nx, ny = image.shape
#    out_image[0] = 0.25*(image[i[0]-1,j[0]]+image[i[0]+1,j[0]]+image[i[0],j[0]-1]+image[i[0],j[0]+1])

@numba.guvectorize(['(float64[:,:], int64[:], float64[:])'], '(nx, ny),()->(ny)', target='parallel', nopython=True)
def convolve_mean2_gu(image, i, out_image):
    nx, ny = image.shape
    for j in range(1, ny-1):
        out_image[j-1] = 0.25*(image[i[0]-1,j]+image[i[0]+1,j]+image[i[0],j-1]+image[i[0],j+1])

@numba.guvectorize(['(float64[:, :, :], int64[:], float64[:, :])'], '(nx, ny, nk),()->(ny, nk)', target='parallel', nopython=True)
def convolve_mean3_gu(image, index, out_image):
    nx, ny, nk = image.shape
    for j in range(1,ny-1):
        for k in range(3):
            out_image[j-1, k] = 0.25*(image[index[0]-1,j,k]+image[index[0]+1,j,k]+image[index[0],j-1,k]+image[index[0],j+1,k])

#@numba.jit
#def convolve_mean2(image):
#    height, width = image.shape
#    out_image = np.empty((height-2, width-2))
#    i = np.arange(1, height-1)
#    j = np.arange(1, width-1)
#    convolve_mean2_gu(image, i[:, np.newaxis], j[np.newaxis, :], out_image)
#    return out_image

@numba.jit(nogil=True)
def convolve_mean2(image):
    height, width = image.shape
    out_image = np.empty((height-2, width))
    i = np.arange(1, height-1)
    convolve_mean2_gu(image, i, out_image)
    return out_image[:, :-2]

@numba.jit(nogil=True)
def convolve_mean3(image):
    height, width, d = image.shape
    out_image = np.empty((height-2,width, d))
    index = np.arange(1, height-1)
    convolve_mean3_gu(image, index, out_image)
    return out_image[:, :-2, :]

"""
Definie l'operateur laplacien comme convolution : permet de detecter les bords dans une image
"""
@numba.guvectorize(['(float64[:,:], int64[:], float64[:])'], '(nx, ny),()->(ny)', target='parallel', nopython=True)
def convolve_laplacien2_gu(image, index, out_image):
    nx, ny = image.shape
    for j in range(1,ny-1):
        out_image[j-1] = np.abs(4*image[index[0],j]-image[index[0]-1,j]-image[index[0]+1,j]
                                                   -image[index[0],j-1]-image[index[0],j+1])

#@numba.guvectorize(['(float64[:,:], int64[:], int64[:], float64[:])'], '(nx, ny),(),()->()', target='parallel', nopython=True)
#def convolve_laplacien2_gu(image, i, j, out_image):
#    nx, ny = image.shape
#    out_image[0] = np.abs(4*image[i[0],j[0]]-image[i[0]-1,j[0]]-image[i[0]+1,j[0]]
#                                            -image[i[0],j[0]-1]-image[i[0],j[0]+1])

#@numba.jit
#def convolve_laplacien2(image):
#    height, width = image.shape
#    out_image = np.empty((height-2,width-2))
#    i = np.arange(1, height-1)[:, np.newaxis]
#    j = np.arange(1, width-1)[np.newaxis, :]
#    convolve_laplacien2_gu(image, i, j, out_image)
#    # On renormalise l'image :
#    valmax = np.max(out_image)
#    valmax = max(1.,valmax)+1.E-9
#    out_image *= 1./valmax
#    return out_image

@numba.jit(nogil=True)
def convolve_laplacien2(image):
    height, width = image.shape
    out_image = np.empty((height-2,width))
    index = np.arange(1, height-1)
    convolve_laplacien2_gu(image, index, out_image)
    # On renormalise l'image :
    valmax = np.max(out_image[:, :-2])
    valmax = max(1.,valmax)+1.E-9
    out_image *= 1./valmax
    return out_image[:, :-2]

@numba.guvectorize(['(float64[:,:,:], int64[:], float64[:,:])'], '(nx, ny, nk),()->(ny, nk)', target='parallel', nopython=True)
def convolve_laplacien3_gu(image, index, out_image):
    nx, ny, nk = image.shape
    for j in range(1,ny-1):
        for k in range(nk):
            out_image[j-1,k] = np.abs(4*image[index[0],j,k]-image[index[0]-1,j,k]-image[index[0]+1,j,k]
                                                           -image[index[0],j-1,k]-image[index[0],j+1,k])

@numba.jit(nogil=True)
def convolve_laplacien3(image):
    height, width, d = image.shape
    out_image = np.empty((height-2,width, d))
    index = np.arange(1, height-1)
    convolve_laplacien3_gu(image, index, out_image)
    # On renormalise l'image :
    valmax = np.max(out_image[:, :-2,:])
    valmax = max(1.,valmax)+1.E-9
    out_image *= 1./valmax
    return out_image[:, :-2,:]

"""
Convolution generale avec une taille de stencil quelconque. Permet de definir tous les stencils que l'on souhaite !
"""
@numba.guvectorize(['(float64[:,:], float64[:,:], int64[:], float64[:])'], '(nx, ny),(ni, nj), ()->(ny)', target='parallel', nopython=True)
def convolve_matrix2_gu(image, convolution_array, index, out_image) :
    height, width = image.shape
    nx     = convolution_array.shape[1]
    ny     = convolution_array.shape[0]
    half_x = nx//2
    half_y = ny//2
    for i in range(half_x, width - half_x):
        out_image[i-half_x] = 0.
        for jw in range(-half_y,ny-half_y):
            for iw in range(-half_x,nx-half_x):
                out_image[i-half_x] += convolution_array[jw+half_y,iw+half_x]*image[index[0]+jw,i+iw]
        out_image[i-half_x] = np.abs(out_image[i-half_x])

@numba.jit(nogil=True)
def convolve_matrix2(image, convolution_array) :
    height, width= image.shape
    nx     = convolution_array.shape[1]
    ny     = convolution_array.shape[0]
    half_x = nx//2
    half_y = ny//2
    out_image = np.empty((height-ny+1,width))
    index = np.arange(half_y, height - half_y)
    convolve_matrix2_gu(image, convolution_array, index, out_image)
    # On renormalise l'image en ramenant les valeurs des couleurs entre 0 et 1
    #out_image = np.abs(out_image[:,::-nx+1])
    valmax = np.max(out_image[:,::-nx+1])
    valmax = max(1.,valmax)+1.E-9
    out_image *= 1./valmax
    return out_image[:,::-nx+1]


@numba.guvectorize(['(float64[:,:,:], float64[:,:], int64[:], float64[:,:])'], '(nx, ny, nk),(ni, nj), ()->(ny, nk)', target='parallel', nopython=True)
def convolve_matrix3_gu(image, convolution_array, index, out_image) :
    height, width, dim = image.shape
    nx     = convolution_array.shape[1]
    ny     = convolution_array.shape[0]
    half_x = nx//2
    half_y = ny//2
    for i in range(half_x, width - half_x):
        out_image[i-half_x,0] = 0.
        out_image[i-half_x,1] = 0.
        out_image[i-half_x,2] = 0.
        for jw in range(-half_y,ny-half_y):
            for iw in range(-half_x,nx-half_x):
                out_image[i-half_x,0] += convolution_array[jw+half_y,iw+half_x]*image[index[0]+jw,i+iw,0]
                out_image[i-half_x,1] += convolution_array[jw+half_y,iw+half_x]*image[index[0]+jw,i+iw,1]
                out_image[i-half_x,2] += convolution_array[jw+half_y,iw+half_x]*image[index[0]+jw,i+iw,2]
        out_image[i-half_x,0] = np.abs(out_image[i-half_x,0])
        out_image[i-half_x,1] = np.abs(out_image[i-half_x,1])
        out_image[i-half_x,2] = np.abs(out_image[i-half_x,2])

@numba.jit(nogil=True)
def convolve_matrix3(image, convolution_array) :
    height, width, d = image.shape
    nx     = convolution_array.shape[1]
    ny     = convolution_array.shape[0]
    half_x = nx//2
    half_y = ny//2
    out_image = np.empty((height-ny+1,width, d))
    index = np.arange(half_y, height - half_y)
    convolve_matrix3_gu(image, convolution_array, index, out_image)
    # On renormalise l'image en ramenant les valeurs des couleurs entre 0 et 1
    #out_image = np.abs(out_image)
    valmax = np.max(out_image[:,::-nx+1])
    valmax = max(1.,valmax)+1.E-9
    out_image *= 1./valmax
    return out_image[:,::-nx+1]

