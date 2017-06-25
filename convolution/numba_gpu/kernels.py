import math
import numpy as np
from numba import cuda
import numba.types

TPB = 16
TPB_o = TPB+2
"""
Definit une convolution faisant une moyenne des voisins d'un pixel donne
( stencil de 3x3 )
"""
@cuda.jit
def convolve_mean2_kernel(image, out_image):
    height, width = image.shape
    data = cuda.shared.array((TPB_o, TPB_o), dtype=numba.types.float64)
    i, j = cuda.grid(2)
    itx, ity = cuda.threadIdx.x+1, cuda.threadIdx.y+1
    data[itx][ity] = image[i, j]
    if cuda.threadIdx.x == 0:
        data[0][ity] = image[i-1, j]
    if cuda.threadIdx.x == TPB-1:
        data[TPB_o-1][ity] = image[i+1, j]
    if cuda.threadIdx.y == 0:
        data[itx][0] = image[i, j-1]
    if cuda.threadIdx.y == TPB-1:
        data[itx][TPB_o-1] = image[i, j+1]
    
    cuda.syncthreads()

    if 0 < i < width-1 and 0 < j < height-1:
        out_image[i-1, j-1] = 0.25*(data[itx-1,ity]+data[itx+1,ity]+data[itx,ity-1]+data[itx,ity+1])


def convolve_mean2(image):
    height, width = image.shape
    #import time
    #t1 = time.time()
    d_image = cuda.to_device(image)
    d_output = cuda.device_array((height-2, width-2))
    #t2 = time.time()
    #print("1:", t2-t1)
    BPGx = (height + TPB - 1) // TPB
    BPGy = (width + TPB - 1) // TPB
    convolve_mean2_kernel[(BPGx, BPGy), (TPB, TPB)](d_image, d_output)
    #t1 = time.time()
    output = d_output.copy_to_host()
    #t2 = time.time()
    #print("2:", t2-t1)
    return output

ROW_TILE_W = 128
COLUMN_TILE_W = 16
COLUMN_TILE_H = 48
KERNEL_RADIUS = 1
KERNEL_RADIUS_ALIGNED = 2*KERNEL_RADIUS
SIZE_ROW = ROW_TILE_W + KERNEL_RADIUS_ALIGNED + KERNEL_RADIUS
SIZE_COL = COLUMN_TILE_W *(KERNEL_RADIUS + COLUMN_TILE_H + KERNEL_RADIUS)

@cuda.jit
def convolveRowGPU(image, out_image):
    height, width = image.shape
    
    data = cuda.shared.array(SIZE_ROW, dtype=numba.types.float64)

    tileStart = cuda.blockIdx.x*ROW_TILE_W
    tileEnd = tileStart + ROW_TILE_W - 1
    apronStart = tileStart - KERNEL_RADIUS
    apronEnd = tileEnd + KERNEL_RADIUS

    tileEndClamped = min(tileEnd, width - 1)
    apronStartClamped = max(apronStart, 0)
    apronEndClamped = min(apronEnd, width - 1)
    
    rowStart = cuda.blockIdx.y 
    apronStartAligned = tileStart - KERNEL_RADIUS_ALIGNED
    loadPos = apronStartAligned + cuda.threadIdx.x

    if loadPos >= apronStart:
        smemPos = loadPos - apronStart

        data[smemPos] = image[rowStart, loadPos] if loadPos >= apronStartClamped and loadPos <= apronEndClamped else 0

    cuda.syncthreads()

    writePos = tileStart + cuda.threadIdx.x

    if writePos <= tileEndClamped:
         smemPos = writePos - apronStart
         mysum = - data[smemPos - 1] + 4*data[smemPos] - data[smemPos + 1]
         out_image[rowStart, writePos] = mysum

@cuda.jit
def convolveColGPU(image, out_image):
    height, width = image.shape
    
    data = cuda.shared.array(SIZE_ROW, dtype=numba.types.float64)

    tileStart = cuda.blockIdx.x*ROW_TILE_W
    tileEnd = tileStart + ROW_TILE_W - 1
    apronStart = tileStart - KERNEL_RADIUS
    apronEnd = tileEnd + KERNEL_RADIUS

    tileEndClamped = min(tileEnd, height - 1)
    apronStartClamped = max(apronStart, 0)
    apronEndClamped = min(apronEnd, height - 1)
    
    colStart = cuda.blockIdx.y 
    apronStartAligned = tileStart - KERNEL_RADIUS_ALIGNED
    loadPos = apronStartAligned + cuda.threadIdx.x

    if loadPos >= apronStart:
        smemPos = loadPos - apronStart

        data[smemPos] = image[loadPos, colStart] if loadPos >= apronStartClamped and loadPos <= apronEndClamped else 0

    cuda.syncthreads()

    writePos = tileStart + cuda.threadIdx.x

    if writePos <= tileEndClamped:
         smemPos = writePos - apronStart
         mysum = - data[smemPos - 1] - data[smemPos + 1]
         out_image[writePos, colStart] += mysum


@cuda.jit
def convolve_laplacien_naive(image, out_image):
    height, width = image.shape
    x, y = cuda.grid(2)
    if x - 1 >= 0 and - x + height - 2 >= 0 and y-1 >= 0 and - y + width-1>=0: 
        out_image[x, y] = -image[x-1, y] - image[x+1,y]+4*image[x,y]-image[x,y-1]-image[x,y+1]


@cuda.jit
def convolve_laplacien2(image, out_image):
    height, width = image.shape
    data = cuda.shared.array((18, 18), dtype=numba.types.float64)            
    x0, y0 = cuda.grid(2)
    x0 = cuda.blockIdx.x * 16 + cuda.threadIdx.x
    y0 = cuda.blockIdx.y * 16 + cuda.threadIdx.y
    KERNEL_RADIUS = 1

    # case1: upper left
    x = x0 - KERNEL_RADIUS
    y = y0 - KERNEL_RADIUS
    if x < 0 or y < 0:
        data[cuda.threadIdx.x, cuda.threadIdx.y] = 0
    else:
        data[cuda.threadIdx.x, cuda.threadIdx.y] = image[x, y]

    # # case2: upper right
    # x = x0 + KERNEL_RADIUS
    # y = y0 - KERNEL_RADIUS
    # if x > height-1 or y < 0:
    #     data[cuda.threadIdx.x + 1, cuda.threadIdx.y] = 0
    # else:
    #     data[cuda.threadIdx.x + 1, cuda.threadIdx.y] = image[x, y]

    # # case3: lower left
    # x = x0 - KERNEL_RADIUS
    # y = y0 + KERNEL_RADIUS
    # if x < 0 or y > width:
    #     data[cuda.threadIdx.x, cuda.threadIdx.y + 1] = 0
    # else:
    #     data[cuda.threadIdx.x, cuda.threadIdx.y + 1] = image[x, y]

    # # case4: lower right
    # x = x0 + KERNEL_RADIUS;
    # y = y0 + KERNEL_RADIUS;
    # if x > height or y > width:
    #     data[cuda.threadIdx.x + 1, cuda.threadIdx.y + 1] = 0
    # else:
    #     data[cuda.threadIdx.x + 1][cuda.threadIdx.y + 1] = image[x, y]

    cuda.syncthreads()

    if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0:
        for i in range(16):
            for j in range(16):
                print(data[i, j], end=' ')
            print("\n")
    x = KERNEL_RADIUS + cuda.threadIdx.x
    y = KERNEL_RADIUS + cuda.threadIdx.y
    #print(x0, y0)
    #out_image[x0, y0] = (4*data[x,y]-data[x-1,y]-data[x+1,y]
    #                                -data[x,y-1]-data[x,y+1])
    #print(out_image[x0,y0])
    # On renormalise l'image :
    # valmax = np.zeros(1, dtype=np.float64)
    # cuda.atomic.max(valmax, 0., out_image)
    # valmax = max(1., valmax)+1.E-9
    # out_image *= 1./valmax

