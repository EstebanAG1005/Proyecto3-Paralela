/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb_image_write.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc)
{
    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
    *acc = new int[rBins * degreeBins];                // el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
    memset(*acc, 0, sizeof(int) * rBins * degreeBins); // init en ceros
    int xCent = w / 2;
    int yCent = h / 2;
    float rScale = 2 * rMax / rBins;

    for (int i = 0; i < w; i++)     // por cada pixel
        for (int j = 0; j < h; j++) //...
        {
            int idx = j * w + i;
            if (pic[idx] > 0) // si pasa thresh, entonces lo marca
            {
                int xCoord = i - xCent;
                int yCoord = yCent - j;                       // y-coord has to be reversed
                float theta = 0;                              // actual angle
                for (int tIdx = 0; tIdx < degreeBins; tIdx++) // add 1 to all lines in that pixel
                {
                    float r = xCoord * cos(theta) + yCoord * sin(theta);
                    int rIdx = (r + rMax) / rScale;
                    (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                    theta += radInc;
                }
            }
        }
}

//*****************************************************************
// TODO usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
//__constant__ float d_Cos[degreeBins];
//__constant__ float d_Sin[degreeBins];

//*****************************************************************
// TODO Kernel memoria compartida
// __global__ void GPU_HoughTranShared(...)
// {
//   //TODO
// }
// TODO Kernel memoria Constante
// __global__ void GPU_HoughTranConst(...)
// {
//   //TODO
// }

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
{
    // TODO calcular: int gloID = ?
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h)
        return;

    int xCent = w / 2;
    int yCent = h / 2;

    // TODO explicar bien bien esta parte. Dibujar un rectangulo a modo de imagen sirve para visualizarlo mejor
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    // TODO eventualmente usar memoria compartida para el acumulador

    if (pic[gloID] > 0)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            // TODO utilizar memoria constante para senos y cosenos
            // float r = xCoord * cos(tIdx) + yCoord * sin(tIdx); //probar con esto para ver diferencia en tiempo
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            // debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
            atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
        }
    }

    // TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
    // utilizar operaciones atomicas para seguridad
    // faltara sincronizar los hilos del bloque en algunos lados
}

// constant memory
__constant__ float dCos[degreeBins];
__constant__ float dSin[degreeBins];

__global__ void GPU_HoughTranConst(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
    // We get the global ID
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;

    if (gloID >= w * h)
        return; // if the global id is greater than the number of pixels we return

    int xCent = w / 2;
    int yCent = h / 2;

    // We get the coordinates of the pixel
    // The x coordinate is obtained by means of the modulo operation with the width of the image. doing the remainder we obtain the column and the subtraction is even to centralize the coordinate
    int xCoord = gloID % w - xCent;
    // The y-coordinate is obtained by means of the integer division operation with the width of the image. We do the integer division to obtain the row and the subtraction is to centralize the coordinate
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            // we calculate the radius
            float r = xCoord * dCos[tIdx] + yCoord * dSin[tIdx];
            // we calculate the index of the radius
            int rIdx = (r + rMax) / rScale;
            // Because it is done based on the angles, it may be that at some point they will touch, which is why an atomic add is done
            atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
        }
    }
}

float calculateAverage(int *array, int size)
{
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += array[i];
    }
    return sum / size;
}

float calculateStdDev(int *array, int size, float average)
{
    float variance = 0;
    for (int i = 0; i < size; i++)
    {
        variance += pow(array[i] - average, 2);
    }
    return sqrt(variance / size);
}

// Función para convertir la imagen a blanco y negro
void convertToBlackAndWhite(unsigned char *pic, int size, unsigned char threshold)
{
    for (int i = 0; i < size; i++)
    {
        pic[i] = pic[i] > threshold ? 255 : 0;
    }
}
//*****************************************************************
int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <image.pgm>\n", argv[0]);
        return -1;
    }

    int i;

    PGMImage inImg(argv[1]);

    int *cpuht;
    int w = inImg.x_dim;
    int h = inImg.y_dim;

    float *d_Cos;
    float *d_Sin;

    cudaMalloc((void **)&d_Cos, sizeof(float) * degreeBins);
    cudaMalloc((void **)&d_Sin, sizeof(float) * degreeBins);

    // CPU calculation
    CPU_HoughTran(inImg.pixels, w, h, &cpuht);

    // pre-compute values to be stored
    float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (i = 0; i < degreeBins; i++)
    {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

    unsigned char *d_in;
    int *d_hough;
    int *h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, inImg.pixels, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    // Define CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, NULL);

    // Launch the kernel
    int blockNum = ceil((float)w * h / 256.0);
    GPU_HoughTranConst<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale);

    // Record the stop event
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    // Calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy results back to host
    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // Calcular el promedio y la desviación estándar
    const int arraySize = degreeBins * rBins;
    float average = calculateAverage(h_hough, arraySize);
    float stdDev = calculateStdDev(h_hough, arraySize, average);

    // Convertir la imagen a blanco y negro
    unsigned char threshold = 10; // Ajuste este valor según sea necesario
    convertToBlackAndWhite(inImg.pixels, w * h, threshold);

    // Crear una copia de la imagen de entrada para dibujar las líneas
    unsigned char *outputImage = new unsigned char[w * h * 3]; // 3 canales: RGB
    for (int i = 0; i < w * h; ++i)
    {
        outputImage[3 * i] = inImg.pixels[i];
        outputImage[3 * i + 1] = inImg.pixels[i];
        outputImage[3 * i + 2] = inImg.pixels[i];
    }

    // Dibujar las líneas cuyo peso es mayor que el umbral dinámico
    for (int rIdx = 0; rIdx < rBins; rIdx++)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            if (h_hough[rIdx * degreeBins + tIdx] > threshold)
            {
                float r = rIdx * rScale - rMax;
                float theta = tIdx * radInc;

                for (int x = 0; x < w; x++)
                {
                    int y = (int)((r - x * cos(theta)) / sin(theta));
                    if (y >= 0 && y < h)
                    {
                        int idx = y * w + x;
                        if (inImg.pixels[idx] > 0)
                        {
                            idx *= 3;
                            outputImage[idx] = 255;   // R
                            outputImage[idx + 1] = 0; // G
                            outputImage[idx + 2] = 0; // B
                        }
                    }
                }
            }
        }
    }

    // Guardar la imagen resultante en formato PNG
    stbi_write_png("output_image_constante.png", w, h, 3, outputImage, w * 3);

    // Liberar memoria
    delete[] outputImage;

    const int tolerance = 1; // Define un margen de tolerancia

    // Compare CPU and GPU results
    for (i = 0; i < degreeBins * rBins; i++)
    {
        if (abs(cpuht[i] - h_hough[i]) > tolerance)
        {
            printf("Mismatch at index %d: CPU=%d, GPU=%d\n", i, cpuht[i], h_hough[i]);
        }
    }

    printf("GPU Hough Transform tomo %f milisegundos\n", milliseconds);

    // Free dynamically allocated memory
    cudaFree(d_Cos);
    cudaFree(d_Sin);
    cudaFree(d_in);
    cudaFree(d_hough);
    free(pcCos);
    free(pcSin);
    free(h_hough);
    delete[] cpuht;

    // Destroy the events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Done!\n");

    return 0;
}
