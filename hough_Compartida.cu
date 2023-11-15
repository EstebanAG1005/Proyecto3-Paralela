#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc)
{
    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    *acc = new int[rBins * degreeBins];
    memset(*acc, 0, sizeof(int) * rBins * degreeBins);
    int xCent = w / 2;
    int yCent = h / 2;
    float rScale = 2 * rMax / rBins;

    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++)
        {
            int idx = j * w + i;
            if (pic[idx] > 0)
            {
                int xCoord = i - xCent;
                int yCoord = yCent - j;
                float theta = 0;
                for (int tIdx = 0; tIdx < degreeBins; tIdx++)
                {
                    float r = xCoord * cos(theta) + yCoord * sin(theta);
                    int rIdx = (r + rMax) / rScale;
                    (*acc)[rIdx * degreeBins + tIdx]++;
                    theta += radInc;
                }
            }
        }
}

__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

__global__ void GPU_HoughTranShared(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{

    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID > w * h)
        return;

    int i;
    int locID = threadIdx.x;
    int xCent = w / 2;
    int yCent = h / 2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    __shared__ int localAcc[degreeBins * rBins];

    for (i = locID; i < degreeBins * rBins; i += blockDim.x)
        localAcc[i] = 0;

    __syncthreads();

    if (pic[gloID] > 0)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            atomicAdd(localAcc + (rIdx * degreeBins + tIdx), 1);
        }
    }

    __syncthreads();

    for (i = locID; i < degreeBins * rBins; i += blockDim.x)
        atomicAdd(acc + i, localAcc[i]);
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

int main(int argc, char **argv)
{
    int i;

    PGMImage inImg(argv[1]);

    int *cpuht;
    int w = inImg.x_dim;
    int h = inImg.y_dim;

    CPU_HoughTran(inImg.pixels, w, h, &cpuht);

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

    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    h_in = inImg.pixels;

    h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    // Define CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, NULL);

    int blockNum = ceil(w * h / 256);
    GPU_HoughTranShared<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale);

    // Record the stop event
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    // Calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaDeviceSynchronize();

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
    const int staticThreshold = 3000;                          // Static threshold set to 3000

    for (int i = 0; i < w * h; ++i)
    {
        outputImage[3 * i] = inImg.pixels[i];
        outputImage[3 * i + 1] = inImg.pixels[i];
        outputImage[3 * i + 2] = inImg.pixels[i];
    }

    // Dibujar las líneas cuyo peso es mayor que el umbral dinámico
    // Dibujar las líneas cuyo peso es mayor que el umbral estático
    for (int rIdx = 0; rIdx < rBins; rIdx++)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            if (h_hough[rIdx * degreeBins + tIdx] > staticThreshold)
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
    stbi_write_png("output_image_compartida.png", w, h, 3, outputImage, w * 3);

    // Liberar la memoria utilizada
    delete[] outputImage;
    for (i = 0; i < degreeBins * rBins; i++)
    {
        if (cpuht[i] != h_hough[i])
            printf("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
    }
    printf("Done!\n");
    printf("GPU Hough Transform tomo %f milisegundos\n", milliseconds);
    cudaFree(d_in);
    cudaFree(d_hough);
    free(h_hough);
    free(cpuht);
    free(pcCos);
    free(pcSin);

    // Destroy the events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}