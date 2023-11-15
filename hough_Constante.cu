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
#include <algorithm>

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
// __global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
// {
//     // TODO calcular: int gloID = ?
//     int gloID = blockIdx.x * blockDim.x + threadIdx.x;
//     if (gloID >= w * h)
//         return;

//     int xCent = w / 2;
//     int yCent = h / 2;

//     // TODO explicar bien bien esta parte. Dibujar un rectangulo a modo de imagen sirve para visualizarlo mejor
//     int xCoord = gloID % w - xCent;
//     int yCoord = yCent - gloID / w;

//     // TODO eventualmente usar memoria compartida para el acumulador

//     if (pic[gloID] > 0)
//     {
//         for (int tIdx = 0; tIdx < degreeBins; tIdx++)
//         {
//             // TODO utilizar memoria constante para senos y cosenos
//             // float r = xCoord * cos(tIdx) + yCoord * sin(tIdx); //probar con esto para ver diferencia en tiempo
//             float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
//             int rIdx = (r + rMax) / rScale;
//             // debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
//             atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
//         }
//     }

//     // TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
//     // utilizar operaciones atomicas para seguridad
//     // faltara sincronizar los hilos del bloque en algunos lados
// }

// Memoria Constante
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

__global__ void GPU_HoughTranConst(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{

    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID > w * h)
        return;
    int xCent = w / 2;
    int yCent = h / 2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    __syncthreads();

    if (pic[gloID] > 0)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {

            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;

            atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
        }
    }
}

// Función para convertir la imagen a blanco y negro
void convertToBlackAndWhite(unsigned char *pic, int size, unsigned char threshold)
{
    for (int i = 0; i < size; i++)
    {
        pic[i] = pic[i] > threshold ? 255 : 0;
    }
}

void drawLines(unsigned char *outputImage, int w, int h, int *h_hough, float rMax, float rScale)
{
    // Define a threshold for considering a line as detected
    const int detectionThreshold = 3800;

    // Iterate through the Hough space to find lines with sufficient votes
    for (int rIdx = 0; rIdx < rBins; rIdx++)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            if (h_hough[rIdx * degreeBins + tIdx] > detectionThreshold)
            {
                // Convert Hough space coordinates back to image space
                float theta = tIdx * radInc;
                float r = rIdx * rScale - rMax;

                // Calculate the coordinates of two points on the line
                int x0 = static_cast<int>(w / 2 + r * cos(theta));
                int y0 = static_cast<int>(h / 2 - r * sin(theta));

                int x1 = static_cast<int>(x0 - (w / 2) * (-sin(theta)));
                int y1 = static_cast<int>(y0 + (h / 2) * (cos(theta)));

                // Clip the line coordinates to be within the image boundaries
                x0 = std::max(0, std::min(x0, w - 1));
                y0 = std::max(0, std::min(y0, h - 1));
                x1 = std::max(0, std::min(x1, w - 1));
                y1 = std::max(0, std::min(y1, h - 1));

                // Draw the line on the output image
                for (int i = 0; i < 2000; i++)
                {
                    int x = static_cast<int>(x0 + i * (x1 - x0) / (w / 2));
                    int y = static_cast<int>(y0 + i * (y1 - y0) / (h / 2));

                    // Ensure that the coordinates are within the image boundaries
                    if (x >= 0 && x < w && y >= 0 && y < h)
                    {
                        outputImage[3 * (y * w + x)] = 0;       // Canal rojo
                        outputImage[3 * (y * w + x) + 1] = 255; // Canal verde
                        outputImage[3 * (y * w + x) + 2] = 255; // Canal azul
                    }
                }
            }
        }
    }
    // Define a threshold for considering a line as detected
    // const int detectionThreshold = 3800;

    // Iterate through the Hough space to find lines with sufficient votes
    for (int rIdx = 0; rIdx < rBins; rIdx++)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            if (h_hough[rIdx * degreeBins + tIdx] > detectionThreshold)
            {
                // Convert Hough space coordinates back to image space
                float theta = tIdx * radInc;
                float r = rIdx * rScale - rMax;

                // Calculate the coordinates of two points on the line
                int x0 = static_cast<int>(w / 2 + r * cos(theta));
                int y0 = static_cast<int>(h / 2 - r * sin(theta));

                int x1 = static_cast<int>(x0 - (w / 2) * (-sin(theta)));
                int y1 = static_cast<int>(y0 + (h / 2) * (cos(theta)));

                // Clip the line coordinates to be within the image boundaries
                x0 = std::max(0, std::min(x0, w - 1));
                y0 = std::max(0, std::min(y0, h - 1));
                x1 = std::max(0, std::min(x1, w - 1));
                y1 = std::max(0, std::min(y1, h - 1));

                // Draw the line on the output image
                for (int i = 0; i < 2000; i++)
                {
                    int x = w - static_cast<int>(x0 + i * (x1 - x0) / (w / 2));
                    int y = static_cast<int>(y0 + i * (y1 - y0) / (h / 2));

                    // Ensure that the coordinates are within the image boundaries
                    if (x >= 0 && x < w && y >= 0 && y < h)
                    {
                        outputImage[3 * (y * w + x)] = 0;       // Canal rojo
                        outputImage[3 * (y * w + x) + 1] = 255; // Canal verde
                        outputImage[3 * (y * w + x) + 2] = 255; // Canal azul
                    }
                }
            }
        }
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

    GPU_HoughTranConst<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale);

    // Record the stop event
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    // Calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaDeviceSynchronize();

    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    unsigned char threshold = 10;
    convertToBlackAndWhite(inImg.pixels, w * h, threshold);
    unsigned char *outputImage = new unsigned char[w * h * 3];
    for (int i = 0; i < w * h; ++i)
    {
        outputImage[3 * i] = inImg.pixels[i];
        outputImage[3 * i + 1] = inImg.pixels[i];
        outputImage[3 * i + 2] = inImg.pixels[i];
    }

    drawLines(outputImage, w, h, h_hough, rMax, rScale);
    // Guardar la imagen resultante en formato PNG
    stbi_write_png("output_image_constante.png", w, h, 3, outputImage, w * 3);

    // Liberar la memoria utilizada
    delete[] outputImage;

    const int tolerance = 1;

    for (int i = 0; i < degreeBins * rBins; i++)
    {
        // Calcula la diferencia absoluta entre los dos valores
        int diff = abs(cpuht[i] - h_hough[i]);

        // Verifica si la diferencia excede la tolerancia
        if (diff > tolerance)
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

    cudaDeviceReset();

    return 0;
}