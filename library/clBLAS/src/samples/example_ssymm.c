/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/


#include <sys/types.h>
#include <stdio.h>
#include <string.h>

/* Include CLBLAS header. It automatically includes needed OpenCL header,
 * so we can drop out explicit inclusion of cl.h header.
 */
#include <clBLAS.h>

/* This example uses predefined matrices and their characteristics for
 * simplicity purpose.
 */
static const clblasOrder order = clblasRowMajor;

#define M  4
#define N  3

static const cl_float alpha = 10;

static const clblasSide side = clblasLeft;
static const clblasUplo uplo = clblasLower;
static const cl_float A[M*M] = {
    11, -1, -1, -1,
    21, 22, -1, -1,
    31, 32, 33, -1,
    41, 42, 43, 44,
};
static const size_t lda = M;

static const cl_float B[M*N] = {
    11, 12, 13,
    21, 22, 23,
    31, 32, 33,
    41, 42, 43,
};
static const size_t ldb = N;

static const cl_float beta = 20;

static cl_float C[M*N] = {
    11, 12, 13,
    21, 22, 23,
    31, 32, 33,
    41, 42, 43,
};
static const size_t ldc = N;

static void
printResult(void)
{
    size_t i, j, nrows;

    printf("Result:\n");

    nrows = (sizeof(C) / sizeof(cl_float)) / ldc;
    for (i = 0; i < nrows; i++) {
        for (j = 0; j < ldc; j++) {
            printf("%d ", (int)C[i * ldc + j]);
        }
        printf("\n");
    }
}

int
main(void)
{
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufA, bufB, bufC;
    cl_event event = NULL;
    int ret = 0;

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetPlatformIDs() failed with %d\n", err );
        return 1;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetDeviceIDs() failed with %d\n", err );
        return 1;
    }

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateContext() failed with %d\n", err );
        return 1;
    }

    queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateCommandQueue() failed with %d\n", err );
        clReleaseContext(ctx);
        return 1;
    }

    /* Setup clblas. */
    err = clblasSetup();
    if (err != CL_SUCCESS) {
        printf("clblasSetup() failed with %d\n", err);
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        return 1;
    }

    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M * M * sizeof(*A),
                          NULL, &err);
    bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M * N * sizeof(*B),
                          NULL, &err);
    bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, M * N * sizeof(*C),
                          NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
        M * M * sizeof(*A), A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0,
        M * N * sizeof(*B), B, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0,
        M * N * sizeof(*C), C, 0, NULL, NULL);

    /* Call clblas function. */
    err = clblasSsymm(order, side, uplo, M, N, alpha, bufA,
                         0, lda, bufB, 0, ldb, beta, bufC, 0, ldc, 1, &queue,
                         0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSsymm() failed with %d\n", err);
        ret = 1;
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);

        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, M * N * sizeof(*C),
                                  C, 0, NULL, NULL);

        /* At this point you will get the result of SYMM placed in C array. */
        printResult();
    }

    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufC);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufA);

    /* Finalize work with clblas. */
    clblasTeardown();

    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return ret;
}
