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


#ifndef SYR2K_H_
#define SYR2K_H_

#include <gtest/gtest.h>
#include <clBLAS.h>
#include <common.h>
#include <BlasBase.h>
#include <ExtraTestSizes.h>
#include <common.h>

using namespace clMath;
using ::testing::TestWithParam;

class SYR2K : public TestWithParam<
    ::std::tr1::tuple<
        clblasOrder,     // order
        clblasUplo,      // uplo
        clblasTranspose, // transA
        int,                // N
        int,                // K
        ExtraTestSizes,
        int                 // numCommandQueues
        > > {
public:
    void getParams(TestParams *params)
    {
        memset(params, 0, sizeof(TestParams));

        params->order = order;
        params->uplo = uplo;
        params->transA = transA;
        params->seed = seed;
        params->N = N;
        params->K = K;
        params->offA = offA;
        params->offBX = offB;
        params->offCY = offC;
        params->lda = lda;
        params->ldb = ldb;
        params->ldc = ldc;
        params->rowsA = rowsA;
        params->columnsA = columnsA;
        params->rowsB = rowsB;
        params->columnsB = columnsB;
        params->rowsC = rowsC;
        params->columnsC = columnsC;
        params->alpha = paramAlpha;
        params->beta = paramBeta;
        params->numCommandQueues = numCommandQueues;
    }

protected:
    virtual void SetUp()
    {
        ExtraTestSizes extra;

        order = ::std::tr1::get<0>(GetParam());
        uplo = ::std::tr1::get<1>(GetParam());
        transA = ::std::tr1::get<2>(GetParam());
        N = ::std::tr1::get<3>(GetParam());
        K = ::std::tr1::get<4>(GetParam());
        extra = ::std::tr1::get<5>(GetParam());
        offA = extra.offA;
        offB = extra.offBX;
        offC = extra.offCY;
        lda = extra.strideA.ld;
        ldb = extra.strideBX.ld;
        ldc = extra.strideCY.ld;
        numCommandQueues = ::std::tr1::get<6>(GetParam());

        base = ::clMath::BlasBase::getInstance();
        seed = base->seed();

        useNumCommandQueues = base->useNumCommandQueues();
        if (useNumCommandQueues) {
            numCommandQueues = base->numCommandQueues();
        }

        useAlpha = base->useAlpha();
        if (useAlpha != 0) {
            paramAlpha = base->alpha();
        }
        useBeta = base->useBeta();
        if (useBeta != 0) {
            paramBeta = base->beta();
        }
        if (base->useN()) {
            N = base->N();
        }
        if (base->useK()) {
            K = base->K();
        }

        if (transA == clblasNoTrans) {
            rowsA = N;
            columnsA = K;
            rowsB = N;
            columnsB = K;
        }
        else {
            rowsA = K;
            columnsA = N;
            rowsB = K;
            columnsB = N;
        }
        rowsC = N;
        columnsC = N;

        switch (order) {
        case clblasRowMajor:
            lda = ::std::max(lda, columnsA);
            columnsA = lda;
            ldb = ::std::max(ldb, columnsB);
            columnsB = ldb;
            ldc = ::std::max(ldc, columnsC);
            columnsC = ldc;
            break;
        case clblasColumnMajor:
            lda = ::std::max(lda, rowsA);
            rowsA = lda;
            ldb = ::std::max(ldb, rowsB);
            rowsB = ldb;
            ldc = ::std::max(ldc, rowsC);
            rowsC = ldc;
            break;
        }

        printTestParams(order, uplo, transA, N, K, useAlpha, base->alpha(),
                        offA, lda, offB, ldb, useBeta, base->beta(), offC, ldc);
        ::std::cerr << "seed = " << seed << ::std::endl;
        ::std::cerr << "queues = " << numCommandQueues << ::std::endl;
    }

    clblasOrder order;
    clblasUplo uplo;
    clblasTranspose transA;
    size_t N, K;
    size_t offA, offB, offC;
    size_t lda, ldb, ldc;
    unsigned int seed;

    bool useAlpha, useBeta;
    ComplexLong paramAlpha, paramBeta;

    size_t rowsA, columnsA;
    size_t rowsB, columnsB;
    size_t rowsC, columnsC;

    ::clMath::BlasBase *base;
    cl_ulong imageA, imageB;

    bool useNumCommandQueues;
    cl_uint numCommandQueues;
};

#endif  // SYR2K_H_
