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


#ifndef KTEST_PATTERN_H__
#define KTEST_PATTERN_H__

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <list>
#include <map>
#include <string>

#include <clblas-internal.h>
#include <blas_funcs.h>
#include <granulation.h>
#include <kernel_extra.h>
#include <solution_seq.h>
#include <mempat.h>
#include <list.h>
#include "var.h"

namespace clMath {

// This structure reflects CLBlasKargs structure, declared in clblas-internal.h
typedef struct StepKargs {
    // kernType
    // dtype
    // order
    // side
    // uplo
    // transA
    // transB
    // diag
    Variable *M;
    Variable *N;
    Variable *K;
    Variable *alpha;
    Variable *A;
    Variable *lda;
    Variable *B;
    Variable *ldb;
    Variable *beta;
    Variable *C;
    Variable *ldc;
    // addrBits
    Variable *offsetM;
    Variable *offsetN;
    Variable *offsetK;
    Variable *scimage0;
    Variable *scimage1;
    Variable *offA;
    Variable *offBX;
    Variable *offCY;
} Kargs;

typedef std::list<Variable*> VarList;
typedef std::list<ArrayVariableInterface*> ArrayVarList;
typedef std::map<unsigned int, const Variable*> KArgMap;

/**
 * @internal
 * @brief SolutionStep wrapper object
 * @ingroup MAKE_KTEST
 *
 * Objects of this class are used for problem decomposition. Each Step object
 * contains single SolutionStep structure. For disabled multikernel feature
 * case there is only one solution step always. For multikernel case there is
 * one master step storing arguments of original problem and inner steps
 * which are received from solution sequence list generated by clBLAS in
 * makeSolutionSequence call.
 *
 */
class Step {
private:
    CLBLASKernExtra kextra_;
    cl_platform_id platform_;

    VarList vars_;
    ArrayVarList arrays_;
    VarList buffers_;
    /**
     * @internal
     * @brief Kernel arguments map
     *
     * Contains variables objects for arguments of step kernel,
     * in respective order.
     */
    KArgMap kargMap_;

    std::string dumpMemoryPattern();
    std::string dumpSubdim(const SubproblemDim *subdim);
    std::string dumpPgran();
    std::string dumpKextra();

    cl_device_id device()                       { return step_.device.id; };
    void setKernelArg(unsigned int index, const Variable *var);

protected:
    /**
     * @internal
     * @brief Associated SolutionStep structure
     */
    SolutionStep step_;
    /**
     * @internal
     * @brief Selected memory pattern pointer
     */
    MemoryPattern* pattern_;
    /**
     * @internal
     * @brief Naive call string
     *
     * This string contains naive call for processing step problem. Is used in
     * master step.
     */
    std::string naiveCall_;
    /**
     * @internal
     * @brief Comparison call string
     *
     * This string contains comparison call for processed matrixes. Is used in
     * master step.
     */
    std::string compareCall_;
    /**
     * @internal
     * @brief Post process matrixes
     *
     * This string contains function call for post-processing matrixes after
     * filling them with random data. Can be used in master step.
     */
    std::string postRandomCall_;
    /**
     * @internal
     * @brief Step kernel name
     */
    std::string kernelName_;


    /**
     * @internal
     * @brief Add variable into step variables list.
     *
     * Variable value is given by string.
     */
    Variable* addVar(const std::string& name, const std::string& type,
        const std::string& defaultValue = "");
    /**
     * @internal
     * @brief Add variable into step variables list.
     *
     * Variable value is given by unsigned value.
     */
    Variable* addVar(const std::string& name, const std::string& type,
        size_t value);
    /**
     * @internal
     * @brief Add variable into step variables list.
     *
     * Variable value is given by signed integer.
     */
    Variable* addVar(const std::string& name, const std::string& type,
        int value);
    /**
     * @internal
     * @brief Add constant variable into step variables list.
     *
     * Constant value is given by string.
     */
    Variable* addConst(const std::string& name, const std::string& type,
        const std::string& defaultValue);
    /**
     * @internal
     * @brief Add constant variable into step variables list.
     *
     * Constant value is given by unsigned value.
     */
    Variable* addConst(const std::string& name, const std::string& type,
        size_t value);
    /**
     * @internal
     * @brief Add constant variable into step variables list.
     *
     * Constant value is given by signed integer.
     */
    Variable* addConst(const std::string& name, const std::string& type,
        int value);
    /**
     * @internal
     * @brief Add matrix array into step host arrays list.
     */
    MatrixVariable* addMatrix(const std::string& name, const std::string& type,
        Variable *rows, Variable *columns, Variable *ld, Variable *off = NULL);
    /**
     * @internal
     * @brief Add vector into step host arrays list.
     */
    VectorVariable* addVector(const std::string& name, const std::string& type,
        Variable *N, Variable *inc, Variable *off = NULL);
    /**
     * @internal
     * @brief Add variable for OpenCL buffer into step buffers list.
     */
    Variable* addBuffer(BufferID bufID, const std::string& name,
        const std::string& type, cl_mem_flags flags,
        ArrayVariableInterface* hostPtr);

    /**
     * @internal
     * @brief Assign kernel arguments
     *
     * Run pattern assign-kernel-arguments function and get information about
     * used variables and their order which is used for generating kernel test
     * code.
     */
    void assignKargs(const Kargs& kargs);

    /**
     * @internal
     * @brief Get device vendor string
     */
    static std::string deviceVendor(cl_device_id device);

public:
    /**
     * @internal
     * @brief Constructor for master step
     *
     * @param[in] funcID          Function identifier
     * @param[in] device          Device identifier
     *
     * Uses function id and device to compose step object. It is used for
     * master step.
     *
     */
    Step(BlasFunctionID funcID, cl_device_id device);
    /**
     * @internal
     * @brief Constructor for inner step
     *
     * @param[in] node            Solution sequence list node
     *
     * Uses solution sequence node to compose step object. It is used for
     * making inner steps from solution sequence list received from
     * clBLAS frontend using makeSolutionSequence.
     *
     */
    Step(ListNode *node);
    /**
     * @internal
     * @brief Step destructor
     */
    virtual ~Step();

    /**
     * @internal
     * @brief Get step variables list
     */
    const VarList& vars() const                 { return vars_; };
    /**
     * @internal
     * @brief Get step host arrays list
     */
    const ArrayVarList& arrays() const          { return arrays_; };
    /**
     * @internal
     * @brief Get step OpenCL buffers list
     */
    const VarList& buffers() const              { return buffers_; };

    /**
     * @internal
     * @brief Fix leading dimensions to fit matrixes sizes
     */
    virtual void fixLD() = 0;
    /**
     * @internal
     * @brief Declare variables
     *
     * @param[in] masterStep      Master step object
     *
     * Add function-specific variables and fill comparison call and naive
     * implementation call strings. Master step object is used for handling
     * buffers A, B, C rearrangement.
     *
     */
    virtual void declareVars(Step *masterStep) = 0;
    /**
     * @internal
     * @brief Get buffer by id
     *
     * @param[in] bufID           Buffer identifier
     *
     * Return variable of step for buffer A, B or C. Is used for multi-step
     * configurations for handling buffers rearrangement in inner steps. Inner
     * steps get buffer variables names from respective master step buffers.
     */
    Variable* getBuffer(BufferID bufID);

    /**
     * @internal
     * @brief Complete problem decomposition of a single step
     *
     * Parallelism granularity, tails flags and vectorization values are
     * guaranteed to be set in appropriate values after this function call.
     */
    void completeDecompositionSingle();
    /**
     * @internal
     * @brief Wrapper for makeSolutionSeq
     *
     * @param[out] seq             Solution sequence list head
     * @param[in]  platform        Platform identifier
     *
     * Call makeSolutionSeq from clBLAS frontend and return solution sequence
     * list for it.
     */
    void makeSolutionSequence(ListHead *seq, cl_platform_id platform);
    /**
     * @internal
     * @brief Wrapper for freeSolutionSeq
     *
     * @param[out] seq             Solution sequence list head
     *
     * Call freeSolutionSeq from clBLAS frontend.
     */
    void freeSolutionSequence(ListHead *seq);
    /**
     * @internal
     * @brief Generate step kernel code
     *
     * @return String containing kernel code for this step
     */
    std::string generate();
    /**
     * @internal
     * @brief Generate step global work size string
     *
     * @return String containing global work size for this step
     */
    std::string globalWorkSize();

    /**
     * @internal
     * @brief Get step blas function identifier
     * @return blas function id
     */
    BlasFunctionID blasFunctionID() const       { return step_.funcID; };
    /**
     * @internal
     * @brief Get step kernel arguments
     * @return step kernel arguments structure
     */
    const CLBlasKargs& kargs() const            { return step_.args; };
    /**
     * @internal
     * @brief Get step parallelism granularity
     * @return step parallelism granularity structure
     */
    const PGranularity& pgran() const           { return step_.pgran; };
    /**
     * @internal
     * @brief Get naive call string
     *
     * Get string containing naive blas function call for step blas function
     * with respective step flags and arguments.
     * @return naive blas call string
     */
    const std::string& naiveCall() const        { return naiveCall_; };
    /**
     * @internal
     * @brief Get comparison call string
     *
     * Get string containing resulting vectors of matrixes comparison function
     * call for step blas function.
     * @return comparison call string
     */
    const std::string& compareCall() const      { return compareCall_; };
    /**
     * @internal
     * @brief Get post-processing call
     *
     * Get string containing function call which is called after setting step
     * matrixes. Is used in TRSM now for making divisible B matrix.
     * @return step matrixes post-processing call
     */
    const std::string& postRandomCall() const   { return postRandomCall_; };
    /**
     * @internal
     * @brief Get step kernel name
     * @return step kernel name
     */
    const std::string& kernelName() const       { return kernelName_; };
    /**
     * @internal
     * @brief Get blas function name
     * Returns blas function name from naive blas for this step.
     * @return blas function name
     */
    const char* getBlasFunctionName();

    /**
     * @internal
     * @brief Get kernel arguments variables map
     * @return step kernel arguments variables map
     */
    const std::map<unsigned int, const Variable*>& kargMap() const { return kargMap_; }
    /**
     * @internal
     * @brief Set step blas arguments
     * @param[in]  kargs           Step blas arguments structure
     */
    void setKargs(const CLBlasKargs& kargs);
    /**
     * @internal
     * @brief Set step blas subdimensions
     * @param[in]  subdims         Step subproblem dimensions
     */
    void setDecomposition(const SubproblemDim *subdims);
    /**
     * @internal
     * @brief Set step kernel name
     * @param[in]  name            Step kernel name
     */
    void setKernelName(std::string name);
    /**
     * @internal
     * @brief Get string containing matrix size
     * @param[in]  var             Matrix variable
     * @return matrix variable size string
     */
    std::string matrixSize(MatrixVariable *var);
    /**
     * @internal
     * @brief Get string containing vector size
     * @param[in]  var             Vector variable
     * @return vector variable size string
     */
    std::string vectorSize(VectorVariable *vector);

    /**
     * @internal
     * @brief Get string containing argument value
     * @param[in]  dtype           Argument type
     * @param[in]  arg             Argument value
     * Get string containing argument value. Argument can have complex type.
     * @return string containing argument value
     */
    static std::string multiplierToString(DataType dtype, ArgMultiplier arg);
    /**
     * @internal
     * @brief Get string containing type
     * @param[in]  dtype           Data type
     * @return Data type string
     */
    static std::string dtypeToString(DataType dtype);
    /**
     * @internal
     * @brief Get solution node blas function identifier
     * @param[in]  node            Solution sequence node
     * @return blas function id
     */
    static BlasFunctionID getStepNodeFuncID(ListNode *node);
};

}   // namespace clMath

#endif  // KTEST_PATTERN_H__
