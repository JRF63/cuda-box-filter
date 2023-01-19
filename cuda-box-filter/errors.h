#pragma once

#include "cuda_runtime.h"
#include "npp.h"

void cudaCheck(cudaError_t result, char const* const func, const char* const file, int const line);

#define CUDA_CHECK(val) cudaCheck((val), #val, __FILE__, __LINE__)

void nppCheck(NppStatus status, char const* const func, const char* const file, int const line);

#define NPP_CHECK(val) nppCheck((val), #val, __FILE__, __LINE__)