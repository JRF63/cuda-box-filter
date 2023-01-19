#pragma once

#include "errors.h"

#include "cuda_runtime.h"
#include "npp.h"

#include <cstdlib>
#include <cstdio>

void cudaCheck(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		fprintf(
			stderr,
			"CUDA error at %s:%d code=%d (%s) \"%s\" \n",
			file,
			line,
			static_cast<unsigned int>(result),
			cudaGetErrorName(result),
			func
		);
		exit(EXIT_FAILURE);
	}
}

void nppCheck(NppStatus status, char const* const func, const char* const file, int const line) {
	if (status != NPP_SUCCESS) {
		fprintf(
			stderr,
			"CUDA error at %s:%d code=%d \"%s\" \n",
			file,
			line,
			static_cast<int>(status),
			func
		);
		exit(EXIT_FAILURE);
	}
}
