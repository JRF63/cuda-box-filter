#include "info.h"
#include "errors.h"

#include "cuda_runtime.h"

constexpr size_t combineMajorMinor(int major, int minor) {
	return static_cast<size_t>(major) << 32 | minor;
}

// CUDA C Programming Guide Table 15. Technical Specifications per Compute Capability
size_t maxConcurrentKernel() {
	int dev;
	int major = 0;
	int minor = 0;

	CUDA_CHECK(cudaGetDevice(&dev));
	CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
	CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));

	size_t computeCap = combineMajorMinor(major, minor);
	size_t num;
	switch (computeCap) {
	case combineMajorMinor(5, 0):
	case combineMajorMinor(5, 2):
		num = 32;
		break;
	case combineMajorMinor(5, 3):
		num = 16;
		break;
	case combineMajorMinor(6, 0):
		num = 128;
		break;
	case combineMajorMinor(6, 1):
		num = 32;
		break;
	case combineMajorMinor(6, 2):
		num = 16;
		break;
	case combineMajorMinor(7, 0):
		num = 128;
		break;
	case combineMajorMinor(7, 2):
		num = 16;
		break;
	case combineMajorMinor(7, 5):
	case combineMajorMinor(8, 0):
	case combineMajorMinor(8, 6):
	case combineMajorMinor(8, 7):
	case combineMajorMinor(8, 9):
	case combineMajorMinor(9, 0):
		num = 128;
		break;
	default:
		num = 128; // Future versions
		break;
	}
	return num;
}
