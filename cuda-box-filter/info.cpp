#include "info.h"
#include "errors.h"

#include "cuda_runtime.h"
#include "npp.h"

#include <cstdio>

inline bool checkCudaCapabilities(int major_version, int minor_version) {
	int dev;
	int major = 0;
	int minor = 0;

	CUDA_CHECK(cudaGetDevice(&dev));
	CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
	CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));

	if ((major > major_version) || (major == major_version && minor >= minor_version)) {
		printf("  Device %d: %s, Compute SM %d.%d detected\n", dev, nppGetGpuName(), major, minor);
		return true;
	}
	else {
		printf(
			"  No GPU device was found that can support "
			"CUDA compute capability %d.%d.\n",
			major_version,
			minor_version
		);
		return false;
	}
}

bool printNPPinfo() {
	const NppLibraryVersion* libVer = nppGetLibVersion();

	printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

	int driverVersion;
	int runtimeVersion;
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);

	printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
	printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

	// Min spec is SM 1.0 devices
	bool bVal = checkCudaCapabilities(1, 0);
	return bVal;
}