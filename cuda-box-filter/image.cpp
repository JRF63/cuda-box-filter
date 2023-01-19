#include "image.h"

cudaError_t ImageGpu::copyFromCpuAsync(const ImageCpu& other, cudaStream_t stream) {
	return cudaMemcpy2DAsync(
		data(),
		step(),
		other.data(),
		other.step(),
		other.widthBytes(),
		other.height(),
		cudaMemcpyHostToDevice,
		stream
	);
}

NppStatus ImageGpu::filterBox(const ImageGpu& output, NppiSize maskSize, NppStreamContext nppStreamCtx) {
	return nppiFilterBoxBorder_8u_C3R_Ctx(
		data(),
		step(),
		{ width(), height() },
		{ 0, 0 },
		output.data(),
		output.step(),
		{ width(), height() },
		maskSize,
		{ maskSize.width / 2, maskSize.height / 2 },
		NPP_BORDER_REPLICATE,
		nppStreamCtx
	);
}

cudaError_t ImageCpu::copyFromGpuAsync(const ImageGpu& other, cudaStream_t stream) {
	return cudaMemcpy2DAsync(
		data(),
		step(),
		other.data(),
		other.step(),
		other.widthBytes(),
		other.height(),
		cudaMemcpyDeviceToHost,
		stream
	);
}