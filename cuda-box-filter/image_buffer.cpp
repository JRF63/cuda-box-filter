#include "image_buffer.h"
#include "util.h"

void PinnedMemoryDestroyer::operator()(void* ptr) { cudaFreeHost(ptr); }

cudaError_t StagingBuffer::copyFromCpu(const ImageCpu& other, cudaStream_t stream) {
	// Include the padding because this memcpy's the whole buffer
	if (allocatedSize_ < other.bytesDataWithPadding()) {
		fprintf(stderr, "Unrecoverable error: allocated buffer is too small for input image\n");
		exit(1);
	}
	width_ = other.width();
	height_ = other.height();
	step_ = other.step();
	bytesPerPixel_ = other.bytesPerPixel();
	return cudaMemcpyAsync(data(), other.data(), other.bytesDataWithPadding(), cudaMemcpyHostToHost);
}

cudaError_t StagingBuffer::copyBackFromGpuAsync(const GpuBuffer& other, cudaStream_t stream) {
	if (allocatedSize_ < other.bytesData()) {
		fprintf(stderr, "Unrecoverable error: allocated buffer is too small for input image\n");
		exit(2);
	}
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

void DeviceMemoryDeleter::operator()(void* ptr) { cudaFree(ptr); }

cudaError_t GpuBuffer::copyFromStagingBufferAsync(const StagingBuffer& other, cudaStream_t stream) {
	width_ = other.width();
	height_ = other.height();
	step_ = cudaStepSize(other.width(), other.bytesPerPixel());
	bytesPerPixel_ = other.bytesPerPixel();
	if (allocatedSize_ < bytesData()) {
		fprintf(stderr, "Unrecoverable error: allocated buffer is too small for input image\n");
		exit(3);
	}
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

NppStatus GpuBuffer::filterBox(GpuBuffer& output, NppiSize maskSize, NppStreamContext nppStreamCtx) {
	output.width_ = width();
	output.height_ = height();
	output.step_ = step();
	output.bytesPerPixel_ = bytesPerPixel();
	if (bytesPerPixel() == 1) {
		return nppiFilterBoxBorder_8u_C1R_Ctx(
			static_cast<Npp8u*>(data()),
			step(),
			{ width(), height() },
			{ 0, 0 },
			static_cast<Npp8u*>(output.data()),
			step(),
			{ width(), height() },
			maskSize,
			{ maskSize.width / 2, maskSize.height / 2 },
			NPP_BORDER_REPLICATE,
			nppStreamCtx
		);
	}
	else if (bytesPerPixel() == 3) {
		return nppiFilterBoxBorder_8u_C3R_Ctx(
			static_cast<Npp8u*>(data()),
			step(),
			{ width(), height() },
			{ 0, 0 },
			static_cast<Npp8u*>(output.data()),
			step(),
			{ width(), height() },
			maskSize,
			{ maskSize.width / 2, maskSize.height / 2 },
			NPP_BORDER_REPLICATE,
			nppStreamCtx
		);
	}
	else {
		return NppStatus::NPP_SUCCESS;
	}
	
}