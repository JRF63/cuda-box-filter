#pragma once

#include "image_base.h"
#include "image_cpu.h"

#include "nppi.h"

#include <memory>
#include <string>

class ImageCpu;
class GpuBuffer;

struct PinnedMemoryDestroyer {
	void operator()(void* ptr);
};

class StagingBuffer: public ImageBase {
public:
	StagingBuffer(size_t size)
		: ImageBase(), allocatedSize_(size)
	{
		void* ptr = nullptr;
		cudaMallocHost(&ptr, size);
		data_ = std::unique_ptr<void, PinnedMemoryDestroyer>(ptr);
	}

	void* data() const override {
		return data_.get();
	}

	cudaError_t copyFromCpu(const ImageCpu& other, cudaStream_t stream);

	cudaError_t copyBackFromGpuAsync(const GpuBuffer& other, cudaStream_t stream);

private:
	std::unique_ptr<void, PinnedMemoryDestroyer> data_;
	size_t allocatedSize_;
};

struct DeviceMemoryDeleter {
	void operator()(void* ptr);
};

class GpuBuffer: public ImageBase
{
public:
	GpuBuffer(size_t size)
		: ImageBase(), allocatedSize_(size)
	{
		void* ptr = nullptr;
		cudaMalloc(&ptr, size);
		data_ = std::unique_ptr<void, DeviceMemoryDeleter>(ptr);
	}

	void* data() const override {
		return data_.get();
	}

	// Applies the box filter and writes the result to another `GpuBuffer`.
	NppStatus filterBox(GpuBuffer& output, NppiSize maskSize, NppStreamContext nppStreamCtx);

	cudaError_t copyFromStagingBufferAsync(const StagingBuffer& other, cudaStream_t stream);

private:
	std::unique_ptr<void, DeviceMemoryDeleter> data_;
	size_t allocatedSize_;
};
