#pragma once

#include "nppi.h"

#include <memory>
#include <cstdlib>

class ImageCpu;

struct NppiDeleter {
	void operator()(Npp8u* ptr) { nppiFree(ptr); }
};

class ImageGpu
{
public:
	ImageGpu(int width, int height): width_(width), height_(height) {
		data_ = std::unique_ptr<Npp8u, NppiDeleter>(nppiMalloc_8u_C3(width, height, &step_));
	}

	Npp8u* data() const {
		return data_.get();
	}

	int width() const {
		return width_;
	}

	int height() const {
		return height_;
	}

	int step() const {
		return step_;
	}

	size_t widthBytes() const {
		return width() * sizeof(Npp8u) * 3;
	}

	// Applies the box filter and writes the result to another `ImageGpu`.
	NppStatus filterBox(const ImageGpu& output, NppiSize maskSize, NppStreamContext nppStreamCtx);

	cudaError_t copyFromCpuAsync(const ImageCpu& other, cudaStream_t stream);

private:
	std::unique_ptr<Npp8u, NppiDeleter> data_;
	int width_;
	int height_;
	int step_;
};

class ImageCpu {
public:
	ImageCpu(int width, int height) : width_(width), height_(height) {
		// Cast to `long long` first to suppress compiler warnings
		auto size = static_cast<long long>(width) * height * 3;
		data_ = std::make_unique<Npp8u[]>(size);
		step_ = width * sizeof(Npp8u) * 3;
	}

	ImageCpu(Npp8u* data, int width, int height, int step)
		: width_(width), height_(height), step_(step)
	{
		auto size = static_cast<long long>(height) * step;
		data_ = std::make_unique<Npp8u[]>(size);
		// CUDA code samples reverses the order of the rows but that's not need for box filtering
		memcpy(data_.get(), data, size);
	}

	ImageCpu() : width_(0), height_(0), step_(0) {
		data_ = nullptr;
	}

	Npp8u* data() const {
		return data_.get();
	}

	int width() const {
		return width_;
	}

	int height() const {
		return height_;
	}

	int step() const {
		return step_;
	}

	size_t widthBytes() const {
		return width() * sizeof(Npp8u) * 3;
	}

	cudaError_t copyFromGpuAsync(const ImageGpu& other, cudaStream_t stream);

private:
	// Usage of `std::unique_ptr` implicitly disables the copy constructor
	std::unique_ptr<Npp8u[]> data_;
	int width_;
	int height_;
	int step_;
};
