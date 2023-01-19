#pragma once

#include "nppi.h"

#include <memory>

class ImageCpu;

class ImageGpu
{
public:
	ImageGpu(int width, int height): width_(width), height_(height) {
		data_ = nppiMalloc_8u_C3(width, height, &step_);
	}

	~ImageGpu() {
		nppiFree(data_);
	}

	// Move only; no copying
	ImageGpu(const ImageGpu&) = delete;
	ImageGpu& operator= (const ImageGpu&) = delete;
	ImageGpu(ImageGpu&& other) = default;
	ImageGpu& operator=(ImageGpu&&) = default;

	Npp8u* data() const {
		return data_;
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
	NppStatus filterBox(const ImageGpu& output, NppiSize maskSize, NppStreamContext nppStreamCtx) {
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

	// Forward declare because of usage of `ImageCpu`
	cudaError_t copyFromCpuAsync(const ImageCpu& other, cudaStream_t stream);

private:
	Npp8u* data_;
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

	cudaError_t copyFromGpuAsync(const ImageGpu& other, cudaStream_t stream) {
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

private:
	// Usage of `std::unique_ptr` implicitly disables the copy constructor
	std::unique_ptr<Npp8u[]> data_;
	int width_;
	int height_;
	int step_;
};

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