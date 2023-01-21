#pragma once

#include "nppi.h"

#include <memory>
#include <cstdlib>
#include <string>

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

struct PinnedMemoryDestroyer {
	void operator()(void* ptr) { cudaFreeHost(ptr); }
};

class ImageCpu {
public:
	ImageCpu(int width, int height) : width_(width), height_(height) {
		// Cast to `long long` first to suppress compiler warnings
		auto size = static_cast<long long>(width) * height * 3;
		void* ptr = nullptr;
		cudaMallocHost(&ptr, size);
		data_ = std::unique_ptr<Npp8u, PinnedMemoryDestroyer>(static_cast<Npp8u*>(ptr));
		step_ = width * sizeof(Npp8u) * 3;
	}

	ImageCpu(Npp8u* data, int width, int height, int step)
		: width_(width), height_(height), step_(step)
	{
		auto size = static_cast<long long>(height) * step;
		void* ptr = nullptr;
		cudaMallocHost(&ptr, size);
		data_ = std::unique_ptr<Npp8u, PinnedMemoryDestroyer>(static_cast<Npp8u*>(ptr));
		// CUDA code samples reverses the order of the rows but that's not need for box filtering
		memcpy(data_.get(), data, size);
	}

	ImageCpu() : width_(0), height_(0), step_(0) {
		data_ = nullptr;
	}

	void setFilename(std::string&& fileName) {
		fileName_ = std::move(fileName);
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

	const std::string& fileName() const {
		return fileName_;
	}

	cudaError_t copyFromGpuAsync(const ImageGpu& other, cudaStream_t stream);

private:
	// Usage of `std::unique_ptr` implicitly disables the copy constructor
	std::unique_ptr<Npp8u, PinnedMemoryDestroyer> data_;
	std::string fileName_;
	int width_;
	int height_;
	int step_;
};
