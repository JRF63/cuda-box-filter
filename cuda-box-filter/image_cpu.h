#pragma once

#include "image_base.h"
#include "image_buffer.h"

#include "cuda_runtime.h"

#include <memory>
#include <cstdlib>
#include <filesystem>

struct FIBITMAP;
class StagingBuffer;

enum class IoError {
	LoadingFailed,
	UnknownFileFormat,
	UnhandledFormat,
	AllocationError,
	WritingFailed,
};

struct FreeImageDestroyer {
	void operator()(FIBITMAP* ptr);
};

class ImageCpu: public ImageBase {
public:
	ImageCpu(const std::filesystem::path& fileName);

	ImageCpu(const std::filesystem::path& fileName, int width, int height, int bytesPerPixel);

	void* data() const override {
		return data_;
	}

	const std::string& fileName() const {
		return fileName_;
	}

	cudaError_t copyBackFromStagingBuffer(const StagingBuffer& other, cudaStream_t stream);

	void saveToDirectory(const std::filesystem::path& directory);

private:
	std::unique_ptr<FIBITMAP, FreeImageDestroyer> bitmap_;
	void* data_;
	std::string fileName_;
};
