#include "image_cpu.h"

#include "FreeImage.h"

#include <string>

void FreeImageDestroyer::operator()(FIBITMAP* ptr) {
	FreeImage_Unload(ptr);
}

ImageCpu::ImageCpu(const std::filesystem::path& fileName) {
	// Create an L-value std::string to ensure that the pointer from c_str does not get invalidated
	std::string tmp = fileName.string();
	const char* fileNameCstr = tmp.c_str();

	FREE_IMAGE_FORMAT format = FreeImage_GetFileType(fileNameCstr);
	if (format == FIF_UNKNOWN) {
		format = FreeImage_GetFIFFromFilename(fileNameCstr);
		if (format == FIF_UNKNOWN) {
			throw IoError::UnknownFileFormat;
		}
	}

	FIBITMAP* bitmap = nullptr;

	if (FreeImage_FIFSupportsReading(format)) {
		bitmap = FreeImage_Load(format, fileNameCstr);
	}

	if (bitmap == nullptr) {
		throw IoError::LoadingFailed;
	}

	data_ = FreeImage_GetBits(bitmap);
	width_ = FreeImage_GetWidth(bitmap);
	height_ = FreeImage_GetHeight(bitmap);
	step_ = FreeImage_GetPitch(bitmap);
	bytesPerPixel_ = FreeImage_GetBPP(bitmap) / 8;
	fileName_ = std::move(fileName.filename().string());
	bitmap_ = std::unique_ptr<FIBITMAP, FreeImageDestroyer>(bitmap);
}

void ImageCpu::saveToDirectory(const std::filesystem::path& directory) {
	std::filesystem::path fileName = directory;
	fileName /= fileName_;

	std::string tmp = fileName.string();
	const char* fileNameCstr = tmp.c_str();

	auto result = FreeImage_Save(FIF_PNG, bitmap_.get(), fileNameCstr, 0);

	if (!result) {
		throw IoError::WritingFailed;
	}
}

ImageCpu::ImageCpu(const std::filesystem::path& fileName, int width, int height, int bytesPerPixel) {
	FIBITMAP* bitmap = FreeImage_Allocate(width, height, 8 * bytesPerPixel);
	if (bitmap == nullptr) {
		throw IoError::AllocationError;
	}
	data_ = FreeImage_GetBits(bitmap);
	width_ = width;
	height_ = height;
	step_ = FreeImage_GetPitch(bitmap);
	bytesPerPixel_ = bytesPerPixel;
	fileName_ = std::move(fileName.filename().string());
	bitmap_ = std::unique_ptr<FIBITMAP, FreeImageDestroyer>(bitmap);
}

cudaError_t ImageCpu::copyBackFromStagingBuffer(const StagingBuffer& other, cudaStream_t stream) {
	return cudaMemcpyAsync(data(), other.data(), bytesData(), cudaMemcpyHostToHost);
}