#include "image_cpu.h"

#include "util.h"

#include "FreeImage.h"

#include <cstdlib>
#include <fstream>
#include <ios>
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

void ImageCpu::saveToDirectory(const std::filesystem::path& directory) const {
	std::filesystem::path fileName = directory;
	fileName /= fileName_;

	std::string tmp = fileName.string();
	const char* fileNameCstr = tmp.c_str();

	auto result = FreeImage_Save(FIF_PNG, bitmap_.get(), fileNameCstr, 0);

	if (!result) {
		throw IoError::WritingFailed;
	}
}

cudaError_t ImageCpu::copyBackFromStagingBuffer(const StagingBuffer& other, cudaStream_t stream) {
	return cudaMemcpyAsync(data(), other.data(), bytesData(), cudaMemcpyHostToHost);
}

// Manually implement check for width, height and BPP of a PNG image since FreeImage cannot do it.
size_t decodedImageSize(const std::filesystem::path& fileName) {
	constexpr size_t N = 26;
	Npp8u buffer[N];
	Npp8u pngHeader[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };

	std::ifstream file(fileName, std::ios::binary);
	try {
		file.read(reinterpret_cast<char*>(buffer), N);
		if (file.gcount() != N) {
			throw IoError::UnknownFileFormat;
		}
	}
	catch (std::ios_base::failure err) {
		throw IoError::UnknownFileFormat;
	}

	// Check for the presence of a PNG header
	if (memcmp(buffer, pngHeader, 8) != 0) {
		throw IoError::UnknownFileFormat;
	}

	auto colorType = buffer[25];

	// Only RGB and grayscale images are allowed
	if (colorType != 0 && colorType != 2) {
		throw IoError::UnhandledFormat;
	}

	auto bytesToInt = [](Npp8u* ptr) {
		Npp32u res = ptr[0];
		for (int i = 0; i <= 3; i++) {
			res = (res << 8) | ptr[i];
		}
		return res;
	};

	Npp32u width = bytesToInt(&buffer[16]);
	Npp32u height = bytesToInt(&buffer[20]);
	Npp32u bitDepth = buffer[24];
	Npp32u channels = colorType == 0 ? 1 : 3;
	Npp32u bpp = bitDepth * channels;

	return cudaStepSize(width, bpp / 8) * height;
}