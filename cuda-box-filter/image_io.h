#pragma once

#include "image.h"

#include <filesystem>

enum class IoError {
	Success,
	LoadingFailed,
	UnknownFileFormat,
	UnhandledFormat,
	AllocationError,
	WritingFailed,
};

IoError loadImage(const std::filesystem::path& fileName, ImageCpu& image);

IoError saveImage(const std::filesystem::path& fileName, const ImageCpu& image);