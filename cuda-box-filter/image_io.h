#pragma once

#include "image.h"

#include <string>

enum class IoError {
	Success,
	LoadingFailed,
	UnknownFileFormat,
	UnhandledFormat,
	AllocationError,
	WritingFailed,
};

IoError loadImage(const std::string& fileName, ImageCpu& image);

IoError saveImage(const std::string& fileName, const ImageCpu& image);