#include "image_io.h"

// FreeImage has a permissive license:
// https://freeimage.sourceforge.io/freeimage-license.txt
#include "FreeImage.h"

IoError loadImage(const std::string& fileName, ImageCpu& image) {

	FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(fileName.c_str());
	if (eFormat == FIF_UNKNOWN) {
		eFormat = FreeImage_GetFIFFromFilename(fileName.c_str());
		if (eFormat == FIF_UNKNOWN) {
			return IoError::UnknownFileFormat;
		}
	}

	FIBITMAP* bitmap = nullptr;

	if (FreeImage_FIFSupportsReading(eFormat)) {
		bitmap = FreeImage_Load(eFormat, fileName.c_str());
	}

	if (bitmap == nullptr) {
		return IoError::LoadingFailed;
	}

	// The classes are hardcoded for 3-channel 24 BPP
	if (FreeImage_GetBPP(bitmap) != 24 || FreeImage_GetColorType(bitmap) != FIC_RGB) {
		return IoError::UnhandledFormat;
	}
	
	Npp8u* data = FreeImage_GetBits(bitmap);
	int width = FreeImage_GetWidth(bitmap);
	int height = FreeImage_GetHeight(bitmap);
	int pitch = FreeImage_GetPitch(bitmap);
	ImageCpu newImage(data, width, height, pitch);

	std::swap(image, newImage);

	FreeImage_Unload(bitmap);

	return IoError::Success;
}

IoError saveImage(const std::string& fileName, const ImageCpu& image) {
	FIBITMAP* bitmap = FreeImage_Allocate(image.width(), image.height(), 24);
	if (bitmap == nullptr) {
		return IoError::AllocationError;
	}

	memcpy(FreeImage_GetBits(bitmap), image.data(), image.step() * image.height());

	auto result = FreeImage_Save(FIF_PNG, bitmap, fileName.c_str(), 0);

	FreeImage_Unload(bitmap);

	if (result) {
		return IoError::Success;
	}
	else {
		return IoError::WritingFailed;
	}
}