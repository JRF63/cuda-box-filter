#pragma once

class ImageBase
{
public:
	ImageBase() : width_(0), height_(0), step_(0), bytesPerPixel_(0) {}

	// Pointer to the start of the image buffer.
	virtual void* data() const = 0;

	// Width of the image.
	int width() const {
		return width_;
	}

	// Height of the image.
	int height() const {
		return height_;
	}

	// This is the difference between the addresses of the start of two consecutive rows. Libraries
	// add padding to align the rows to optimal memory boundaries.
	int step() const {
		return step_;
	}

	// Bytes per pixel. Either 1 or 3.
	int bytesPerPixel() const {
		return bytesPerPixel_;
	}

	// Size of a row excluding padding.
	size_t widthBytes() const {
		return static_cast<size_t>(width()) * bytesPerPixel();
	}

	// Total number of bytes held including padding.
	size_t bytesData() const {
		return static_cast<size_t>(step()) * height();
	}

protected:
	int width_;
	int height_;
	int step_;
	// We're dealing with either 24 BPP RGB or 8 BPP grayscale images. Instead of making this a member
	// variable, BPP can alternatively be abstracted to a template parameter.
	int bytesPerPixel_;
};

