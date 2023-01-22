#pragma once

constexpr size_t CUDA_ALIGNMENT = 512;
constexpr size_t FREE_IMAGE_ALIGNMENT = 4;

size_t constexpr ceilDiv(size_t num, size_t den) {
	return (num + (den - 1)) / den;
}

size_t constexpr cudaStepSize(size_t width, size_t bytesPerPixel) {
	return CUDA_ALIGNMENT * ceilDiv(width * bytesPerPixel, CUDA_ALIGNMENT);
}

size_t constexpr freeImageStepSize(size_t width, size_t bytesPerPixel) {
	return FREE_IMAGE_ALIGNMENT * ceilDiv(width * bytesPerPixel, FREE_IMAGE_ALIGNMENT);
}