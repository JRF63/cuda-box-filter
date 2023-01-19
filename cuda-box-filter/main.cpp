#include "info.h"
#include "errors.h"
#include "image.h"
#include "util.h"

#include <cstdio>

void printDummyImage(ImageCpu& image) {
	printf("data: %p\n", image.data());
	int width = 8;
	int height = 8;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < (width * 3); j++) {
			printf("%3d, ", (image.data())[i * width * 3 + j]);
		}
		printf("\n");
	}
	printf("\n");
}

ImageCpu createDummyImage() {
	int width = 8;
	int height = 8;
	ImageCpu image(width, height);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < (width * 3); j++) {
			unsigned char val = 0;
			if (i == 0 || i == (height - 1)) {
				val = 255;
			}
			if (j < 3 || j >= (width * 3 - 3)) {
				val = 255;
			}
			(image.data())[i * width * 3 + j] = val;
		}
	}
	printDummyImage(image);
	return image;
}

void doFilter(ImageCpu& image, NppiSize maskSize, NppStream& nppStream, TimingEvents& events) {
	// These can be pre-allocated outside this call if the images have the same resolution

	ImageGpu input(image.width(), image.height());
	ImageGpu output(image.width(), image.height());

	CUDA_CHECK(events.record(TimingEventKind::START, nppStream.stream()));

	CUDA_CHECK(input.copyFromCpuAsync(image, nppStream.stream()));

	CUDA_CHECK(events.record(TimingEventKind::WRITE_END, nppStream.stream()));

	NPP_CHECK(input.filterBox(output, maskSize, nppStream.context()));

	CUDA_CHECK(events.record(TimingEventKind::FILTERING_END, nppStream.stream()));

	CUDA_CHECK(image.copyFromGpuAsync(output, nppStream.stream()));

	CUDA_CHECK(events.record(TimingEventKind::READ_END, nppStream.stream()));
}

int main() {
	if (!printNPPinfo()) {
		return 1;
	}

	ImageCpu image = createDummyImage();
	NppiSize maskSize = { 5, 5 };
	NppStream nppStream = NppStream();
	TimingEvents events = TimingEvents();

	doFilter(image, maskSize, nppStream, events);

	float writeDuration;
	float filteringDuration;
	float readDuration;
	float latency;

	CUDA_CHECK(events.getTimingData(&writeDuration, &filteringDuration, &readDuration, &latency));
	printf("%f\n", writeDuration);
	printf("%f\n", filteringDuration);
	printf("%f\n", readDuration);
	printf("%f\n", latency);

	printDummyImage(image);

	return 0;
}

