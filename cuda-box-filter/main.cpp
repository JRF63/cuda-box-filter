#include "info.h"
#include "errors.h"
#include "image.h"
#include "util.h"
#include "ring_buffer.h"

#include <cstdio>
#include <algorithm>
#include <memory>
#include <chrono>

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
	// printDummyImage(image);
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

struct Batch {
	NppStream nppStream;
	TimingEvents events;

	Batch(NppStream&& a, TimingEvents&& b) : nppStream(std::move(a)), events(std::move(b)) {}
};

int main() {
	if (!printNPPinfo()) {
		return 1;
	}

	int N = 4;
	int NUM_BATCHES = 10;
	NppiSize maskSize = { 5, 5 };

	std::shared_ptr<std::atomic_bool> stop = std::make_shared<std::atomic_bool>(false);
	std::shared_ptr<AtomicRingBuffer<ImageCpu>> images = std::make_shared<AtomicRingBuffer<ImageCpu>>(N);

	// Preload images
	for (int i = 0; i < (1 << N); i++) {
		images->push(createDummyImage());
	}

	std::thread timer([=]() {
		using namespace std::chrono_literals;
		std::this_thread::sleep_for(10s);
		stop->store(true, std::memory_order_release);
	});

	std::thread imageProducer([=]() {
		while (!stop->load(std::memory_order_acquire)) {
			images->push(createDummyImage());
		}
	});

	std::vector<Batch> batches;

	for (int i = 0; i < NUM_BATCHES; i++) {
		printf("%d\n", i);
		batches.push_back(Batch{ NppStream(), TimingEvents() });
	}

	int i = 0;
	while (!stop->load(std::memory_order_acquire)) {
		if (i == NUM_BATCHES) {
			i = 0;
		}

		float writeDuration;
		float filteringDuration;
		float readDuration;
		float latency;

		ImageCpu image = images->pop();
		doFilter(image, maskSize, batches[i].nppStream, batches[i].events);

		CUDA_CHECK(batches[i].events.getTimingData(&writeDuration, &filteringDuration, &readDuration, &latency));
		printf("%5.5f %5.5f %5.5f %5.5f\n", writeDuration, filteringDuration, readDuration, latency);

		i++;
	}

	timer.join();
	imageProducer.join();

	return 0;
}

