#include "info.h"
#include "errors.h"
#include "image.h"
#include "util.h"
#include "ring_buffer.h"
#include "image_io.h"
#include "signal.h"

#include <atomic>
#include <cstdio>
#include <chrono>
#include <deque>
#include <filesystem> // Need C++17 for this
#include <fstream>
#include <iostream>
#include <vector>

void doFilter(ImageCpu& image, NppiSize maskSize, NppStream& nppStream, TimingEvents& events) {
	// These can be pre-allocated outside this call if the images have the same resolution
	ImageGpu input(image.width(), image.height());
	ImageGpu output(image.width(), image.height());

	CUDA_CHECK(events.record(TimingEventKind::Start, nppStream.stream()));

	// Copy from CPU to GPU
	CUDA_CHECK(input.copyFromCpuAsync(image, nppStream.stream()));

	CUDA_CHECK(events.record(TimingEventKind::WriteEnd, nppStream.stream()));

	// Do the filter
	NPP_CHECK(input.filterBox(output, maskSize, nppStream.context()));

	CUDA_CHECK(events.record(TimingEventKind::FilteringEnd, nppStream.stream()));

	// Copy from GPU to CPU and write back to the original image
	CUDA_CHECK(image.copyFromGpuAsync(output, nppStream.stream()));

	CUDA_CHECK(events.record(TimingEventKind::ReadEnd, nppStream.stream()));
}

struct Batch {
	NppStream nppStream;
	TimingEvents events;
	ImageCpu output;
	std::atomic_bool hasData;

	Batch(): nppStream(NppStream()), events(TimingEvents()), output(ImageCpu()), hasData(false) {}

	// Move only
	Batch(const Batch&) = delete;
	Batch& operator=(const Batch&) = delete;
	Batch(Batch&&) noexcept {}
	Batch& operator=(Batch&&) noexcept {}

	void setDataAvailable(bool val) {
		hasData.store(val, std::memory_order_release);
	}

	bool isDataAvailable() {
		return hasData.load(std::memory_order_acquire);
	}

	void setOutput(ImageCpu&& image) {
		output = std::move(image);
	}

	ImageCpu getOutput() {
		ImageCpu dummy;
		std::swap(dummy, output);
		return dummy;
	}
};

// Computes the logarithm base-2 of a number
int log2(int n) {
	int ans = 0;
	while (n >>= 1) {
		ans++;
	}
	return ans;
}

int main() {
	if (!printNPPinfo()) {
		return 1;
	}

	// Directory that contains the images to be processed
	const std::string& imagesDir = "images";

	// Directory where the outputs will be written
	const std::string& outputDir = "outputs";

	// Hardcoded to 10 seconds, trivial to modify.
	auto processingDuration = std::chrono::milliseconds(10000);

	// Upper bound on the amount of images loaded to main RAM.
	// Hardcoded. This is ideally derived from the amount of system RAM.
	int numImagesBuffered = 64;

	// Max amount of images that is submitted to the GPU.
	// Also hardcoded. It's best to compute this based on VRAM since the amount of CUDA stream is not
	// usually the limit.
	int batchSize = 64;

	// Size of the box filter in pixels x pixels
	NppiSize maskSize = { 8, 8 };

	// This is needed because the queue needs to be a power of two
	int log2NumImages = log2(numImagesBuffered);

	// Single producer, single consumer queue of images
	AtomicRingBuffer<ImageCpu> images(log2NumImages);

	Signal stop;
	Signal wait;

	// Thread that loads the images to CPU memory
	std::thread imageProducer([&]() {
		int i = 0;
		int limit = (1 << log2NumImages); // Pause other threads before this num of images is loaded

		printf("Loading images to RAM");
		for (;;) {
			for (const auto& entry : std::filesystem::directory_iterator{ "images" }) {
				if (stop.shouldStop()) {
					// Exit out of both loops
					return;
				}

				printf(".");
				if (i == limit) {
					// Signal main thread to stop waiting
					wait.signalStop();
					printf("\nProcessing images");
				}

				ImageCpu image;
				auto result = loadImage(entry, image);

				switch (result) {
				case IoError::Success:
					break;
				case IoError::UnknownFileFormat:
					// .gitignore is in the images folder
					continue;
				case IoError::LoadingFailed:
				case IoError::UnhandledFormat:
				case IoError::AllocationError:
				case IoError::WritingFailed:
					fprintf(stderr, "I/O error while loading image");
					exit(1);
				}

				images.push(std::move(image));
				i++;
			}
		}
	});

	// Busy loop to wait for sufficient images to be pre-loaded
	while (!wait.shouldStop()) {}

	// Thread that signals the processing to stop
	std::thread timer([&]() {
		std::this_thread::sleep_for(processingDuration);
		stop.signalStop();
	});

	std::vector<Batch> batches;
	for (int i = 0; i < batchSize; i++) {
		batches.emplace_back();
	}

	// Thread that processes the filtered images
	std::thread outputHandler([&]() {
		std::vector<TimingData> timings;
		std::deque<ImageCpu> processedImages;

		int i = 0;
		while (!stop.shouldStop()) {
			if (batches[i].isDataAvailable()) {
				TimingData timingData;
				// getTimingData blocks this thread until the output is read back to system RAM
				CUDA_CHECK(batches[i].events.getTimingData(timingData));

				timings.push_back(timingData);
				
				ImageCpu output = batches[i].getOutput();
				batches[i].setDataAvailable(false);

				// Cache the last 10 images
				processedImages.push_back(std::move(output));
				if (processedImages.size() > 10) {
					processedImages.pop_front();
				}

				i++;
				if (i == batchSize) {
					i = 0;
				}
			}
		}
		printf("\n");
		printf("Processed %lld images", timings.size());

		// Save the last 10 images
		for (const auto& image : processedImages) {
			if (!image.fileName().empty()) {
				const auto fileName = outputDir + "/" + image.fileName();
				//saveImage(fileName, image);
			}
		}

		// Write timing data
		std::ofstream out(outputDir + "/timings.csv");
		out << "write_time,processing_time,read_time,latency" << std::endl;
		for (const auto& timing : timings) {
			out << timing.writeDuration << ",";
			out << timing.filteringDuration << ",";
			out << timing.readDuration << ",";
			out << timing.latency << ",";
			out << std::endl;
		}
	});

	// GPU processing loop
	int i = 0;
	while (!stop.shouldStop()) {
		if (!batches[i].isDataAvailable()) {
			ImageCpu image = images.pop();
			doFilter(image, maskSize, batches[i].nppStream, batches[i].events);

			batches[i].setOutput(std::move(image));
			batches[i].setDataAvailable(true);

			i++;
			if (i == batchSize) {
				i = 0;
			}
		}
	}

	timer.join();
	imageProducer.join();
	outputHandler.join();

	return 0;
}

