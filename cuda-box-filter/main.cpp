#include "info.h"
#include "errors.h"
#include "image_cpu.h"
#include "image_buffer.h"
#include "cuda_helper.h"
#include "util.h"
#include "sync.h"

#include <algorithm>
#include <cstdio>
#include <chrono>
#include <deque>
#include <filesystem> // Need C++17 for this
#include <fstream>
#include <mutex>
#include <iostream>
#include <vector>

int main() {
	// Hardcoded to 10 seconds, trivial to modify.
	auto processingDuration = std::chrono::milliseconds(10000);

	// Directory that contains the images to be processed
	const std::string& imagesDir = "images";

	// Directory where the outputs will be written
	const std::string& outputDir = "outputs";

	// Max number of images that will be saved
	constexpr size_t NUM_IMAGES_TO_OUTPUT = 10;

	// Reserve a certain amount of GPU memory to avoid starving the OS
	constexpr size_t GPU_MEM_RESERVED = 250000000;

	Signal stopSignal;
	Signal waitSignal;

	std::vector<TimingData> timings;
	std::mutex timingsMutex;
	std::deque<ImageCpu> outputImages;
	std::mutex outputImagesMutex;

	std::vector<std::thread> threads;

	// Thread that signals the processing to stop
	std::thread timer([&]() {
		std::this_thread::sleep_for(processingDuration);
		stopSignal.signalStop();
		});

	// Check outside the loop to avoid race conditions
	size_t freeMem;
	size_t totalMem;
	CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
	freeMem -= GPU_MEM_RESERVED;

	printf("Loading images to main memory");
	for (auto entry : std::filesystem::directory_iterator{ imagesDir }) {
		size_t memGpu;
		try {
			memGpu = decodedImageSize(entry);
		}
		catch (IoError err) {
			continue;
		}

		// There are 2 GPU buffers of the same size, one for the input and one for the output
		size_t requiredMemGpu = 2 * memGpu;

		if (requiredMemGpu <= freeMem) {
			// Manually subtract from the free memory estimate instead of quering again via `cudaMemGetInfo`
			freeMem -= requiredMemGpu;

			std::thread t([memGpu, entry, NUM_IMAGES_TO_OUTPUT, &waitSignal, &timings, &timingsMutex, &outputImages, &outputImagesMutex]() {
				GpuBuffer gpuInput(memGpu);
				GpuBuffer gpuOutput(memGpu);

				NppStream nppStream;
				TimingEvents events;

				try {
					ImageCpu image(entry);
					printf(".");

					size_t memCpu = freeImageStepSize(image.width(), 3) * image.height();
					StagingBuffer stagingBuffer(memCpu);

					int blurAmount = ceilDiv(std::min(image.width(), image.height()), 200);
					NppiSize maskSize{ blurAmount, blurAmount };

					CUDA_CHECK(stagingBuffer.copyFromCpu(image, nppStream.stream()));
					nppStream.synchronize();

					// Wait until all threads have loaded the image to the staging buffer
					while (!waitSignal.shouldStop()) {
						std::this_thread::yield();
					}

					CUDA_CHECK(events.record(TimingEventKind::Start, nppStream.stream()));
					CUDA_CHECK(gpuInput.copyFromStagingBufferAsync(stagingBuffer, nppStream.stream()));
					CUDA_CHECK(events.record(TimingEventKind::WriteEnd, nppStream.stream()));
					NPP_CHECK(gpuInput.filterBox(gpuOutput, maskSize, nppStream.context()));
					CUDA_CHECK(events.record(TimingEventKind::FilteringEnd, nppStream.stream()));
					CUDA_CHECK(stagingBuffer.copyBackFromGpuAsync(gpuOutput, nppStream.stream()));
					CUDA_CHECK(events.record(TimingEventKind::ReadEnd, nppStream.stream()));

					TimingData timingData;
					CUDA_CHECK(events.getTimingData(timingData));

					CUDA_CHECK(image.copyBackFromStagingBuffer(stagingBuffer, nppStream.stream()));
					nppStream.synchronize();

					// Combine the timing data
					{
						std::lock_guard<std::mutex> guard(timingsMutex);
						timings.push_back(timingData);
					}

					{
						std::lock_guard<std::mutex> guard(outputImagesMutex);
						outputImages.push_back(std::move(image));
						if (outputImages.size() > NUM_IMAGES_TO_OUTPUT) {
							outputImages.pop_front();
						}
					}
				}
				catch (IoError err) {}

				});
			threads.push_back(std::move(t));
		}
		else {
			waitSignal.signalStop();
			for (auto&& t : threads) {
				t.join();
			}
			threads.clear();
			waitSignal.reset();

			cudaDeviceSynchronize();

			// Exit the loop upon timeout
			if (stopSignal.shouldStop()) {
				break;
			}

			CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
			freeMem -= GPU_MEM_RESERVED;

			printf("\nLoading images to main memory");
		}
	}

	timer.join();
	for (auto& t : threads) {
		t.join();
	}

	// Write timing data
	printf("\nWriting performance data");
	std::ofstream out(outputDir + "/timings.csv");
	out << "write_time,processing_time,read_time,latency" << std::endl;
	for (const auto& timing : timings) {
		out << timing.writeDuration << ",";
		out << timing.filteringDuration << ",";
		out << timing.readDuration << ",";
		out << timing.latency << ",";
		out << std::endl;
		printf(".");
	}

	printf("\nSaving images");
	while (!outputImages.empty()) {
		auto image = std::move(outputImages.front());
		outputImages.pop_front();
		std::thread t([image = std::move(image), &outputDir]() {
			image.saveToDirectory(outputDir);
			printf(".");
		});
		threads.push_back(std::move(t));
	}
	for (auto& t : threads) {
		t.join();
	}
	printf("\n");

	return 0;
}

