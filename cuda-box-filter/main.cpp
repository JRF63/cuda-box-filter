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

	// Dataset properties. The max size of the image is the limiting factor for concurrency.
	constexpr size_t IMAGE_MAX_WIDTH_RGB = 10240;
	constexpr size_t IMAGE_MAX_HEIGHT_RGB = 7665;
	constexpr size_t IMAGE_CUDA_MAX_SIZE = cudaStepSize(IMAGE_MAX_WIDTH_RGB, 3) * IMAGE_MAX_HEIGHT_RGB;
	constexpr size_t IMAGE_CPU_MAX_SIZE = freeImageStepSize(IMAGE_MAX_WIDTH_RGB, 3) * IMAGE_MAX_HEIGHT_RGB;

	size_t freeMem;
	size_t totalMem;
	CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));

	// There are 2 GPU buffers of the same size, one for the input and one for the output
	size_t batchSize = freeMem / (2 * IMAGE_CUDA_MAX_SIZE);

	Signal stopSignal;
	Barrier barrier(batchSize);

	std::mutex fileNamesMutex;
	std::deque<std::filesystem::path> fileNames;
	std::mutex timingsMutex;
	std::vector<TimingData> timings;
	std::mutex outputImagesMutex;
	std::deque<ImageCpu> outputImages;
	std::vector<std::thread> threads;

	for (auto entry : std::filesystem::directory_iterator{ imagesDir }) {
		fileNames.push_back(entry);
	}

	// Thread that signals the processing to stop
	std::thread timer([&]() {
		std::this_thread::sleep_for(processingDuration);
		stopSignal.signalStop();
		});

	for (size_t i = 0; i < batchSize; i++) {
		std::thread t([&]() {
			StagingBuffer stagingBuffer(IMAGE_CPU_MAX_SIZE);
			GpuBuffer gpuInput(IMAGE_CUDA_MAX_SIZE);
			GpuBuffer gpuOutput(IMAGE_CUDA_MAX_SIZE);

			NppStream nppStream;
			TimingEvents events;
			TimingData timingData;
			std::string currentFilename;

			std::vector<TimingData> streamTimings;

			barrier.wait();

			while (!stopSignal.shouldStop()) {
				if (fileNamesMutex.try_lock()) {
					if (fileNames.empty()) {
						fileNamesMutex.unlock();
						break;
					}
					auto entry = fileNames.front();
					fileNames.pop_front();
					fileNamesMutex.unlock();

					try {
						ImageCpu image(entry);

						int blurAmount = ceilDiv(std::min(image.width(), image.height()), 200);
						NppiSize maskSize{ blurAmount, blurAmount };

						CUDA_CHECK(stagingBuffer.copyFromCpu(image, nppStream.stream()));
						CUDA_CHECK(events.record(TimingEventKind::Start, nppStream.stream()));
						CUDA_CHECK(gpuInput.copyFromStagingBufferAsync(stagingBuffer, nppStream.stream()));
						CUDA_CHECK(events.record(TimingEventKind::WriteEnd, nppStream.stream()));
						NPP_CHECK(gpuInput.filterBox(gpuOutput, maskSize, nppStream.context()));
						CUDA_CHECK(events.record(TimingEventKind::FilteringEnd, nppStream.stream()));
						CUDA_CHECK(stagingBuffer.copyBackFromGpuAsync(gpuOutput, nppStream.stream()));
						CUDA_CHECK(events.record(TimingEventKind::ReadEnd, nppStream.stream()));

						currentFilename = image.fileName();

						CUDA_CHECK(events.getTimingData(timingData));
						streamTimings.push_back(timingData);

						//CUDA_CHECK(image.copyBackFromStagingBuffer(stagingBuffer, nppStream.stream()));
						//nppStream.synchronize();
						//image.saveToDirectory("outputs");
					}
					catch (IoError err) {
						switch (err) {
						case IoError::UnknownFileFormat:
						case IoError::LoadingFailed:
						case IoError::UnhandledFormat:
						case IoError::AllocationError:
						case IoError::WritingFailed:
							// TODO: Logging
							continue;
						}
					}
				}
			}

			// Combine the timing data
			timingsMutex.lock();
			for (auto timing : streamTimings) {
				timings.push_back(timing);
			}
			timingsMutex.unlock();

			// Save the last images processed
			try {
				if (currentFilename.empty()) {
					// No images were processed
					return;
				}
				ImageCpu image(currentFilename, stagingBuffer.width(), stagingBuffer.height(), stagingBuffer.bytesPerPixel());
				CUDA_CHECK(image.copyBackFromStagingBuffer(stagingBuffer, nppStream.stream()));
				nppStream.synchronize();
				
				outputImagesMutex.lock();
				outputImages.push_back(std::move(image));
				if (outputImages.size() > NUM_IMAGES_TO_OUTPUT) {
					outputImages.pop_front();
				}
				outputImagesMutex.unlock();
			}
			catch (IoError err) {
				switch (err) {
				case IoError::UnknownFileFormat:
				case IoError::LoadingFailed:
				case IoError::UnhandledFormat:
				case IoError::AllocationError:
				case IoError::WritingFailed:
					// TODO: Logging
					break;
				}
			}
			});

		threads.push_back(std::move(t));
	}

	timer.join();
	for (auto&& t : threads) {
		t.join();
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

	for (auto&& image : outputImages) {
		image.saveToDirectory(outputDir);
	}

	return 0;
}

