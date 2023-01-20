#pragma once

#include "errors.h"

#include "cuda_runtime.h"
#include "nppi.h"

#include <memory>

enum class TimingEventKind {
	Start,
	WriteEnd,
	FilteringEnd,
	ReadEnd,
};

struct TimingData {
	float writeDuration;
	float filteringDuration;
	float readDuration;
	float latency;
};

struct CudaEventDestroyer {
	void operator()(cudaEvent_t ptr) { cudaEventDestroy(ptr); }
};

class TimingEvents {
public:
	TimingEvents() {
		cudaEvent_t start;
		CUDA_CHECK(cudaEventCreate(&start));
		start_ = std::unique_ptr<CUevent_st, CudaEventDestroyer>(start);

		cudaEvent_t writeEnd;
		CUDA_CHECK(cudaEventCreate(&writeEnd));
		writeEnd_ = std::unique_ptr<CUevent_st, CudaEventDestroyer>(writeEnd);

		cudaEvent_t filteringEnd;
		CUDA_CHECK(cudaEventCreate(&filteringEnd));
		filteringEnd_ = std::unique_ptr<CUevent_st, CudaEventDestroyer>(filteringEnd);

		cudaEvent_t readEnd;
		CUDA_CHECK(cudaEventCreate(&readEnd));
		readEnd_ = std::unique_ptr<CUevent_st, CudaEventDestroyer>(readEnd);
	}

	// Records the event to the stream asynchronously.
	cudaError_t record(TimingEventKind kind, cudaStream_t stream) {
		switch (kind) {
		case TimingEventKind::Start:
			return cudaEventRecord(start_.get(), stream);
		case TimingEventKind::WriteEnd:
			return cudaEventRecord(writeEnd_.get(), stream);
		case TimingEventKind::FilteringEnd:
			return cudaEventRecord(filteringEnd_.get(), stream);
		case TimingEventKind::ReadEnd:
			return cudaEventRecord(readEnd_.get(), stream);
		}
		// Unreachable
		return cudaErrorUnknown;
	}

	// Get the raw timing data. This call implicitly waits for the last event to complete.
	cudaError_t getTimingData(TimingData& timingData) {
		cudaError_t error;

		error = synchronizeReadEnd();
		if (error) {
			return error;
		}

		error = cudaEventElapsedTime(&timingData.writeDuration, start_.get(), writeEnd_.get());
		if (error) {
			return error;
		}

		error = cudaEventElapsedTime(&timingData.filteringDuration, writeEnd_.get(), filteringEnd_.get());
		if (error) {
			return error;
		}

		error = cudaEventElapsedTime(&timingData.readDuration, filteringEnd_.get(), readEnd_.get());
		if (error) {
			return error;
		}

		error = cudaEventElapsedTime(&timingData.latency, start_.get(), readEnd_.get());
		if (error) {
			return error;
		}

		return cudaSuccess;
	}

private:
	// Assuming the events happen sequentially, we only need to sync with the last one.
	cudaError_t synchronizeReadEnd() {
		return cudaEventSynchronize(readEnd_.get());
	}

	std::unique_ptr<CUevent_st, CudaEventDestroyer> start_;
	std::unique_ptr<CUevent_st, CudaEventDestroyer> writeEnd_;
	std::unique_ptr<CUevent_st, CudaEventDestroyer> filteringEnd_;
	std::unique_ptr<CUevent_st, CudaEventDestroyer> readEnd_;
};

struct CudaStreamDestroyer {
	void operator()(cudaStream_t ptr) { cudaStreamDestroy(ptr); }
};

// Wrapper type that bundles `NppStreamContext` and `cudaStream_t` together
class NppStream {
public:
	NppStream() {
		cudaStream_t stream;
		CUDA_CHECK(cudaStreamCreate(&stream));
		stream_ = std::unique_ptr<CUstream_st, CudaStreamDestroyer>(stream);

		ctx_.hStream = stream_.get();
		CUDA_CHECK(cudaGetDevice(&ctx_.nCudaDeviceId));
		CUDA_CHECK(
			cudaDeviceGetAttribute(
				&ctx_.nCudaDevAttrComputeCapabilityMajor,
				cudaDevAttrComputeCapabilityMajor,
				ctx_.nCudaDeviceId
			)
		);
		CUDA_CHECK(
			cudaDeviceGetAttribute(
				&ctx_.nCudaDevAttrComputeCapabilityMinor,
				cudaDevAttrComputeCapabilityMinor,
				ctx_.nCudaDeviceId
			)
		);
		CUDA_CHECK(cudaStreamGetFlags(ctx_.hStream, &ctx_.nStreamFlags));

		cudaDeviceProp deviceProperties;
		CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, ctx_.nCudaDeviceId));
		ctx_.nMultiProcessorCount = deviceProperties.multiProcessorCount;
		ctx_.nMaxThreadsPerMultiProcessor = deviceProperties.maxThreadsPerMultiProcessor;
		ctx_.nMaxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;
		ctx_.nSharedMemPerBlock = deviceProperties.sharedMemPerBlock;
	}

	NppStreamContext context() {
		return ctx_;
	}

	cudaStream_t stream() {
		return stream_.get();
	}
private:
	NppStreamContext ctx_;
	std::unique_ptr<CUstream_st, CudaStreamDestroyer> stream_;
};
