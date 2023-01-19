#pragma once

#include "errors.h"

#include "cuda_runtime.h"
#include "nppi.h"

enum class TimingEventKind {
	START,
	WRITE_END,
	FILTERING_END,
	READ_END,
};

class TimingEvents {
public:
	TimingEvents() {
		CUDA_CHECK(cudaEventCreate(&start_));
		CUDA_CHECK(cudaEventCreate(&writeEnd_));
		CUDA_CHECK(cudaEventCreate(&filteringEnd_));
		CUDA_CHECK(cudaEventCreate(&readEnd_));
	}

	~TimingEvents() {
		CUDA_CHECK(cudaEventDestroy(start_));
		CUDA_CHECK(cudaEventDestroy(writeEnd_));
		CUDA_CHECK(cudaEventDestroy(filteringEnd_));
		CUDA_CHECK(cudaEventDestroy(readEnd_));
	}

	// Move only; no copying
	TimingEvents(const TimingEvents&) = delete;
	TimingEvents& operator= (const TimingEvents&) = delete;
	TimingEvents(TimingEvents&& other) = default;
	TimingEvents& operator=(TimingEvents&&) = default;

	// Records the event to the stream asynchronously.
	cudaError_t record(TimingEventKind kind, cudaStream_t stream) {
		switch (kind) {
		case TimingEventKind::START:
			return cudaEventRecord(start_, stream);
		case TimingEventKind::WRITE_END:
			return cudaEventRecord(writeEnd_, stream);
		case TimingEventKind::FILTERING_END:
			return cudaEventRecord(filteringEnd_, stream);
		case TimingEventKind::READ_END:
			return cudaEventRecord(readEnd_, stream);
		}
		// Unreachable
		return cudaErrorUnknown;
	}

	// Get the raw timing data. This call implicitly waits for the last event to complete.
	cudaError_t getTimingData(
		float* writeDuration,
		float* filteringDuration,
		float* readDuration,
		float* latency
	) {
		cudaError_t error;

		error = synchronizeReadEnd();
		if (error) {
			return error;
		}

		error = cudaEventElapsedTime(writeDuration, start_, writeEnd_);
		if (error) {
			return error;
		}

		error = cudaEventElapsedTime(filteringDuration, writeEnd_, filteringEnd_);
		if (error) {
			return error;
		}

		error = cudaEventElapsedTime(readDuration, filteringEnd_, readEnd_);
		if (error) {
			return error;
		}

		error = cudaEventElapsedTime(latency, start_, readEnd_);
		if (error) {
			return error;
		}

		return cudaSuccess;
	}

private:
	// Assuming the events happen sequentially, we only need to sync with the last one.
	cudaError_t synchronizeReadEnd() {
		return cudaEventSynchronize(readEnd_);
	}

	cudaEvent_t start_;
	cudaEvent_t writeEnd_;
	cudaEvent_t filteringEnd_;
	cudaEvent_t readEnd_;
};

// Wrapper type that bundles `NppStreamContext` and `cudaStream_t` together
class NppStream {
public:
	NppStream() {
		CUDA_CHECK(cudaStreamCreate(&stream_));
		ctx_.hStream = stream_;
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

	~NppStream() {
		CUDA_CHECK(cudaStreamDestroy(stream_));
		// NppStreamContext does not need to be freed
	}

	// Move only; no copying
	NppStream(const NppStream&) = delete;
	NppStream& operator= (const NppStream&) = delete;
	NppStream(NppStream&& other) = default;
	NppStream& operator=(NppStream&&) = default;

	NppStreamContext context() {
		return ctx_;
	}

	cudaStream_t stream() {
		return stream_;
	}
private:
	NppStreamContext ctx_;
	cudaStream_t stream_;
};
