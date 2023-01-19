#pragma once

#include <atomic>
#include <vector>
#include <algorithm>
#include <thread>

// Lock-free ring buffer that can be shared between threads
template<typename T>
class AtomicRingBuffer
{
public:
	AtomicRingBuffer(int powTwo): head_(0), tail_(0) {
		for (int i = 0; i < (1 << powTwo); i++) {
			buffer_.emplace_back();
		}
	}

	// Move only; no copying
	AtomicRingBuffer(const AtomicRingBuffer&) = delete;
	AtomicRingBuffer& operator= (const AtomicRingBuffer&) = delete;
	AtomicRingBuffer(AtomicRingBuffer&& other) = default;
	AtomicRingBuffer& operator=(AtomicRingBuffer&&) = default;

	void push(T&& val) {
		size_t head = head_.load(std::memory_order_acquire);
		for (;;) {
			size_t tail = tail_.load(std::memory_order_acquire);

			// Proceed if not full; The indices can wrap around so `!=` must be used here
			if (head - tail != buffer_.size()) {
				break;
			}
			else {
				std::this_thread::yield();
			}
		}

		size_t index = head & (buffer_.size() - 1);
		std::swap(buffer_[index], val);
		head_.store(head + 1, std::memory_order_release);
	}

	T pop() {
		size_t tail = tail_.load(std::memory_order_acquire);
		for (;;) {
			size_t head = head_.load(std::memory_order_acquire);

			// Proceed if not empty; `head` is not always >= `tail` because of wrap-around
			if (head != tail) {
				break;
			}
			else {
				std::this_thread::yield();
			}
		}

		size_t index = tail & (buffer_.size() - 1);
		T val;
		std::swap(buffer_[index], val);
		tail_.store(tail + 1, std::memory_order_release);
		return val;
	}

private:
	std::atomic_size_t head_;
	std::atomic_size_t tail_;
	std::vector<T> buffer_;
};

