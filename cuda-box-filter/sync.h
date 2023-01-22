#pragma once

#include <atomic>

class Signal {
public:
	Signal() : stop_(false) {}

	void signalStop() {
		stop_.store(true, std::memory_order_release);
	}

	bool shouldStop() {
		return stop_.load(std::memory_order_acquire);
	}

private:
	std::atomic_bool stop_;
};

class Barrier {
public:
	Barrier(size_t numWaiters) : numWaiters_(numWaiters) {}
	
	void wait() {
		counter_.fetch_add(1, std::memory_order_seq_cst);
		while (counter_.load(std::memory_order_seq_cst) < numWaiters_);
	}
private:
	std::atomic_size_t counter_;
	size_t numWaiters_;
};