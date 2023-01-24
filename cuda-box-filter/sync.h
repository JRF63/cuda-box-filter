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

	void reset() {
		stop_.store(false, std::memory_order_release);
	}

private:
	std::atomic_bool stop_;
};
