#pragma once
#include <thread>             // std::thread
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable
#include <iostream>

class RWLock {
	std::mutex m_mutex;				// re-entrance not allowed
	std::condition_variable m_readingAllowed, m_writingAllowed;
	bool m_writeLocked = false;	// locked for writing
	size_t m_readLocked = 0;	// number of concurrent readers

public:
	size_t getReaders() const {
		return m_readLocked;
	}

	void lockR() {
		std::unique_lock<std::mutex> monitor(m_mutex);
		while (m_writeLocked) m_readingAllowed.wait(monitor);
		std::cout << "thread " << std::this_thread::get_id() << " locks for reading " << m_readLocked << std::endl;
		m_readLocked++;
	}

	void unlockR() {
		std::unique_lock<std::mutex> monitor(m_mutex);
		std::cout << "thread " << std::this_thread::get_id() << " unlocks reading " << std::endl;
		m_readLocked--;
	}

	void lockW() {
		std::unique_lock<std::mutex> monitor(m_mutex);
		while (m_readLocked > 0 || m_writeLocked) m_writingAllowed.wait(monitor);
		std::cout << "thread " << std::this_thread::get_id() << " locks for writing " << std::endl;
		m_writeLocked = true;
	} // mutex unlocks in here

	void unlockW() {
		std::cout << "thread " << std::this_thread::get_id() << " unlocks for writing " << std::endl;
		m_writeLocked = false;
	}
};