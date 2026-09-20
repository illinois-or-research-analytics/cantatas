#ifndef LADYBUG_MMAP_H
#define LADYBUG_MMAP_H

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace ladybug {

enum class MmapMode {
  ReadOnly,
  ReadWrite,
  CreateOrTruncate
};

enum class MadviseAdvice {
  Normal = POSIX_MADV_NORMAL,
  Sequential = POSIX_MADV_SEQUENTIAL,
  Random = POSIX_MADV_RANDOM,
  WillNeed = POSIX_MADV_WILLNEED,
  DontNeed = POSIX_MADV_DONTNEED
};

/*
 * MmapBuffer<T>: A zero-copy, hardware-page-aligned memory-mapped persistent array.
 * Delegates page tracking directly to the OS kernel page table, eliminating user-space
 * buffer bloat and enabling direct streaming to/from NVMe storage.
 */
template <typename T>
class MmapBuffer {
public:
  MmapBuffer()
      : fd_(-1), data_(nullptr), size_(0), capacity_(0), mode_(MmapMode::ReadWrite) {}

  MmapBuffer(const std::string &filepath, MmapMode mode, size_t initial_capacity = 0)
      : fd_(-1), data_(nullptr), size_(0), capacity_(0), mode_(mode), filepath_(filepath) {
    Open(filepath, mode, initial_capacity);
  }

  ~MmapBuffer() {
    Close();
  }

  // Disable copy semantics to prevent accidental unmap race conditions
  MmapBuffer(const MmapBuffer &) = delete;
  MmapBuffer &operator=(const MmapBuffer &) = delete;

  // Move semantics
  MmapBuffer(MmapBuffer &&other) noexcept
      : fd_(other.fd_), data_(other.data_), size_(other.size_),
        capacity_(other.capacity_), mode_(other.mode_), filepath_(std::move(other.filepath_)) {
    other.fd_ = -1;
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
  }

  MmapBuffer &operator=(MmapBuffer &&other) noexcept {
    if (this != &other) {
      Close();
      fd_ = other.fd_;
      data_ = other.data_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      mode_ = other.mode_;
      filepath_ = std::move(other.filepath_);

      other.fd_ = -1;
      other.data_ = nullptr;
      other.size_ = 0;
      other.capacity_ = 0;
    }
    return *this;
  }

  void Open(const std::string &filepath, MmapMode mode, size_t initial_capacity = 0) {
    Close();
    filepath_ = filepath;
    mode_ = mode;

    int open_flags = 0;
    mode_t create_mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;

    if (mode == MmapMode::ReadOnly) {
      open_flags = O_RDONLY;
    } else if (mode == MmapMode::ReadWrite) {
      open_flags = O_RDWR | O_CREAT;
    } else if (mode == MmapMode::CreateOrTruncate) {
      open_flags = O_RDWR | O_CREAT | O_TRUNC;
    }

    fd_ = open(filepath.c_str(), open_flags, create_mode);
    if (fd_ < 0) {
      throw std::runtime_error("MmapBuffer::Open failed to open file '" + filepath +
                               "': " + std::string(strerror(errno)));
    }

    struct stat st;
    if (fstat(fd_, &st) < 0) {
      close(fd_);
      fd_ = -1;
      throw std::runtime_error("MmapBuffer::Open failed to fstat file '" + filepath +
                               "': " + std::string(strerror(errno)));
    }

    size_t file_bytes = static_cast<size_t>(st.st_size);
    size_t min_bytes = initial_capacity * sizeof(T);

    if (mode != MmapMode::ReadOnly && file_bytes < min_bytes) {
      // Ensure page alignment
      size_t page_size = sysconf(_SC_PAGESIZE);
      size_t aligned_bytes = ((min_bytes + page_size - 1) / page_size) * page_size;
      if (aligned_bytes == 0) {
        aligned_bytes = page_size;
      }
      if (ftruncate(fd_, aligned_bytes) != 0) {
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("MmapBuffer::Open failed to ftruncate file '" + filepath +
                                 "': " + std::string(strerror(errno)));
      }
      file_bytes = aligned_bytes;
    }

    if (file_bytes == 0) {
      // Empty file in ReadOnly mode or 0 capacity requested
      data_ = nullptr;
      size_ = 0;
      capacity_ = 0;
      return;
    }

    int prot = (mode == MmapMode::ReadOnly) ? PROT_READ : (PROT_READ | PROT_WRITE);
    int flags = MAP_SHARED;

    void *mapped = mmap(nullptr, file_bytes, prot, flags, fd_, 0);
    if (mapped == MAP_FAILED) {
      close(fd_);
      fd_ = -1;
      throw std::runtime_error("MmapBuffer::Open failed to mmap file '" + filepath +
                               "': " + std::string(strerror(errno)));
    }

    data_ = static_cast<T *>(mapped);
    capacity_ = file_bytes / sizeof(T);
    size_ = capacity_; // Default size to capacity unless explicitly managed
  }

  void Close() {
    if (data_ != nullptr && capacity_ > 0) {
      Sync(false);
      munmap(data_, capacity_ * sizeof(T));
      data_ = nullptr;
    }
    if (fd_ >= 0) {
      close(fd_);
      fd_ = -1;
    }
    size_ = 0;
    capacity_ = 0;
  }

  void Resize(size_t new_size) {
    if (mode_ == MmapMode::ReadOnly) {
      throw std::runtime_error("Cannot resize ReadOnly MmapBuffer");
    }
    if (new_size > capacity_) {
      Reserve(std::max(new_size, capacity_ == 0 ? 1024 : capacity_ * 2));
    }
    size_ = new_size;
  }

  void Reserve(size_t new_capacity) {
    if (mode_ == MmapMode::ReadOnly) {
      throw std::runtime_error("Cannot reserve ReadOnly MmapBuffer");
    }
    if (new_capacity <= capacity_) {
      return;
    }

    size_t page_size = sysconf(_SC_PAGESIZE);
    size_t new_bytes = new_capacity * sizeof(T);
    size_t aligned_bytes = ((new_bytes + page_size - 1) / page_size) * page_size;
    size_t old_bytes = capacity_ * sizeof(T);

    if (ftruncate(fd_, aligned_bytes) != 0) {
      throw std::runtime_error("MmapBuffer::Reserve failed to ftruncate file '" + filepath_ +
                               "': " + std::string(strerror(errno)));
    }

    int prot = PROT_READ | PROT_WRITE;
    int flags = MAP_SHARED;

    void *new_data = nullptr;
#if defined(__linux__) && defined(_GNU_SOURCE)
    if (data_ != nullptr && old_bytes > 0) {
      new_data = mremap(data_, old_bytes, aligned_bytes, MREMAP_MAYMOVE);
    } else {
      new_data = mmap(nullptr, aligned_bytes, prot, flags, fd_, 0);
    }
#else
    if (data_ != nullptr && old_bytes > 0) {
      munmap(data_, old_bytes);
    }
    new_data = mmap(nullptr, aligned_bytes, prot, flags, fd_, 0);
#endif

    if (new_data == MAP_FAILED) {
      data_ = nullptr;
      capacity_ = 0;
      size_ = 0;
      throw std::runtime_error("MmapBuffer::Reserve failed to remap file '" + filepath_ +
                               "': " + std::string(strerror(errno)));
    }

    data_ = static_cast<T *>(new_data);
    capacity_ = aligned_bytes / sizeof(T);
  }

  void PushBack(const T &value) {
    if (size_ >= capacity_) {
      Reserve(capacity_ == 0 ? 1024 : capacity_ * 2);
    }
    data_[size_++] = value;
  }

  void AppendSpan(std::span<const T> elements) {
    if (size_ + elements.size() > capacity_) {
      Reserve(std::max(size_ + elements.size(), capacity_ * 2));
    }
    std::memcpy(data_ + size_, elements.data(), elements.size() * sizeof(T));
    size_ += elements.size();
  }

  void Sync(bool async = false) {
    if (data_ != nullptr && capacity_ > 0 && mode_ != MmapMode::ReadOnly) {
      int flags = async ? MS_ASYNC : MS_SYNC;
      msync(data_, capacity_ * sizeof(T), flags);
    }
  }

  void Advise(MadviseAdvice advice, size_t element_offset = 0, size_t element_count = 0) {
    if (data_ != nullptr && capacity_ > 0) {
      size_t byte_offset = element_offset * sizeof(T);
      size_t byte_count = (element_count == 0) ? (capacity_ * sizeof(T) - byte_offset)
                                               : (element_count * sizeof(T));
      posix_madvise(reinterpret_cast<char *>(data_) + byte_offset, byte_count,
                    static_cast<int>(advice));
    }
  }

  // Accessors
  inline T *data() noexcept { return data_; }
  inline const T *data() const noexcept { return data_; }
  inline size_t size() const noexcept { return size_; }
  inline size_t capacity() const noexcept { return capacity_; }
  inline bool empty() const noexcept { return size_ == 0; }
  inline const std::string &filepath() const noexcept { return filepath_; }

  inline void SetSize(size_t s) noexcept { size_ = s; }

  inline T &operator[](size_t index) noexcept { return data_[index]; }
  inline const T &operator[](size_t index) const noexcept { return data_[index]; }

  inline std::span<T> span() noexcept {
    return std::span<T>(data_, size_);
  }

  inline std::span<const T> span() const noexcept {
    return std::span<const T>(data_, size_);
  }

  inline std::span<T> subspan(size_t offset, size_t count) noexcept {
    return std::span<T>(data_ + offset, count);
  }

  inline std::span<const T> subspan(size_t offset, size_t count) const noexcept {
    return std::span<const T>(data_ + offset, count);
  }

private:
  int fd_;
  T *data_;
  size_t size_;
  size_t capacity_;
  MmapMode mode_;
  std::string filepath_;
};

} // namespace ladybug

#endif // LADYBUG_MMAP_H
