#ifndef LADYBUG_COLUMN_H
#define LADYBUG_COLUMN_H

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <new>
#include <span>
#include <type_traits>

namespace ladybug {

/*
 * Column<T>: a growable array backed by an anonymous memory mapping.
 *
 * Reserving capacity only reserves address space; physical memory is used as
 * elements are written. On Linux growth uses mremap, so multi-gigabyte columns
 * are never copied when they grow. Columns are the storage for every large
 * array in LadyBugDB (CSR offsets and targets, node attributes, clusters).
 */
template <typename T> class Column {
  static_assert(std::is_trivially_copyable_v<T>,
                "Column<T> requires a trivially copyable element type");

public:
  Column() = default;
  ~Column() { Release(); }

  Column(const Column &) = delete;
  Column &operator=(const Column &) = delete;

  Column(Column &&other) noexcept
      : data_(other.data_), size_(other.size_), capacity_(other.capacity_),
        mapped_bytes_(other.mapped_bytes_) {
    other.data_ = nullptr;
    other.size_ = other.capacity_ = other.mapped_bytes_ = 0;
  }

  Column &operator=(Column &&other) noexcept {
    if (this != &other) {
      Release();
      data_ = other.data_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      mapped_bytes_ = other.mapped_bytes_;
      other.data_ = nullptr;
      other.size_ = other.capacity_ = other.mapped_bytes_ = 0;
    }
    return *this;
  }

  void Reserve(size_t n) {
    if (n <= capacity_) {
      return;
    }
    size_t bytes = RoundUpToPage(n * sizeof(T));
    void *mapped = nullptr;
    if (data_ == nullptr) {
      mapped = MapAnonymous(bytes);
    } else {
#if defined(__linux__)
      mapped = mremap(data_, mapped_bytes_, bytes, MREMAP_MAYMOVE);
      if (mapped == MAP_FAILED) {
        throw std::bad_alloc();
      }
#else
      mapped = MapAnonymous(bytes);
      std::memcpy(mapped, data_, size_ * sizeof(T));
      munmap(data_, mapped_bytes_);
#endif
    }
    data_ = static_cast<T *>(mapped);
    mapped_bytes_ = bytes;
    capacity_ = bytes / sizeof(T);
  }

  // Resizes, filling any new elements with `fill`.
  void Resize(size_t n, const T &fill = T{}) {
    if (n > capacity_) {
      Reserve(Grow(n));
    }
    if (n > size_) {
      std::fill(data_ + size_, data_ + n, fill);
    }
    size_ = n;
  }

  // Resizes without initializing new elements; the caller must write them.
  void ResizeUninitialized(size_t n) {
    if (n > capacity_) {
      Reserve(Grow(n));
    }
    size_ = n;
  }

  void PushBack(const T &value) {
    if (size_ == capacity_) {
      Reserve(Grow(size_ + 1));
    }
    data_[size_++] = value;
  }

  void Append(const T *src, size_t count) {
    if (count == 0) {
      return;
    }
    if (size_ + count > capacity_) {
      Reserve(Grow(size_ + count));
    }
    std::memcpy(data_ + size_, src, count * sizeof(T));
    size_ += count;
  }

  void Append(std::span<const T> values) { Append(values.data(), values.size()); }

  void Clear() noexcept { size_ = 0; }

  // Unmaps the column and returns its memory to the operating system.
  void Release() noexcept {
    if (data_ != nullptr) {
      munmap(data_, mapped_bytes_);
    }
    data_ = nullptr;
    size_ = capacity_ = mapped_bytes_ = 0;
  }

  T *data() noexcept { return data_; }
  const T *data() const noexcept { return data_; }
  size_t size() const noexcept { return size_; }
  size_t capacity() const noexcept { return capacity_; }
  bool empty() const noexcept { return size_ == 0; }
  size_t SizeBytes() const noexcept { return size_ * sizeof(T); }

  T &operator[](size_t i) noexcept { return data_[i]; }
  const T &operator[](size_t i) const noexcept { return data_[i]; }
  T &back() noexcept { return data_[size_ - 1]; }
  const T &back() const noexcept { return data_[size_ - 1]; }

  T *begin() noexcept { return data_; }
  T *end() noexcept { return data_ + size_; }
  const T *begin() const noexcept { return data_; }
  const T *end() const noexcept { return data_ + size_; }

  std::span<T> span() noexcept { return {data_, size_}; }
  std::span<const T> span() const noexcept { return {data_, size_}; }

private:
  size_t Grow(size_t needed) const {
    return std::max(needed, capacity_ + capacity_ / 2 + 1024);
  }

  static size_t RoundUpToPage(size_t bytes) {
    static const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    if (bytes == 0) {
      bytes = 1;
    }
    return ((bytes + page - 1) / page) * page;
  }

  static void *MapAnonymous(size_t bytes) {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    void *mapped = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (mapped == MAP_FAILED) {
      throw std::bad_alloc();
    }
    return mapped;
  }

  T *data_ = nullptr;
  size_t size_ = 0;
  size_t capacity_ = 0;
  size_t mapped_bytes_ = 0;
};

} // namespace ladybug

#endif // LADYBUG_COLUMN_H
