#ifndef IMAGE_H
#define IMAGE_H

#include <Eigen/StdVector>
#include <cassert>
#include <type_traits>
#include <vector>

namespace se {

template<typename T>
class Image {
    public:

    Image(const int w, const int h) : width_(w), height_(h), data_(width_ * height_)
    {
        assert(width_ > 0 && height_ > 0);
    }

    Image(const int w, const int h, const T& val) :
            width_(w), height_(h), data_(width_ * height_, val)
    {
        assert(width_ > 0 && height_ > 0);
    }

    T& operator[](std::size_t idx)
    {
        return data_[idx];
    }
    const T& operator[](std::size_t idx) const
    {
        return data_[idx];
    }

    T& operator()(const int x, const int y)
    {
        return data_[x + y * width_];
    }
    const T& operator()(const int x, const int y) const
    {
        return data_[x + y * width_];
    }

    std::size_t size() const
    {
        return width_ * height_;
    };
    int width() const
    {
        return width_;
    };
    int height() const
    {
        return height_;
    };

    T* data()
    {
        return data_.data();
    }
    const T* data() const
    {
        return data_.data();
    }

    auto begin()
    {
        return data_.begin();
    }
    auto begin() const
    {
        return data_.cbegin();
    }
    auto cbegin() const
    {
        return data_.cbegin();
    }

    auto end()
    {
        return data_.end();
    }
    auto end() const
    {
        return data_.cend();
    }
    auto cend() const
    {
        return data_.cend();
    }

    private:
    int width_;
    int height_;
    std::vector<T, Eigen::aligned_allocator<T>> data_;

    static_assert(!std::is_same<T, bool>::value,
                  "Use char/uint8_t instead of bool to avoid the std::vector<bool> specialization");
};

template<typename T>
using ImageVec = std::vector<Image<T>, Eigen::aligned_allocator<Image<T>>>;

} // end namespace se
#endif
