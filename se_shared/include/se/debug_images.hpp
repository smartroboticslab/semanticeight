/*
 * Copyright (C) 2020 Sotiris Papatheodorou
 */

#ifndef __DEBUG_IMAGES_HPP
#define __DEBUG_IMAGES_HPP

#include <cassert>
#include <cstdint>
#include <lodepng.h>
#include <string>
#include <sys/stat.h>
#include <vector>



namespace dbg {
#ifdef SE_DEBUG_IMAGES
class Image {
    public:
    Image(const size_t width, const size_t height, const uint32_t value) :
            width_(width), height_(height)
    {
        assert(width_ > 0 && "Error: width must be positive");
        assert(height_ > 0 && "Error: height must be positive");
        data_.resize(width_ * height_, value);
    }

    Image(const size_t width, const size_t height) : Image(width, height, 0xFF000000)
    {
    }

    size_t size() const
    {
        return width_ * height_;
    }

    size_t width() const
    {
        return width_;
    }

    size_t height() const
    {
        return height_;
    }

    uint32_t& operator()(const size_t x, const size_t y)
    {
        return data_[x + y * width_];
    }

    const uint32_t& operator()(const size_t x, const size_t y) const
    {
        return data_[x + y * width_];
    }

    uint32_t* data()
    {
        return data_.data();
    }

    const uint32_t* data() const
    {
        return data_.data();
    }

    private:
    size_t width_;
    size_t height_;
    std::vector<uint32_t> data_;
};



class DebugImages {
    public:
    DebugImages()
    {
    }

    void init(const size_t num_images, const size_t width, const size_t height)
    {
        assert(width > 0 && "Error: width must be positive");
        assert(height > 0 && "Error: height must be positive");

        images_.clear();
        images_ = std::vector<Image>(num_images, Image(width, height, 0xFF000000));
    }

    void save(const std::string& directory, const std::string& basename) const
    {
        // Create the output directory.
        const int status = mkdir(directory.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if ((status != 0) && (errno != EEXIST)) {
            std::cerr << "Warning: debug image output directory " << directory
                      << " could not be created.\n";
        }

        // Save each image.
        for (size_t i = 0; i < images_.size(); ++i) {
            // Construct the image filename.
            const std::string filename =
                directory + "/" + basename + "_" + std::to_string(i) + ".png";
            // Save the image as PNG.
            const unsigned status =
                lodepng_encode32_file(filename.c_str(),
                                      reinterpret_cast<const unsigned char*>(images_[i].data()),
                                      images_[i].width(),
                                      images_[i].height());
            if (status != 0) {
                std::cerr << "Warning: debug image " << filename << " could not be created.\n";
            }
        }
    }

    void set(const size_t image_index, const size_t x, const size_t y, const uint32_t value)
    {
        assert(image_index < images_.size() && "Error: image_index out of bounds");
        assert(x < images_[image_index].width() && "Error: x out of bounds");
        assert(y < images_[image_index].height() && "Error: y out of bounds");

        if (!images_.empty()) {
            images_[image_index](x, y) = value;
        }
    }

    uint32_t get(const size_t image_index, const size_t x, const size_t y) const
    {
        assert(image_index < images_.size() && "Error: image_index out of bounds");
        assert(x < images_[image_index].width() && "Error: x out of bounds");
        assert(y < images_[image_index].height() && "Error: y out of bounds");

        return images_[image_index](x, y);
    }

    private:
    std::vector<Image> images_;
};
#else
class DebugImages {
    public:
    DebugImages()
    {
    }
    void init(const size_t, const size_t, const size_t)
    {
    }
    void save(const std::string&, const std::string&) const
    {
    }
    void set(const size_t, const size_t, const size_t, const uint32_t)
    {
    }
    uint32_t get(const size_t, const size_t, const size_t) const
    {
        return 0xDEADBEEF;
    }
};
#endif



static DebugImages images;



enum RaycastingResultColors : uint32_t {
    // Green: Object was hit
    raycast_ok = 0xFF00FF00,
    // White: The ray was outside the object bounding box
    raycast_outside_bounding = 0xFFFFFFFF,
    // Blue: Another object closer than this was hit
    raycast_far = 0xFFFF0000,
    // Red: The foreground probability was below 0.5
    raycast_low_fg = 0xFF0000FF,
    // Purple: Same distance but lower foreground probability than another object
    raycast_same_dist_lower_fg = 0xFF700070,
    // Black: missed all objects
    raycast_missed = 0xFF000000,
};
} // End namespace dbg

#endif
