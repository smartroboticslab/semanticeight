// SPDX-FileCopyrightText: 2022 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2022 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include <fstream>
#include <iostream>

namespace se {

template<typename T>
int save_pgm(const Image<T>& image, const std::string& filename)
{
    std::ofstream f(filename.c_str());
    if (!f.is_open()) {
        std::cerr << "Unable to write file " << filename << "\n";
        return 1;
    }
    // Write the PGM header.
    f << "P2\n";
    f << image.width() << " " << image.height() << "\n";
    const int max_value = (1 << 8 * sizeof(T)) - 1;
    f << max_value << "\n";
    // Write the image data.
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            f << static_cast<int>(image(x, y));
            // Do not add a whitespace after the last element of a row.
            if (x < image.width() - 1) {
                f << " ";
            }
        }
        // Add a newline at the end of each row.
        f << "\n";
    }
    f.close();
    return f.good();
    static_assert(!std::is_same<T, uint8_t>::value || !std::is_same<T, uint16_t>::value,
                  "Only 8 and 16 bit grayscale images are supported.");
}

} // namespace se
