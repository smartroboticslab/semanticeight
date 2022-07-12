#include "se/segmentation_result.hpp"

#include <gtest/gtest.h>

#include "se/filesystem.hpp"

#ifndef SEQUENCE_PATH
#    define SEQUENCE_PATH "."
#endif

TEST(SegmentationResult, readAndWrite)
{
    const std::string segmentation_in = SEQUENCE_PATH "/segmentation";
    const std::string segmentation_out = stdfs::temp_directory_path()
        / stdfs::path("semanticeight_test_results/segmentation_result_unittest");
    stdfs::create_directories(segmentation_out);

    se::SegmentationResult s(640, 480);
    for (const auto& n : {"1305031453.359684", "1305031453.391690"}) {
        ASSERT_TRUE(s.read(segmentation_in, n));
        ASSERT_TRUE(s.write(segmentation_out, n));
    }
}
