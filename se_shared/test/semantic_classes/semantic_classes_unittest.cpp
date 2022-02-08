// SPDX-FileCopyrightText: 2022 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2022 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "se/semantic_classes.hpp"

#include <gtest/gtest.h>
#include <memory>



class SemanticClassesTest : public ::testing::Test {
    protected:
    SemanticClassesTest() : classes(se::SemanticClasses::coco_classes())
    {
    }

    se::SemanticClasses classes;
};



TEST_F(SemanticClassesTest, Initialization)
{
    EXPECT_EQ(classes.backgroundId(), 0);
    EXPECT_EQ(classes.invalidId(), -1);
    EXPECT_EQ(classes.size(), 80 + 1);
    EXPECT_EQ(classes.classes().size(), 80 + 2);
    for (const auto& p : classes.classes()) {
        EXPECT_FALSE(p.second.name.empty());
        EXPECT_FALSE(p.second.enabled);
        EXPECT_FLOAT_EQ(p.second.res, se::SemanticClass::default_res);
    }
}

TEST_F(SemanticClassesTest, Enabled)
{
    EXPECT_FALSE(classes.enabled("chair"));
    classes.setEnabled("chair");
    EXPECT_TRUE(classes.enabled("chair"));

    classes.setEnabledAll(true);
    for (const auto& p : classes.classes()) {
        EXPECT_TRUE(p.second.enabled);
    }
}

TEST_F(SemanticClassesTest, Res)
{
    const float new_res = se::SemanticClass::default_res + 0.01f;
    EXPECT_FLOAT_EQ(classes.res("chair"), se::SemanticClass::default_res);
    classes.setRes("chair", new_res);
    EXPECT_FLOAT_EQ(classes.res("chair"), new_res);

    classes.setResAll(new_res);
    for (const auto& p : classes.classes()) {
        EXPECT_FLOAT_EQ(p.second.res, new_res);
    }
}

TEST_F(SemanticClassesTest, IdName)
{
    for (const auto& p : classes.classes()) {
        EXPECT_EQ(p.first, classes.id(classes.name(p.first)));
    }

    ASSERT_THROW(classes.id("foo"), std::out_of_range);
}
