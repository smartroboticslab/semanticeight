// SPDX-FileCopyrightText: 2022 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2022 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include "se/semantic_classes.hpp"

#include <algorithm>
#include <iomanip>

namespace se {

SemanticClasses::SemanticClasses(const std::vector<SemanticClass>& classes)
{
    for (const auto& c : classes) {
        classes_[c.id] = c;
    }
    background_id_ = classes_.empty() ? 0 : classes_.at(id("background")).id;
    invalid_id_ = background_id_ - 1;
    classes_[invalid_id_] = {invalid_id_, "invalid"};
}



bool SemanticClasses::enabled(const int class_id) const
{
    return classes_.at(class_id).enabled;
}

bool SemanticClasses::enabled(const std::string& class_name) const
{
    return enabled(id(class_name));
}



void SemanticClasses::setEnabled(const int class_id, const bool enabled)
{
    classes_.at(class_id).enabled = enabled;
}

void SemanticClasses::setEnabled(const std::string& class_name, const bool enabled)
{
    setEnabled(id(class_name), enabled);
}



void SemanticClasses::setEnabledAll(const bool enabled)
{
    for (auto& pair : classes_) {
        pair.second.enabled = enabled;
    }
}



float SemanticClasses::res(const int class_id) const
{
    return classes_.at(class_id).res;
}

float SemanticClasses::res(const std::string& class_name) const
{
    return res(id(class_name));
}



void SemanticClasses::setRes(const int class_id, const float res)
{
    classes_.at(class_id).res = res;
}

void SemanticClasses::setRes(const std::string& class_name, const float res)
{
    setRes(id(class_name), res);
}



void SemanticClasses::setResAll(const float res)
{
    for (auto& pair : classes_) {
        pair.second.res = res;
    }
}



const std::string& SemanticClasses::name(const int class_id) const
{
    return classes_.at(class_id).name;
}



int SemanticClasses::id(const std::string& name) const
{
    auto it = std::find_if(
        classes_.begin(), classes_.end(), [&](const auto& x) { return x.second.name == name; });
    if (it != classes_.end()) {
        return it->second.id;
    }
    else {
        throw std::out_of_range("Unknown class name " + name);
    }
}



int SemanticClasses::backgroundId() const
{
    return background_id_;
}



int SemanticClasses::invalidId() const
{
    return invalid_id_;
}



size_t SemanticClasses::size() const
{
    // Don't count the invalid class.
    return std::max(classes_.size() - 1, static_cast<size_t>(0));
}



const std::map<int, SemanticClass>& SemanticClasses::classes() const
{
    return classes_;
}



SemanticClasses SemanticClasses::coco_classes()
{
    return SemanticClasses({{0, "background"},
                            {1, "person"},
                            {2, "bicycle"},
                            {3, "car"},
                            {4, "motorcycle"},
                            {5, "airplane"},
                            {6, "bus"},
                            {7, "train"},
                            {8, "truck"},
                            {9, "boat"},
                            {10, "traffic_light"},
                            {11, "fire_hydrant"},
                            {12, "stop_sign"},
                            {13, "parking_meter"},
                            {14, "bench"},
                            {15, "bird"},
                            {16, "cat"},
                            {17, "dog"},
                            {18, "horse"},
                            {19, "sheep"},
                            {20, "cow"},
                            {21, "elephant"},
                            {22, "bear"},
                            {23, "zebra"},
                            {24, "giraffe"},
                            {25, "backpack"},
                            {26, "umbrella"},
                            {27, "handbag"},
                            {28, "tie"},
                            {29, "suitcase"},
                            {30, "frisbee"},
                            {31, "skis"},
                            {32, "snowboard"},
                            {33, "sports_ball"},
                            {34, "kite"},
                            {35, "baseball_bat"},
                            {36, "baseball_glove"},
                            {37, "skateboard"},
                            {38, "surfboard"},
                            {39, "tennis_racket"},
                            {40, "bottle"},
                            {41, "wine_glass"},
                            {42, "cup"},
                            {43, "fork"},
                            {44, "knife"},
                            {45, "spoon"},
                            {46, "bowl"},
                            {47, "banana"},
                            {48, "apple"},
                            {49, "sandwich"},
                            {50, "orange"},
                            {51, "broccoli"},
                            {52, "carrot"},
                            {53, "hot_dog"},
                            {54, "pizza"},
                            {55, "donut"},
                            {56, "cake"},
                            {57, "chair"},
                            {58, "couch"},
                            {59, "potted_plant"},
                            {60, "bed"},
                            {61, "dining_table"},
                            {62, "toilet"},
                            {63, "tv"},
                            {64, "laptop"},
                            {65, "mouse"},
                            {66, "remote"},
                            {67, "keyboard"},
                            {68, "cell_phone"},
                            {69, "microwave"},
                            {70, "oven"},
                            {71, "toaster"},
                            {72, "sink"},
                            {73, "refrigerator"},
                            {74, "book"},
                            {75, "clock"},
                            {76, "vase"},
                            {77, "scissors"},
                            {78, "teddy_bear"},
                            {79, "hair_drier"},
                            {80, "toothbrush"}});
}

SemanticClasses SemanticClasses::matterport3d_classes()
{
    return SemanticClasses({{0, "background"},  {1, "wall"},
                            {2, "floor"},       {3, "chair"},
                            {4, "door"},        {5, "table"},
                            {6, "picture"},     {7, "cabinet"},
                            {8, "cushion"},     {9, "window"},
                            {10, "sofa"},       {11, "bed"},
                            {12, "curtain"},    {13, "chest of drawers"},
                            {14, "plant"},      {15, "sink"},
                            {16, "stairs"},     {17, "ceiling"},
                            {18, "toilet"},     {19, "stool"},
                            {20, "towel"},      {21, "mirror"},
                            {22, "tv monitor"}, {23, "shower"},
                            {24, "column"},     {25, "bathtub"},
                            {26, "counter"},    {27, "fireplace"},
                            {28, "lighting"},   {29, "beam"},
                            {30, "railing"},    {31, "shelving"},
                            {32, "blinds"},     {33, "gym equipment"},
                            {34, "seating"},    {35, "board panel"},
                            {36, "furniture"},  {37, "appliances"},
                            {38, "clothes"},    {39, "objects"},
                            {40, "misc"},       {41, "unlabel"}});
}

SemanticClasses SemanticClasses::replica_classes()
{
    return SemanticClasses({{0, "background"},
                            {1, "backpack"},
                            {2, "base cabinet"},
                            {3, "basket"},
                            {4, "bathtub"},
                            {5, "beam"},
                            {6, "beanbag"},
                            {7, "bed"},
                            {8, "bench"},
                            {9, "bike"},
                            {10, "bin"},
                            {11, "blanket"},
                            {12, "blinds"},
                            {13, "book"},
                            {14, "bottle"},
                            {15, "box"},
                            {16, "bowl"},
                            {17, "camera"},
                            {18, "cabinet"},
                            {19, "candle"},
                            {20, "chair"},
                            {21, "chopping board"},
                            {22, "clock"},
                            {23, "cloth"},
                            {24, "clothing"},
                            {25, "coaster"},
                            {26, "comforter"},
                            {27, "computer keyboard"},
                            {28, "cup"},
                            {29, "cushion"},
                            {30, "curtain"},
                            {31, "ceiling"},
                            {32, "cooktop"},
                            {33, "countertop"},
                            {34, "desk"},
                            {35, "desk organizer"},
                            {36, "desktop computer"},
                            {37, "door"},
                            {38, "exercise ball"},
                            {39, "faucet"},
                            {40, "floor"},
                            {41, "handbag"},
                            {42, "hair dryer"},
                            {43, "handrail"},
                            {44, "indoor plant"},
                            {45, "knife block"},
                            {46, "kitchen utensil"},
                            {47, "lamp"},
                            {48, "laptop"},
                            {49, "major appliance"},
                            {50, "mat"},
                            {51, "microwave"},
                            {52, "monitor"},
                            {53, "mouse"},
                            {54, "nightstand"},
                            {55, "pan"},
                            {56, "panel"},
                            {57, "paper towel"},
                            {58, "phone"},
                            {59, "picture"},
                            {60, "pillar"},
                            {61, "pillow"},
                            {62, "pipe"},
                            {63, "plant stand"},
                            {64, "plate"},
                            {65, "pot"},
                            {66, "rack"},
                            {67, "refrigerator"},
                            {68, "remote control"},
                            {69, "scarf"},
                            {70, "sculpture"},
                            {71, "shelf"},
                            {72, "shoe"},
                            {73, "shower stall"},
                            {74, "sink"},
                            {75, "small appliance"},
                            {76, "sofa"},
                            {77, "stair"},
                            {78, "stool"},
                            {79, "switch"},
                            {80, "table"},
                            {81, "table runner"},
                            {82, "tablet"},
                            {83, "tissue paper"},
                            {84, "toilet"},
                            {85, "toothbrush"},
                            {86, "towel"},
                            {87, "tv screen"},
                            {88, "tv stand"},
                            {89, "umbrella"},
                            {90, "utensil holder"},
                            {91, "vase"},
                            {92, "vent"},
                            {93, "wall"},
                            {94, "wall cabinet"},
                            {95, "wall plug"},
                            {96, "wardrobe"},
                            {97, "window"},
                            {98, "rug"},
                            {99, "logo"},
                            {100, "bag"},
                            {101, "set of clothing"}});
}



std::ostream& operator<<(std::ostream& os, const SemanticClasses& classes)
{
    const std::ios_base::fmtflags f(os.flags());
    for (auto& pair : classes.classes()) {
        const auto& c = pair.second;
        os << std::setw(3) << c.id;
        os << std::setw(20) << c.name;
        os << std::setw(9) << (c.enabled ? "enabled" : "disabled");
        os << std::setw(5) << c.res << "\n";
    }
    os.flags(f);
    return os;
}

} // namespace se
