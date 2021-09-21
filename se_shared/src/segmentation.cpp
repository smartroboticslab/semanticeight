/*
 * Created by binbin on 25/04/18.
 * Modified by Sotiris Papatheodorou
 */

#include "se/segmentation.hpp"

namespace se {
  // The names of the 41 Matterport3D semantic classes plus the background. Everything but "chair" is stuff.
  const std::vector<std::string> matterport3d_class_names    {"background",  "wall",       "floor",      "chair",      "door",       "table",      "picture",    "cabinet",    "cushion",    "window",     "sofa",       "bed",        "curtain",    "chest_of_drawers",  "plant",      "sink",       "stairs",     "ceiling",    "toilet",     "stool",      "towel",      "mirror",     "tv_monitor",  "shower",     "column",     "bathtub",    "counter",    "fireplace",  "lighting",   "beam",       "railing",    "shelving",   "blinds",     "gym_equipment",  "seating",    "board_panel",  "furniture",   "appliances",  "clothes",    "objects",    "misc",       "unlabel"};
  const std::set<std::string> matterport3d_stuff_class_names {"background",  "wall",       "floor",                    "door",       "table",      "picture",    "cabinet",    "cushion",    "window",     "sofa",       "bed",        "curtain",    "chest_of_drawers",  "plant",      "sink",       "stairs",     "ceiling",    "toilet",     "stool",      "towel",      "mirror",     "tv_monitor",  "shower",     "column",     "bathtub",    "counter",    "fireplace",  "lighting",   "beam",       "railing",    "shelving",   "blinds",     "gym_equipment",  "seating",    "board_panel",  "furniture",   "appliances",  "clothes",    "objects",    "misc",       "unlabel"};
  const std::vector<float> matterport3d_class_res            {default_res,   default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,         default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,   default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,      default_res,  default_res,    default_res,   default_res,   default_res,  default_res,  default_res,  default_res};
  // The names of the 80 COCO semantic classes plus the background. Everything but "book" is stuff.
  const std::vector<std::string> coco_class_names    {"void",       "person",     "bicycle",    "car",        "motorcycle",  "airplane",   "bus",        "train",      "truck",      "boat",       "traffic_light",  "fire_hydrant",  "stop_sign",  "parking_meter",  "bench",      "bird",       "cat",        "dog",        "horse",      "sheep",      "cow",        "elephant",   "bear",       "zebra",      "giraffe",    "backpack",   "umbrella",   "handbag",    "tie",        "suitcase",   "frisbee",    "skis",       "snowboard",  "sports_ball",  "kite",       "baseball_bat",  "baseball_glove",  "skateboard",  "surfboard",  "tennis_racket",  "bottle",     "wine_glass",  "cup",        "fork",       "knife",      "spoon",      "bowl",       "banana",     "apple",      "sandwich",   "orange",     "broccoli",   "carrot",     "hot_dog",    "pizza",      "donut",      "cake",       "chair",      "couch",      "potted_plant",  "bed",        "dining_table",  "toilet",     "tv",         "laptop",     "mouse",      "remote",     "keyboard",   "cell_phone",  "microwave",  "oven",       "toaster",    "sink",       "refrigerator",  "book",      "clock",      "vase",       "scissors",   "teddy_bear",  "hair_drier",  "toothbrush"};
  const std::set<std::string> coco_stuff_class_names {"void",       "person",     "bicycle",    "car",        "motorcycle",  "airplane",   "bus",        "train",      "truck",      "boat",       "traffic_light",  "fire_hydrant",  "stop_sign",  "parking_meter",  "bench",      "bird",       "cat",        "dog",        "horse",      "sheep",      "cow",        "elephant",   "bear",       "zebra",      "giraffe",    "backpack",   "umbrella",   "handbag",    "tie",        "suitcase",   "frisbee",    "skis",       "snowboard",  "sports_ball",  "kite",       "baseball_bat",  "baseball_glove",  "skateboard",  "surfboard",  "tennis_racket",  "bottle",     "wine_glass",  "cup",        "fork",       "knife",      "spoon",      "bowl",       "banana",     "apple",      "sandwich",   "orange",     "broccoli",   "carrot",     "hot_dog",    "pizza",      "donut",      "cake",       "chair",      "couch",      "potted_plant",  "bed",        "dining_table",  "toilet",     "tv",         "laptop",     "mouse",      "remote",     "keyboard",   "cell_phone",  "microwave",  "oven",       "toaster",    "sink",       "refrigerator",               "clock",      "vase",       "scissors",   "teddy_bear",  "hair_drier",  "toothbrush"};
  const std::vector<float> coco_class_res            {default_res,  default_res,  default_res,  default_res,  default_res,   default_res,  default_res,  default_res,  default_res,  default_res,  default_res,      default_res,     default_res,  default_res,      default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,    default_res,  default_res,     default_res,       default_res,   default_res,  default_res,      default_res,  default_res,   default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,     default_res,  default_res,     default_res,  default_res,  default_res,  default_res,  default_res,  default_res,  default_res,   default_res,  default_res,  default_res,  default_res,  default_res,     default_res, default_res,  default_res,  default_res,  default_res,   default_res,   default_res};



  // Use COCO classes by default.
  std::vector<std::string> class_names = coco_class_names;
  std::set<std::string> stuff_class_names = coco_stuff_class_names;
  std::vector<float> class_res = coco_class_res;



  void use_coco_classes() {
    class_names = coco_class_names;
    stuff_class_names = coco_stuff_class_names;
    class_res = coco_class_res;
  }



  void use_matterport3d_classes() {
    class_names = matterport3d_class_names;
    stuff_class_names = matterport3d_stuff_class_names;
    class_res = matterport3d_class_res;
  }



  void set_thing(const std::string& class_name) {
    stuff_class_names.erase(class_name);
  }



  void set_stuff(const std::string& class_name) {
    for (const auto& available_class_name : class_names) {
      if (class_name == available_class_name) {
        stuff_class_names.insert(class_name);
        break;
      }
    }
  }



  float IoU(const cv::Mat& mask1, const cv::Mat& mask2) {
    assert((mask1.rows == mask2.rows) && "Masks have different number of rows\n");
    assert((mask1.cols == mask2.cols) && "Masks have different number of columns\n");
    assert((mask1.type() == mask2.type()) && "Masks have different type\n");

    cv::Mat mask_intersection;
    cv::bitwise_and(mask1, mask2, mask_intersection);
    cv::Mat mask_union;
    cv::bitwise_or(mask1, mask2, mask_union);

    const float intersection_area = cv::countNonZero(mask_intersection);
    const float union_area = cv::countNonZero(mask_union);
    const float iou = intersection_area / union_area;
    return iou;
  }



  float notIoU(const cv::Mat& mask1, const cv::Mat& mask2) {
    assert((mask1.rows == mask2.rows) && "Masks have different number of rows\n");
    assert((mask1.cols == mask2.cols) && "Masks have different number of columns\n");
    assert((mask1.type() == mask2.type()) && "Masks have different type\n");

    cv::Mat mask_intersection (mask1.size(), mask1.type());
    cv::bitwise_and(mask1, mask2, mask_intersection);

    const float intersection_area = cv::countNonZero(mask_intersection);
    const float mask1_area = cv::countNonZero(mask1);
    const float mask2_area = cv::countNonZero(mask2);

    const float mask1_ratio = intersection_area / mask1_area;
    const float mask2_ratio = intersection_area / mask2_area;

    const float ratio = fmaxf(mask1_ratio, mask2_ratio);

    return ratio;
  }



  cv::Mat extract_instance(const cv::Mat&                 instance_mask,
                           const se::instance_mask_elem_t instance_id) {

    // Create a copy of the original mask.
    cv::Mat individual_mask;
    instance_mask.copyTo(individual_mask);

    // Set all elements matching the instance ID to 255 and all others to 0.
    const auto match_id = [instance_id](se::instance_mask_elem_t& pixel, const int*) {
          pixel = (pixel == instance_id) ? 255 : 0;
        };
    individual_mask.forEach<se::instance_mask_elem_t>(match_id);

    // Convert from se::instance_mask_t to se::mask_t.
    individual_mask.convertTo(individual_mask, se::mask_t);

    return individual_mask;
  }



  std::string class_id_to_str(const int class_id) {
    if (0 <= class_id && static_cast<size_t>(class_id) < class_names.size()) {
      return se::class_names[class_id];
    } else {
      return "invalid";
    }
  }



  bool is_class_stuff(const int class_id) {
    return stuff_class_names.count(class_id_to_str(class_id));
  }

  void print_classes() {
    for (size_t i = 0; i < class_names.size(); i++) {
      printf("%2zd %s\n", i, class_id_to_str(i).c_str());
    }
  }

} // namespace se

