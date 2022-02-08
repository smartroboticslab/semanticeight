// SPDX-FileCopyrightText: 2022 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2022 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEMANTIC_CLASSES_HPP
#define SEMANTIC_CLASSES_HPP

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace se {

struct SemanticClass {
    /** The map resolution used for most objects. */
    static constexpr float default_res = 0.02f;

    int id;
    std::string name;
    bool enabled = false;
    float res = default_res;
};

class SemanticClasses {
    public:
    SemanticClasses(const std::vector<SemanticClass>& classes);

    bool enabled(const int class_id) const;
    bool enabled(const std::string& class_name) const;
    void setEnabled(const int class_id, const bool enabled = true);
    void setEnabled(const std::string& class_name, const bool enabled = true);

    void setEnabledAll(const bool enabled = true);

    float res(const int class_id) const;
    float res(const std::string& class_name) const;
    void setRes(const int class_id, const float res = SemanticClass::default_res);
    void setRes(const std::string& class_name, const float res = SemanticClass::default_res);

    void setResAll(const float res = SemanticClass::default_res);

    const std::string& name(const int class_id) const;
    int id(const std::string& class_name) const;

    int backgroundId() const;
    int invalidId() const;

    size_t size() const;

    const std::map<int, SemanticClass>& classes() const;

    /** Return an instance containing the 80 COCO semantic classes plus the background.
     */
    static SemanticClasses coco_classes();
    /** Return an instance containing the 41 Matterport3D semantic classes plus the background.
     */
    static SemanticClasses matterport3d_classes();
    /** Return an instance containing the 101 Replica semantic classes plus the background.
     */
    static SemanticClasses replica_classes();

    private:
    std::map<int, SemanticClass> classes_;
    int background_id_;
    int invalid_id_;
};

std::ostream& operator<<(std::ostream& os, const SemanticClasses& classes);

} // namespace se

#endif // SEMANTIC_CLASSES_HPP
