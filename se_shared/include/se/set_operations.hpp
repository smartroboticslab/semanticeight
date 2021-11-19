// SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __SET_OPERATIONS_HPP
#define __SET_OPERATIONS_HPP

#include <set>

namespace se {
/*! \brief Subtract set B from set A in place (A = A \ B).
   */
template<typename T>
void setminus(std::set<T>& A, const std::set<T>& B)
{
    if (!A.empty() && !B.empty()) {
        for (const auto& b : B) {
            A.erase(b);
        }
    }
}

/*! \brief Add all the elements of set B in set A (A = A U B).
   */
template<typename T>
void setunion(std::set<T>& A, const std::set<T>& B)
{
    A.insert(B.begin(), B.end());
}
} // namespace se

#endif // __SET_OPERATIONS_HPP
