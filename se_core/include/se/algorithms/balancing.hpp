/*

Copyright 2019 Emanuele Vespa, Imperial College London 

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

*/
#ifndef BALANCING_HPP
#define BALANCING_HPP
#include <Eigen/Dense>
#include <unordered_set>
#include <vector>

#include "se/node_iterator.hpp"
#include "se/octree.hpp"

namespace se {
/*! \brief Enforces 2:1 balance on the tree.
 *  \param octree unbalanced octree to be balanced. 
 *
 * */
template<typename T>
void balance(se::Octree<T>& octree)
{
    std::unordered_set<key_t> octants;
    std::vector<key_t> alloc_buffer;
    Eigen::Matrix<int, 4, 6> N;
    se::node_iterator<T> it(octree);
    int depth = se::math::log2_const(octree.size());
    while (se::Node<T>* n = it.next()) {
        int depth = se::keyops::depth(n->code());
        if (depth == 0)
            continue; // skip root
        se::one_neighbourhood(N, se::parent(n->code(), depth), depth);
        for (int i = 0; i < 6; ++i) {
            Eigen::Ref<Eigen::Matrix<int, 4, 1>> coords(N.col(i));
            key_t key = octree.hash(coords.x(), coords.y(), coords.z(), depth - 1);
            if (!octree.fetch(coords.x(), coords.y(), coords.z()) && octants.insert(key).second) {
                alloc_buffer.push_back(key);
            }
        }
    }

    int last = 0;
    int end = alloc_buffer.size();
    while (end - last > 0) {
        for (; last < end; ++last) {
            se::one_neighbourhood(N, se::parent(alloc_buffer[last], depth), depth);
            int depth = se::keyops::depth(alloc_buffer[last]);
            if (depth == 0)
                continue; // skip root
            for (int i = 0; i < 6; ++i) {
                Eigen::Ref<Eigen::Matrix<int, 4, 1>> coords(N.col(i));
                key_t key = octree.hash(coords.x(), coords.y(), coords.z(), depth - 1);
                if (!octree.fetch(coords.x(), coords.y(), coords.z())
                    && octants.insert(key).second) {
                    alloc_buffer.push_back(key);
                }
            }
        }
        end = alloc_buffer.size();
    }
    octree.allocate(alloc_buffer.data(), alloc_buffer.size());
}
} // namespace se
#endif
