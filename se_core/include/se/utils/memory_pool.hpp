/*

Copyright 2016 Emanuele Vespa, Imperial College London

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

#ifndef MEM_POOL_H
#define MEM_POOL_H

#include <iostream>
#include <vector>
#include <atomic>
#include <mutex>
#include <se/node.hpp>
#include <se/octant_ops.hpp>

namespace se {
/*! \brief Manage the memory allocated for Octree nodes.
 */

  template <typename T>
  class DummyMemoryPool {
  public:
    DummyMemoryPool() {
      root_ = new se::Node<T>;
      nodes_updated_ = false;
      blocks_updated_ = false;
    }

    ~DummyMemoryPool() {
      deleteNodeRecurse(root_);
    }

    se::Node<T>* root() { return root_; };

    void reserveNodes(const size_t n) { };
    void reserveBlocks(const size_t n) { };

    se::Node<T>*       acquireNode()  { nodes_updated_ = false; return new se::Node<T>; };
    se::Node<T>*       acquireNode(typename T::VoxelData init_value)  { nodes_updated_ = false; return new se::Node<T>(init_value); };
    se::VoxelBlock<T>* acquireBlock() { blocks_updated_ = false; return new se::VoxelBlock<T>; };
    se::VoxelBlock<T>* acquireBlock(typename T::VoxelData init_value) { blocks_updated_ = false; return new se::VoxelBlock<T>(init_value); };

    void deleteNode(se::Node<T>* node, size_t max_level) {
      nodes_updated_ = false;
      const unsigned int id = se::child_id(node->code_,
                                           se::keyops::level(node->code_), max_level);
      node->parent()->child(id) = NULL;
      node->parent()->children_mask_ = node->parent()->children_mask_ & ~(1 << id);

      for (int i = 0; i < 8; i++)
        deleteNodeRecurse(node->child(i));
      delete(node);
    }

    void deleteNodeRecurse(se::Node<T>* node) {
      if (!node) {
        return;
      }
      if (node->isLeaf()) {
        deleteBlockRecurse(dynamic_cast<se::VoxelBlock<T>*>(node));
      } else {
        for (int i = 0; i < 8; i++) {
          deleteNodeRecurse(node->child(i));
        }
        delete(node);
      }
    }

    void deleteBlock(se::VoxelBlock<T>* block, size_t max_level) {
      blocks_updated_ = false;
      const unsigned int id = se::child_id(block->code_,
                                           se::keyops::level(block->code_), max_level);
      block->parent()->child(id) = NULL;
      block->parent()->children_mask_ = block->parent()->children_mask_ & ~(1 << id);
      delete(block);
    }

    void deleteBlockRecurse(se::VoxelBlock<T>* block) {
      blocks_updated_ = false;
      delete(block);
    }

    std::vector<se::Node<T>*>& nodeBuffer() {
      if (!nodes_updated_)
        updateBuffer();
      return node_buffer_;
    };

    const std::vector<se::Node<T>*>& nodeBuffer() const{
      if (!nodes_updated_)
        updateBuffer();
      return node_buffer_;
    };

    std::vector<se::VoxelBlock<T>*>& blockBuffer() {
      if (!blocks_updated_)
        updateBuffer();
      return block_buffer_;
    };

    const std::vector<se::VoxelBlock<T>*>& blockBuffer() const {
      if (!blocks_updated_)
        updateBuffer();
      return block_buffer_;
    };

    size_t nodeBufferSize()  {
      if (!nodes_updated_)
        updateBuffer();
      return node_buffer_.size();
    }

    size_t blockBufferSize() {
      if (!blocks_updated_)
        updateBuffer();
      return block_buffer_.size();
    }

  private:
    se::Node<T>* root_;
    bool nodes_updated_;
    bool blocks_updated_;
    std::vector<Node<T>*>       node_buffer_;
    std::vector<VoxelBlock<T>*> block_buffer_;

    void updateBuffer() {
      node_buffer_.clear();
      if (!blocks_updated_) {
        block_buffer_.clear();
      }
      addNodeRecurse(root_);
      nodes_updated_ = true;
      blocks_updated_ = true;
    }

    void addNodeRecurse(se::Node<T>* node) {
      node_buffer_.push_back(node);
      for (int i = 0; i < 8; i++) {
        if (node->child(i)) {
          if (node->child(i)->isLeaf()) {
            if (!blocks_updated_) {
              block_buffer_.push_back(static_cast<VoxelBlock<T>*>(node->child(i)));
            }
          } else {
            addNodeRecurse(node->child(i));
          }
        }
      }
    }

    // Disabling copy-constructor
    DummyMemoryPool(const DummyMemoryPool& m);
  };

  template <typename BufferType>
  class MemoryBuffer {
  public:
    MemoryBuffer(){
      current_index_ = 0;
      num_pages_ = 0;
      reserved_ = 0;
    }

    ~MemoryBuffer(){
      for(auto&& i : pages_){
        delete [] i;
      }
    }

    size_t size() const { return current_index_; };

    BufferType* operator[](const size_t i) const {
      const int page_idx = i / pagesize_;
      const int ptr_idx = i % pagesize_;
      return pages_[page_idx] + (ptr_idx);
    }

    void reserve(const size_t n){
      bool requires_realloc = (current_index_ + n) > reserved_;
      if(requires_realloc) expand(n);
    }

    BufferType * acquire(){
      // Fetch-add returns the value before increment
      int current = current_index_.fetch_add(1);
      const int page_idx = current / pagesize_;
      const int ptr_idx = current % pagesize_;
      BufferType * ptr = pages_[page_idx] + (ptr_idx);
      return ptr;
    }

  private:
    size_t reserved_;
    std::atomic<unsigned int> current_index_;
    const int pagesize_ = 1024; // # of blocks per page
    int num_pages_;
    std::vector<BufferType *> pages_;

    void expand(const size_t n){

      // std::cout << "Allocating " << n << " blocks" << std::endl;
      const int new_pages = std::ceil(n/pagesize_);
      for(int p = 0; p <= new_pages; ++p){
        pages_.push_back(new BufferType[pagesize_]);
        ++num_pages_;
        reserved_ += pagesize_;
      }
      // std::cout << "Reserved " << reserved_ << " blocks" << std::endl;
    }

    // Disabling copy-constructor
    MemoryBuffer(const MemoryBuffer& m);
  };

  template <typename T>
  class MemoryPool {
  public:
    MemoryPool() {
      node_buffer_.reserve(1);
      root_ = node_buffer_.acquire();
    }

    se::Node<T>* root() { return root_; };

    void reserveNodes(const size_t n) { node_buffer_.reserve(n); };
    void reserveBlocks(const size_t n) { block_buffer_.reserve(n); };

    se::Node<T>*       acquireNode()  { return node_buffer_.acquire(); };
    se::VoxelBlock<T>* acquireBlock() { return block_buffer_.acquire(); };

    se::MemoryBuffer<se::Node<T>>&       nodeBuffer()  { return node_buffer_; };
    se::MemoryBuffer<se::VoxelBlock<T>>& blockBuffer() { return block_buffer_; };

    const se::MemoryBuffer<se::Node<T>>&       nodeBuffer() const { return node_buffer_; };
    const se::MemoryBuffer<se::VoxelBlock<T>>& blockBuffer() const { return block_buffer_; };

    size_t nodeBufferSize()  { return node_buffer_.size();}
    size_t blockBufferSize() { return block_buffer_.size();}

  private:
    se::Node<T>* root_;
    se::MemoryBuffer<se::Node<T>>       node_buffer_;
    se::MemoryBuffer<se::VoxelBlock<T>> block_buffer_;

    // Disabling copy-constructor
    MemoryPool(const MemoryPool& m);
  };
}
#endif
