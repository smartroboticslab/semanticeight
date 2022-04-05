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

#ifndef NODE_H
#define NODE_H

#include <atomic>
#include <time.h>
#include <vector>

#include "io/se_serialise.hpp"
#include "octree_defines.h"
#include "utils/math_utils.h"

namespace se {

static inline Eigen::Vector3f get_sample_coord(const Eigen::Vector3i& octant_coord,
                                               const int octant_size,
                                               const Eigen::Vector3f& sample_offset_frac)
{
    return octant_coord.cast<float>() + sample_offset_frac * octant_size;
}

/*! \brief A non-leaf node of the Octree. Each Node has 8 children.
 */
template<typename T>
class Node {
    public:
    typedef typename T::VoxelData VoxelData;

    Node(const typename T::VoxelData init_data = T::initData());

    Node(const Node<T>& node);

    void operator=(const Node<T>& node);

    virtual ~Node(){};

    const VoxelData& data() const;

    VoxelData* childrenData()
    {
        return children_data_;
    }
    const VoxelData* childrenData() const
    {
        return children_data_;
    }

    VoxelData& childData(const int child_idx)
    {
        return children_data_[child_idx];
    }
    const VoxelData& childData(const int child_idx) const
    {
        return children_data_[child_idx];
    }

    void childData(const int child_idx, const VoxelData& child_data)
    {
        children_data_[child_idx] = child_data;
    };

    Node*& child(const int x, const int y, const int z)
    {
        return children_ptr_[x + y * 2 + z * 4];
    };
    const Node* child(const int x, const int y, const int z) const
    {
        return children_ptr_[x + y * 2 + z * 4];
    };
    Node*& child(const int child_idx)
    {
        return children_ptr_[child_idx];
    }
    const Node* child(const int child_idx) const
    {
        return children_ptr_[child_idx];
    }

    Node*& parent()
    {
        return parent_ptr_;
    }
    const Node* parent() const
    {
        return parent_ptr_;
    }

    void code(key_t code)
    {
        code_ = code;
    }
    key_t code() const
    {
        return code_;
    }

    key_t childCode(const int child_idx, const int voxel_depth) const;

    void size(int size)
    {
        size_ = size;
    }
    int size() const
    {
        return size_;
    }

    Eigen::Vector3i coordinates() const;
    Eigen::Vector3i centreCoordinates() const;

    Eigen::Vector3i childCoord(const int child_idx) const;
    Eigen::Vector3i childCentreCoord(const int child_idx) const;

    void children_mask(const unsigned char cm)
    {
        children_mask_ = cm;
    }
    unsigned char children_mask() const
    {
        return children_mask_;
    }

    void timestamp(const unsigned int t)
    {
        timestamp_ = t;
    }
    unsigned int timestamp() const
    {
        return timestamp_;
    }

    void active(const bool a)
    {
        active_ = a;
    }
    bool active() const
    {
        return active_;
    }

    virtual bool isBlock() const
    {
        return false;
    }

    protected:
    VoxelData children_data_[8];
    Node* children_ptr_[8];
    Node* parent_ptr_;
    key_t code_;
    unsigned int size_;
    unsigned char children_mask_;
    unsigned int timestamp_;
    bool active_;

    private:
    // Internal copy helper function
    void initFromNode(const Node<T>& node);
    friend std::ofstream& internal::serialise<>(std::ofstream& out, Node& node);
    friend void internal::deserialise<>(Node& node, std::ifstream& in);
};

/*! \brief A leaf node of the Octree. Each VoxelBlock contains compute_num_voxels() voxels
 * voxels.
 */
template<typename T>
class VoxelBlock : public Node<T> {
    public:
    using VoxelData = typename T::VoxelData;

    static constexpr unsigned int size_li = BLOCK_SIZE;
    static constexpr unsigned int size_sq = se::math::sq(size_li);
    static constexpr unsigned int size_cu = se::math::cu(size_li);
    static constexpr int max_scale = se::math::log2_const(size_li);
    static constexpr int num_scales = max_scale + 1;

    VoxelBlock(const int current_scale, const int min_scale);

    VoxelBlock(const VoxelBlock<T>& block);

    void operator=(const VoxelBlock<T>& block);

    virtual ~VoxelBlock(){};

    bool isBlock() const
    {
        return true;
    }

    Eigen::Vector3i coordinates() const
    {
        return coordinates_;
    }
    void coordinates(const Eigen::Vector3i& block_coord)
    {
        coordinates_ = block_coord;
    }

    Eigen::Vector3i voxelCoordinates(const int voxel_idx) const;
    Eigen::Vector3i voxelCoordinates(const int voxel_idx, const int scale) const;

    int current_scale() const
    {
        return current_scale_;
    }
    void current_scale(const int s)
    {
        current_scale_ = s;
    }

    int min_scale() const
    {
        return min_scale_;
    }
    void min_scale(const int s)
    {
        min_scale_ = s;
        this->updateMinScaleReached();
    }
    int8_t minScaleReached() const
    {
        return min_scale_reached_;
    }

    float minDistUpdated() const
    {
        return min_dist_updated_;
    }

    void minDistUpdated(float dist)
    {
        if (dist < min_dist_updated_ && dist >= 0.0f) {
            min_dist_updated_ = dist;
        }
    }

    bool contains(const Eigen::Vector3i& voxel_coord) const;

    virtual VoxelData data(const Eigen::Vector3i& voxel_coord) const = 0;
    virtual void setData(const Eigen::Vector3i& voxel_coord, const VoxelData& voxel_data) = 0;

    virtual VoxelData data(const Eigen::Vector3i& voxel_coord, const int scale) const = 0;
    virtual void
    setData(const Eigen::Vector3i& voxel_coord, const int scale, const VoxelData& voxel_data) = 0;

    virtual VoxelData data(const int voxel_idx) const = 0;
    virtual void setData(const int voxel_idx, const VoxelData& voxel_data) = 0;

    virtual VoxelData data(const int voxel_idx_at_scale, const int scale) const = 0;
    virtual void
    setData(const int voxel_idx_at_scale, const int scale, const VoxelData& voxel_data) = 0;

    virtual VoxelData maxData(const Eigen::Vector3i& voxel_coord) const = 0;
    virtual VoxelData maxData(const Eigen::Vector3i& voxel_coord, const int scale) const = 0;
    virtual VoxelData maxData(const int voxel_idx) const = 0;
    virtual VoxelData maxData(const int voxel_idx_at_scale, const int scale) const = 0;

    /*! \brief The number of voxels per side at scale.
   */
    static constexpr int scaleSize(const int scale);
    /*! \brief The side length of a voxel at scale expressed in primitive voxels.
   * This is e.g. 1 for scale 0 and 8 for scale 3.
   */
    static constexpr int scaleVoxelSize(const int scale);
    /*! \brief The total number of voxels contained in scale.
   * This is equivalent to scaleSize()^3.
   */
    static constexpr int scaleNumVoxels(const int scale);
    /*! \brief The offset needed to get to the first voxel of scale when using
   * a linear index.
   */
    static constexpr int scaleOffset(const int scale);

    protected:
    Eigen::Vector3i coordinates_;
    int current_scale_;
    int min_scale_;
    int8_t min_scale_reached_;
    float min_dist_updated_ = INFINITY;

    void updateMinScaleReached();

    private:
    // Internal copy helper function
    void initFromBlock(const VoxelBlock<T>& block);
};

/*! \brief A leaf node of the Octree. Each VoxelBlock contains compute_num_voxels() voxels
 * voxels.
 */
template<typename T>
class VoxelBlockFinest : public VoxelBlock<T> {
    public:
    using VoxelData = typename VoxelBlock<T>::VoxelData;

    VoxelBlockFinest(const typename T::VoxelData init_data = T::initData());

    VoxelBlockFinest(const VoxelBlockFinest<T>& block);

    void operator=(const VoxelBlockFinest<T>& block);

    virtual ~VoxelBlockFinest(){};

    VoxelData data(const Eigen::Vector3i& voxel_coord) const;
    void setData(const Eigen::Vector3i& voxel_coord, const VoxelData& voxel_data);

    /**
   * \note Data will always retrieved and set at scale 0.
   *       Function only exist to keep API consistent.
   */
    VoxelData data(const Eigen::Vector3i& voxel_coord, const int scale_0) const;
    void
    setData(const Eigen::Vector3i& voxel_coord, const int scale_0, const VoxelData& voxel_data);

    VoxelData data(const int voxel_idx) const;
    void setData(const int voxel_idx, const VoxelData& voxel_data);

    /**
   * \warning This functions should not be used with VoxelBlockFinest and will always just access at scale 0.
   *
   */
    VoxelData data(const int voxel_idx_at_scale_0, const int scale_0) const;
    void setData(const int voxel_idx_at_scale_0, const int scale_0, const VoxelData& voxel_data);

    /**
   * \note Data will always retrieved and set at scale 0.
   *       Function only exist to keep API consistent.
   */
    VoxelData maxData(const Eigen::Vector3i& voxel_coord) const
    {
        return data(voxel_coord);
    };
    VoxelData maxData(const Eigen::Vector3i& voxel_coord, const int /* scale_0 */) const
    {
        return data(voxel_coord);
    };
    VoxelData maxData(const int voxel_idx) const
    {
        return data(voxel_idx);
    };
    VoxelData maxData(const int voxel_idx, const int /* scale_0 */) const
    {
        return data(voxel_idx);
    };

    VoxelData* blockData()
    {
        return block_data_;
    }
    const VoxelData* blockData() const
    {
        return block_data_;
    }
    static constexpr int data_size()
    {
        return sizeof(VoxelBlockFinest<T>);
    }

    private:
    // Internal copy helper function
    void initFromBlock(const VoxelBlockFinest<T>& block);

    static constexpr size_t num_voxels_in_block = VoxelBlock<T>::size_cu;
    VoxelData block_data_[num_voxels_in_block]; // Brick of data.

    friend std::ofstream& internal::serialise<>(std::ofstream& out, VoxelBlockFinest& node);
    friend void internal::deserialise<>(VoxelBlockFinest& node, std::ifstream& in);
};



/*! \brief A leaf node of the Octree. Each VoxelBlock contains compute_num_voxels() voxels
 * voxels.
 */
template<typename T>
class VoxelBlockFull : public VoxelBlock<T> {
    public:
    using VoxelData = typename VoxelBlock<T>::VoxelData;

    VoxelBlockFull(const typename T::VoxelData init_data = T::initData());

    VoxelBlockFull(const VoxelBlockFull<T>& block);

    void operator=(const VoxelBlockFull<T>& block);

    virtual ~VoxelBlockFull(){};

    VoxelData data(const Eigen::Vector3i& voxel_coord) const;
    void setData(const Eigen::Vector3i& voxel_coord, const VoxelData& voxel_data);

    VoxelData data(const Eigen::Vector3i& voxel_coord, const int scale) const;
    void setData(const Eigen::Vector3i& voxel_coord, const int scale, const VoxelData& voxel_data);

    VoxelData data(const int voxel_idx) const;
    void setData(const int voxel_idx, const VoxelData& voxel_data);

    VoxelData data(const int voxel_idx_at_scale, const int scale) const;
    void setData(const int voxel_idx_at_scale, const int scale, const VoxelData& voxel_data);

    /// \warning These functions only return the regular data.
    VoxelData maxData(const Eigen::Vector3i& voxel_coord) const
    {
        return data(voxel_coord);
    };
    VoxelData maxData(const Eigen::Vector3i& voxel_coord, const int scale) const
    {
        return data(voxel_coord, scale);
    };
    VoxelData maxData(const int voxel_idx) const
    {
        return data(voxel_idx);
    };
    VoxelData maxData(const int voxel_idx_at_scale, const int scale) const
    {
        return data(voxel_idx_at_scale, scale);
    };

    VoxelData* blockData()
    {
        return block_data_;
    }
    const VoxelData* blockData() const
    {
        return block_data_;
    }
    static constexpr int data_size()
    {
        return sizeof(VoxelBlockFull<T>);
    }

    private:
    // Internal copy helper function
    void initFromBlock(const VoxelBlockFull<T>& block);

    static constexpr size_t compute_num_voxels()
    {
        size_t voxel_count = 0;
        unsigned int size_at_scale = VoxelBlock<T>::size_li;
        while (size_at_scale >= 1) {
            voxel_count += size_at_scale * size_at_scale * size_at_scale;
            size_at_scale = size_at_scale >> 1;
        }
        return voxel_count;
    }

    static constexpr size_t num_voxels_in_block = compute_num_voxels();
    VoxelData block_data_[num_voxels_in_block]; // Brick of data.

    friend std::ofstream& internal::serialise<>(std::ofstream& out, VoxelBlockFull& node);
    friend void internal::deserialise<>(VoxelBlockFull& node, std::ifstream& in);
};



/*! \brief A leaf node of the Octree. Each VoxelBlock contains compute_num_voxels_in_block() voxels
 * voxels.
 */
template<typename T>
class VoxelBlockSingle : public VoxelBlock<T> {
    public:
    using VoxelData = typename VoxelBlock<T>::VoxelData;

    VoxelBlockSingle(const typename T::VoxelData init_data = T::initData());

    VoxelBlockSingle(const VoxelBlockSingle<T>& block);

    void operator=(const VoxelBlockSingle<T>& block);

    ~VoxelBlockSingle();

    VoxelData initData() const;
    void setInitData(const VoxelData& init_data);

    VoxelData data(const Eigen::Vector3i& voxel_coord) const;
    void setData(const Eigen::Vector3i& voxel_coord, const VoxelData& voxel_data);
    void setDataSafe(const Eigen::Vector3i& voxel_coord, const VoxelData& voxel_data);

    VoxelData data(const Eigen::Vector3i& voxel_coord, const int scale) const;
    void setData(const Eigen::Vector3i& voxel_coord, const int scale, const VoxelData& voxel_data);
    void
    setDataSafe(const Eigen::Vector3i& voxel_coord, const int scale, const VoxelData& voxel_data);

    VoxelData data(const int voxel_idx) const;
    void setData(const int voxel_idx, const VoxelData& voxel_data);
    void setDataSafe(const int voxel_idx, const VoxelData& voxel_data);

    VoxelData data(const int voxel_idx_at_scale, const int scale) const;
    void setData(const int voxel_idx_at_scale, const int scale, const VoxelData& voxel_data);
    void setDataSafe(const int voxel_idx_at_scale, const int scale, const VoxelData& voxel_data);

    /// \warning These functions only return the regular data.
    VoxelData maxData(const Eigen::Vector3i& voxel_coord) const
    {
        return data(voxel_coord);
    };
    VoxelData maxData(const Eigen::Vector3i& voxel_coord, const int scale) const
    {
        return data(voxel_coord, scale);
    };
    VoxelData maxData(const int voxel_idx) const
    {
        return data(voxel_idx);
    };
    VoxelData maxData(const int voxel_idx_at_scale, const int scale) const
    {
        return data(voxel_idx_at_scale, scale);
    };

    void allocateDownTo();
    void allocateDownTo(const int scale);

    void deleteUpTo(const int scale);

    std::vector<VoxelData*>& blockData()
    {
        return block_data_;
    }
    const std::vector<VoxelData*>& blockData() const
    {
        return block_data_;
    }
    static constexpr int data_size()
    {
        return sizeof(VoxelBlock<T>);
    }

    private:
    // Internal copy helper function
    void initFromBlock(const VoxelBlockSingle<T>& block);
    void initialiseData(VoxelData* voxel_data, const int num_voxels);
    std::vector<VoxelData*>
        block_data_; // block_data_[0] returns the data at scale = max_scale and not scale = 0
    VoxelData init_data_;

    friend std::ofstream& internal::serialise<>(std::ofstream& out, VoxelBlockSingle& node);
    friend void internal::deserialise<>(VoxelBlockSingle& node, std::ifstream& in);
};


/*! \brief A leaf node of the Octree. Each VoxelBlock contains compute_num_voxels_in_block() voxels
* voxels.
*/
template<typename T>
class VoxelBlockSingleMax : public VoxelBlock<T> {
    public:
    using VoxelData = typename VoxelBlock<T>::VoxelData;

    VoxelBlockSingleMax(const typename T::VoxelData init_data = T::initData());

    VoxelBlockSingleMax(const VoxelBlockSingleMax<T>& block);

    void operator=(const VoxelBlockSingleMax<T>& block);

    ~VoxelBlockSingleMax();

    VoxelData initData() const;
    void setInitData(const VoxelData& init_data);

    VoxelData data(const Eigen::Vector3i& voxel_coord) const;
    void setData(const Eigen::Vector3i& voxel_coord, const VoxelData& voxel_data);
    void setDataSafe(const Eigen::Vector3i& voxel_coord, const VoxelData& voxel_data);

    VoxelData data(const Eigen::Vector3i& voxel_coord, const int scale) const;
    void setData(const Eigen::Vector3i& voxel_coord, const int scale, const VoxelData& voxel_data);
    void
    setDataSafe(const Eigen::Vector3i& voxel_coord, const int scale, const VoxelData& voxel_data);

    VoxelData data(const int voxel_idx) const;
    void setData(const int voxel_idx, const VoxelData& voxel_data);
    void setDataSafe(const int voxel_idx, const VoxelData& voxel_data);

    VoxelData data(const int voxel_idx_at_scale, const int scale) const;
    void setData(const int voxel_idx_at_scale, const int scale, const VoxelData& voxel_data);
    void setDataSafe(const int voxel_idx_at_scale, const int scale, const VoxelData& voxel_data);

    VoxelData maxData(const Eigen::Vector3i& voxel_coord) const;
    void setMaxData(const Eigen::Vector3i& voxel_coord, const VoxelData& voxel_data);
    void setMaxDataSafe(const Eigen::Vector3i& voxel_coord, const VoxelData& voxel_data);

    VoxelData maxData(const Eigen::Vector3i& voxel_coord, const int scale) const;
    void
    setMaxData(const Eigen::Vector3i& voxel_coord, const int scale, const VoxelData& voxel_data);
    void setMaxDataSafe(const Eigen::Vector3i& voxel_coord,
                        const int scale,
                        const VoxelData& voxel_data);

    VoxelData maxData(const int voxel_idx) const;
    void setMaxData(const int voxel_idx, const VoxelData& voxel_data);
    void setMaxDataSafe(const int voxel_idx, const VoxelData& voxel_data);

    VoxelData maxData(const int voxel_idx_at_scale, const int scale) const;
    void setMaxData(const int voxel_idx_at_scale, const int scale, const VoxelData& voxel_data);
    void setMaxDataSafe(const int voxel_idx_at_scale, const int scale, const VoxelData& voxel_data);

    void allocateDownTo();
    void allocateDownTo(const int scale);

    void deleteUpTo(const int scale);

    VoxelData meanData()
    {
        return block_max_data_[0][0];
    }
    VoxelData maxData()
    {
        return block_max_data_[0][0];
    }

    decltype(T::selectVoxelValue(T::initData())) meanValue()
    {
        return T::selectVoxelValue(block_data_[0][0]);
    }
    decltype(T::selectVoxelValue(T::initData())) maxValue()
    {
        return T::selectVoxelValue(block_max_data_[0][0]);
    }

    std::vector<VoxelData*>& blockData()
    {
        return block_data_;
    }
    const std::vector<VoxelData*>& blockData() const
    {
        return block_data_;
    }
    std::vector<VoxelData*>& blockMaxData()
    {
        return block_max_data_;
    }
    const std::vector<VoxelData*>& blockMaxData() const
    {
        return block_max_data_;
    }
    static constexpr int data_size()
    {
        return sizeof(VoxelBlock<T>);
    }

    const size_t& currIntegrCount() const
    {
        return curr_integr_count_;
    } ///< \brief Get the number of integrations at the current scale.
    const size_t& currObservedCount() const
    {
        return curr_observed_count_;
    } ///< \brief Get the number of observed voxels at the current scale.

    void incrCurrIntegrCount()
    {
        curr_integr_count_++;
    } ///< \brief Increment the number of integrations at the current scale by 1.

    /**
   * \brief Increment the number of observed voxels in at the current scale by 1.
   *
   * \param[in] do_increment The optional flag indicating if the counter should be incremented.
   */
    void incrCurrObservedCount(bool do_increment = true);

    /**
   * \brief Reset the current integration and observation count to 0.
   */
    void resetCurrCount();

    /**
   * \brief When a block is initialised from an observed block (i.e. init_data_.observed == true), set the current
   *        observed count to all voxels observed and the integration count to the nodes value. Otherwise reset the current
   *        count.
   */
    void initCurrCout();

    /**
   * \return The integration scale of the buffer.
   */
    const int& buffer_scale() const
    {
        return buffer_scale_;
    }
    const size_t& bufferIntegrCount() const
    {
        return buffer_integr_count_;
    }
    const size_t& bufferObservedCount() const
    {
        return buffer_observed_count_;
    }

    /**
   * \brief Increment the buffer count if incrementation criterion is met.
   *        I.e. the scale normalised number of observations at the buffer scale >= 95% observations at the current scale.
   */
    void incrBufferIntegrCount(const bool do_increment = true);

    /**
   * \brief Increment the number of observed voxels at the buffers scale by 1.
   *
   * \param[in] do_increment The optional flag indicating if the counter should be incremented.
   */
    void incrBufferObservedCount(const bool do_increment = true);

    /**
   * \brief Reset the buffer integration and observation count to 0.
   */
    void resetBufferCount();

    /**
   *  \brief Reset buffer variables to the initial values and free the buffer data if applicable.
   */
    void resetBuffer();

    /**
   * \brief Init buffer variables.
   *
   * \param[in] buffer_scale The scale the buffer should be initialised at.
   */
    void initBuffer(const int buffer_scale);

    /**
   * \brief Check if the scale should be switched from the current scale to the recommended.
   *
   * \return True is data is switched to recommended scale.
   */
    bool switchData();

    /**
   * \brief Get a `const` reference to the voxel data in the buffer at the voxel coordinates.
   *
   * \param[in] voxel_coord The voxel coordinates of the data to be accessed.
   *
   * \warning The function does not not check if the voxel_idx exceeds the array size.
   *
   * \return `const` reference to the voxel data in the buffer for the provided voxel coordinates.
   */
    VoxelData& bufferData(const Eigen::Vector3i& voxel_coord) const;

    /**
   * \brief Get a reference to the voxel data in the buffer at the voxel coordinates.
   *
   * \param[in] voxel_coord The voxel coordinates of the data to be accessed.
   *
   * \warning The function does not not check if the voxel_idx exceeds the array size.
   *
   * \return Reference to the voxel data in the buffer for the provided voxel coordinates.
   */
    VoxelData& bufferData(const Eigen::Vector3i& voxel_coord);

    /**
   * \brief Get a `const` reference to the voxel data in the buffer at the voxel index.
   *
   * \param[in] voxel_idx The voxel index of the data to be accessed.
   *
   * \warning The function does not not check if the voxel_idx exceeds the array size.
   *
   * \return `const` reference to the voxel data in the buffer for the provided voxel index.
   */
    VoxelData& bufferData(const int voxel_idx) const
    {
        return buffer_data_[voxel_idx];
    }

    /**
   * \brief Get a reference to the voxel data in the buffer at the voxel index.
   *
   * \param[in] voxel_idx The voxel index of the data to be accessed.
   *
   * \warning The function does not not check if the voxel_idx exceeds the array size.
   *
   * \return Reference to the voxel data in the buffer for the provided voxel index.
   */
    VoxelData& bufferData(const int voxel_idx)
    {
        return buffer_data_[voxel_idx];
    }

    /**
   * \brief Get a `const` reference to the mean voxel data at the current scale via the voxel index.
   *
   * \param[in] voxel_idx The voxel index of the data to be accessed.
   *
   * \warning The function does not not check if the voxel_idx exceeds the array size.
   *
   * \return `const` reference to the voxel data in the buffer for the provided voxel index.
   */
    VoxelData& currData(const int voxel_idx) const
    {
        return curr_data_[voxel_idx];
    }

    /**
   * \brief Get a reference to the mean voxel data at the current scale via the voxel index.
   *
   * \param[in] voxel_idx The voxel index of the data to be accessed.
   *
   * \warning The function does not not check if the voxel_idx exceeds the array size.
   *
   * \return Reference to the mean voxel data at the current scale for the provided voxel index.
   */
    VoxelData& currData(const int voxel_idx)
    {
        return curr_data_[voxel_idx];
    }

    /**
   * \brief Get a pointer to the mean block data array at a given scale.
   *
   * \param[in] scale The scale to return the mean block data array from.
   *
   * \return The pointer to the mean block data array at the provided scale.
   *         Returns a nullptr if the scale smaller than the min allocated scale.
   */
    VoxelData* blockDataAtScale(const int scale);

    /**
   * \brief Get a pointer to the max block data array at a given scale.
   *
   * \param[in] scale The scale to return the max block data array from.
   *
   * \return The pointer to the max block data array at the provided scale.
   *         Returns a nullptr if the scale smaller than the min allocated scale.
   */
    VoxelData* blockMaxDataAtScale(const int scale);

    private:
    // Internal copy helper function
    void initFromBlock(const VoxelBlockSingleMax<T>& block);
    void initialiseData(VoxelData* voxel_data,
                        const int num_voxels); ///<< Initalise array of data with `init_data_`.


    /// \note the block_data_ and block_max_data_ point to the same data at the finest scale as they are equivalent.
    std::vector<VoxelData*> block_data_; ///<< Vector containing the mean data at block scales.
        ///< \note block_data_[0] returns the data at scale = max_scale and not scale = 0
    std::vector<VoxelData*> block_max_data_; ///<< Vector containing the max data at block scales.
        ///< \note block_data_[0] returns the data at scale = max_scale and not scale = 0

    VoxelData* curr_data_ = nullptr; ///<< Pointer to the data at the current integration scale.
    size_t curr_integr_count_;       ///<< Number of integrations at that current scale.
    size_t curr_observed_count_;     ///<< Number of observed voxels at the current scale

    /**
   * \brief Rather than switching directly to a different integration scale once the integration scale computation
   *        recommends a different scale, data is continued to be integrated at the current scale and additionally into
   *        a buffer at the recommended scale.
   *        Recommended scale == current scale:
   *            The buffer_data_ points to a `nullptr`
   *        Recommended scale < current scale:
   *            The buffer_data_ points to a independently allocated array of voxel data. The data is initialised with
   *            the parent data at the current integration scale. Once the scale changes the data is inserted into the
   *            block_data_ and block_max_data_ vector.
   *        Recommended scale > current scale:
   *            The buffer_data_ points to according scale in the block_data_ vector. The data integration starts from
   *            the mean up-propagated value. Up until the recommened scale > current scale the mean up-propagation starts
   *            from the recommened scale such that the data is not overwritten by the up-propagation from the current scale.
   *            However the max up-propagation continues from the current integration scale. Once the scale changes the
   *            current_data_ and current_scale_ is set to the buffer setup, the finest scale in the block_data_ and
   *            block_max_data_ is deleted and the new finest scales in the buffers adjusted accordingly.
   *
   * \note  The recommended scale can only differ by +/-1 scale from the current scale.
   *        The overhead of integrating at two different scales is insignificant compared to switching immediately as
   *        the double integration only happens in areas where the recommended integration scale changed and stops
   *        as soon as the criteria for switching to the finer or coarser scale.
   */
    VoxelData* buffer_data_ = nullptr; ///<< Pointer to the buffer data.
    int buffer_scale_;                 ///<< The scale of the buffer.
    size_t
        buffer_integr_count_; ///<< Number of integrations at the buffer scale. \note Is only incremented when 95% of the current observations are reached.
    size_t buffer_observed_count_; ///<< Number of observed voxels in the buffer.

    VoxelData init_data_; ///<< The value the block data is initalised with.

    friend std::ofstream& internal::serialise<>(std::ofstream& out, VoxelBlockSingleMax& node);
    friend void internal::deserialise<>(VoxelBlockSingleMax& node, std::ifstream& in);
};

} // namespace se

#include "node_impl.hpp"

#endif
