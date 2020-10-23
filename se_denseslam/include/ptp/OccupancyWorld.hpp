/**
 * Probabilistic Trajectory Planning, Map interface for supereight library of Emanuele Vespa.
 *
 * Copyright (C) 2018 Imperial College London.
 * Copyright (C) 2018 ETH Zürich.
 *
 * @file OccupancyWorld.hpp
 *
 *
 * @author Nils Funk
 * @date July 5, 2018
 */

#ifndef OCCUPANCYWORLD_HPP
#define OCCUPANCYWORLD_HPP

#include <fstream>
#include <iostream>
#include <string.h>
#include <map>
#include <memory>
#include <bitset>
#include <eigen3/Eigen/Dense>
#include <se/octree.hpp>
#include <se/node_iterator.hpp>
#include <se/node.hpp>
#include "se/voxel_implementations.hpp"
#include <octomap/octomap.h>
#include <boost/foreach.hpp>
#include <boost/range/combine.hpp>
#include <ptp/Header.hpp>
#include <ptp/Path.hpp>
#include <ptp/common.hpp>

namespace ptp {
  class OccupancyWorld {
  public:
    typedef std::shared_ptr<OccupancyWorld> Ptr;

    OccupancyWorld();
    ~OccupancyWorld(){}

    class Color {
    public:
        Color() : r(255), g(255), b(255) {}
        Color(uint8_t _r, uint8_t _g, uint8_t _b)
                : r(_r), g(_g), b(_b) {}
        inline bool operator== (const Color &other) const {
          return (r==other.r && g==other.g && b==other.b);
        }
        inline bool operator!= (const Color &other) const {
          return (r!=other.r || g!=other.g || b!=other.b);
        }
        uint8_t r, g, b;
    };

    /**
     * Supereight I/O:
     * - loading and saving maps with or without multilevel resolution
     * - set and get octree
     * - get octree root
     * - get voxel occupancy
     */
    void readSupereight(const std::string& filename);
    void setOctree(std::shared_ptr<se::Octree<VoxelImpl::VoxelType>> octree);
    std::shared_ptr<se::Octree<VoxelImpl::VoxelType>> getMap(){return octree_;} //TODO: Change to getOctree()
    bool isFree(const Eigen::Vector3i& voxel_coord, float threshold = 0);
    bool isFreeAtPoint(const Eigen::Vector3f& point_M, float threshold = 0);

    /*
     * OccupancyWold I/O
     */
    se::Node<MultiresOFusion::VoxelType>* getNode(const Eigen::Vector3i& node_coord, const int node_size);
    bool getMapBounds(Eigen::Vector3f& map_bounds_min_v, Eigen::Vector3f& map_bounds_max_v);
    bool getMapBoundsMeter(Eigen::Vector3f& map_bounds_min_m, Eigen::Vector3f& map_bounds_max_m);
    bool inMapBounds(const Eigen::Vector3f& voxel_coord);
    bool inMapBoundsMeter(const Eigen::Vector3f& point_M);
    float getMapResolution();

  private:
    void updateMapBounds();

    Eigen::Vector3f map_bounds_max_ = Eigen::Vector3f(-1,-1,-1);
    Eigen::Vector3f map_bounds_min_ = Eigen::Vector3f(-1,-1,-1);

    float res_;
    std::shared_ptr<se::Octree<MultiresOFusion::VoxelType>> octree_ = nullptr;
  };
} // namespace ptp

#endif //OCCUPANCYWORLD_HPP
