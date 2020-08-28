/*
 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#ifndef PERFSTATS_H
#define PERFSTATS_H

#ifdef __APPLE__
    #include <mach/clock.h>
    #include <mach/mach.h>
#endif


#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include <ctime>
#include <iomanip>
#include <time.h>
#include <math.h>
#include <mutex>

struct PerfStats {
	enum Type {
    COUNT,
    CURRENT,
    DISTANCE,
    DOUBLE,
    ENERGY,
    FRAME,
    FREQUENCY,
    INT,
    PERCENTAGE,
    POWER,
    TIME,
    UNDEFINED,
    VOLTAGE
  };

	struct Stats {
		double sum() const {
			return std::accumulate(data_.begin(), data_.end(), 0.0);
		}

    double average() const {
			return sum() / std::max(data_.size(), size_t(1));
		}

    double max() const {
			return *std::max_element(data_.begin(), data_.end());
		}

    double min() const {
			return *std::min_element(data_.begin(), data_.end());
		}

    std::vector<double> data_;
    double last_absolute_;
    double last_period_;
    std::mutex mutex_;
    Type type_;
  };

	struct Results {
		double mean;
		double sd;
		double min;
		double max;
		double sum;
	};

  PerfStats() { insertion_id_ = 0; }

  void debug();

  const Stats& get(const std::string& key) const { return stats_.find(key)->second; }

	double getLastData(const std::string& key);

  double getSampleTime(const std::string& key);

  double getTime();

	Type getType(const std::string& key);

	void print(std::ostream& out = std::cout) const;

  void reset(void) { stats_.clear(); }

  void reset(const std::string& key);

	void printAllData(std::ostream& out, bool include_all_data = true) const;

  double sample(const std::string& key);

  double sample(const std::string& key,
                double             t,
                Type               type = COUNT);

  double start(void);

  int insertion_id_;
  double last_;
  std::map<int, std::string> order_;
  std::map<std::string, Stats> stats_;
};

#include "perfstats_impl.hpp"

extern PerfStats stats;

#endif // PERFSTATS_H
