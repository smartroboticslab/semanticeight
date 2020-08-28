#ifndef __PERFSTATS_IMPL_HPP
#define __PERFSTATS_IMPL_HPP



inline double PerfStats::getLastData(const std::string & key) {
  std::map<std::string, Stats>::iterator s = stats_.find(key);
  if(s != stats_.end())
    return (s->second.data_.back());

  return (0);
}



inline double PerfStats::getSampleTime(const std::string & key) {
  std::map<std::string, Stats>::iterator s = stats_.find(key);
  if(s != stats_.end()) {
    return (s->second.last_period_);
  }
  return (0.0);
}



inline double PerfStats::getTime() {
#ifdef __APPLE__
  clock_serv_t cclock;
		mach_timespec_t clockData;
		host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);
		clock_get_time(cclock, &clockData);
		mach_port_deallocate(mach_task_self(), cclock);
#else
  struct timespec clockData;
  clock_gettime(CLOCK_MONOTONIC, &clockData);
#endif
  return (double) clockData.tv_sec + clockData.tv_nsec / 1000000000.0;
}



inline PerfStats::Type PerfStats::getType(const std::string & key) {
  std::map<std::string, Stats>::iterator s = stats_.find(key);
  if(s != stats_.end())
    return (s->second.type_);
  return (UNDEFINED);
}



inline void PerfStats::printAllData(std::ostream& out, bool include_all_data) const {
  struct Results* res = nullptr;
  struct Results* res_ptr = nullptr;
  int frames = 0;
  bool done = false;
  unsigned int idx = 0;

  res_ptr = (struct Results *) malloc(sizeof(struct Results) * stats_.size());
  out.precision(10);
  res = res_ptr;
  //for (std::map<std::string,Stats>::const_iterator it=stats_.begin(); it!=stats_.end(); it++){
  for(std::map<int, std::string>::const_iterator kt = order_.begin();
      kt != order_.end(); kt++) {
    std::map<std::string, Stats>::const_iterator it = stats_.find(
        kt->second);
    if(it == stats_.end())
      continue;

    (*res).mean = 0.0;
    (*res).min = 9e10;
    (*res).max = -9e10;
    (*res).sd = 0.0;
    res++;

  }
  out.precision(10);
  done = false;
  out.setf(std::ios::fixed, std::ios::floatfield);
  //while (!done) {
  res = res_ptr;
  for (std::map<int, std::string>::const_iterator kt = order_.begin();
       kt != order_.end(); kt++) {
    std::map<std::string, Stats>::const_iterator it = stats_.find(
        kt->second);
    if(it == stats_.end())
      continue;

    for(idx = 0; idx < it->second.data_.size(); idx++) {
      (*res).mean = (*res).mean + it->second.data_[idx];
      if(it->second.data_[idx] > (*res).max)
        (*res).max = it->second.data_[idx];
      if(it->second.data_[idx] < (*res).min)
        (*res).min = it->second.data_[idx];
      frames++;
    }
    (*res).sum = res->mean;
    (*res).mean = (*res).mean / idx;
    res++;
  }
//idx++;
  // }

  idx = 0;

  int count =0;

  if(include_all_data)
    std::cerr << " Done max min\n";

  while(count < insertion_id_) {
    res = res_ptr;
    for(std::map<int, std::string>::const_iterator kt = order_.begin();
        kt != order_.end(); kt++) {
      std::map<std::string, Stats>::const_iterator it = stats_.find(
          kt->second);
      if(it == stats_.end()) {
        res++;
        continue;
      }


      if(idx < it->second.data_.size()) {
        if(include_all_data) {
          switch(it->second.type_) {
            case TIME: {
              out << it->second.data_[idx] << "\t";
            }
              break;
            case COUNT: {
              out << it->second.data_[idx] << "\t";
            }
              break;
            case PERCENTAGE: {
              out << it->second.data_[idx] * 100.0 << "\t";
            }
              break;
            case DISTANCE:
            case POWER:
            case ENERGY:
            case DOUBLE: {
              out << it->second.data_[idx] << "\t";
            }
              break;
            case FRAME:
            case INT: {
              out << std::left << std::setw(10)
                  << int(it->second.data_[idx]) << "\t";
            }
            default:
              break;
          }
        }
        //  out << std::setw(10) << it->second.data_[idx] << "\t"
        (*res).sd = (*res).sd
                    + ((it->second.data_[idx] - (*res).mean)
                       * (it->second.data_[idx] - (*res).mean));
      } else {
        if(idx == it->second.data_.size())
          count++;
      }
      res++;
    }
    if(include_all_data && done) {
      out << "# End of file";
    }

    idx++;

    if(include_all_data)
      out << std::endl;
  }

  res = res_ptr;

  int i = 0;
  int max = order_.size();

  for(std::map<int, std::string>::const_iterator kt = order_.begin(); kt != order_.end(); kt++) {
    std::map<std::string, Stats>::const_iterator it = stats_.find(kt->second);
    if(it == stats_.end())
      continue;

    std::cout.precision(10);
    std::cout << "\"" << it->first << "\" : { ";

    std::cout << "\"mean\":\"" << (*res).mean << "\", ";
    std::cout << "\"std\":\"" << sqrt((*res).sd / idx)  << "\", ";
    std::cout << "\"min\":\"" << (*res).min << "\", ";
    std::cout << "\"max\":\"" << (*res).max << "\", ";
    std::cout << "\"sum\":\"" << (*res).sum << "\"";
    std::cout << "}";

    if(i + 1 != max)
    {
      std::cout << ", ";
    }

    ++i;
    res++;
  }

  free(res_ptr);
  return;

}



inline void PerfStats::reset(const std::string& key) {
  std::map<std::string, Stats>::iterator s = stats_.find(key);
  if (s != stats_.end())
    s->second.data_.clear();
}


inline double PerfStats::sample(const std::string& key) {
  const double now = getTime();
  sample(key, now - last_, TIME);
  last_ = now;
  return now;
}



inline double PerfStats::sample(const std::string& key,
                                double             t,
                                Type               type) {
  double now = getTime();
  Stats& s = stats_[key];

  s.mutex_.lock();
  double last_ = s.last_absolute_;
  if (last_ == 0) {
    order_[insertion_id_] = key;
    insertion_id_++;
  }

  s.data_.push_back(t);
  s.type_ = type;
  s.last_period_ = now - last_;
  s.last_absolute_ = now;

  s.mutex_.unlock();
  return (now);
}



inline double PerfStats::start(void) {
  last_ = getTime();
  return last_;
}

#endif // __PERFSTATS_IMPL_HPP
