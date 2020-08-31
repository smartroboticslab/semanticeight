#ifndef TIMINGS_H
#define TIMINGS_H

#include <chrono>

#include "perfstats.h"

#if defined(SE_ENABLE_PERFSTATS) && SE_ENABLE_PERFSTATS

#define TICK(str) stats.sampleDurationStart(str);
#define TICKD(str) stats.sampleDurationStart(str, true);
#define TOCK(str) stats.sampleDurationEnd(str);

#else

#define TICK(str)
#define TICKD(str)
#define TOCK(str)

#endif

#endif // TIMINGS_H

