// SPDX-FileCopyrightText: 2019-2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2019-2021 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __SEMANTICEIGHT_DEFINITIONS_HPP
#define __SEMANTICEIGHT_DEFINITIONS_HPP



#define SE_BV_NONE 0
#define SE_BV_SPHERE 1
#define SE_BV_BOX 2

// Use bounding boxes by default.
#ifndef SE_BOUNDING_VOLUME
#    define SE_BOUNDING_VOLUME SE_BV_BOX
#endif



#define SE_VERBOSE_SILENT 0
#define SE_VERBOSE_MINIMAL 1
#define SE_VERBOSE_NORMAL 2
#define SE_VERBOSE_DETAILED 3
#define SE_VERBOSE_FLOOD 4

// Show minimal output by default.
#ifndef SE_VERBOSE
#    define SE_VERBOSE SE_VERBOSE_MINIMAL
#endif

#endif // __SEMANTICEIGHT_DEFINITIONS_HPP
