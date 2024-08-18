#ifndef OMEGA_DRIVERMETHODS_H
#define OMEGA_DRIVERMETHODS_H
//===--  --------------------*- C++ -*-===//
//
/// \file
/// \brief
///
//
//===----------------------------------------------------------------------===//

namespace OMEGA {

int OcnInit(MPIComm Comm, TimeInstant &StartTime, TimeInterval &RunInterval);

int OcnRun(TimeInstant &CurrTime, TimeInterval &RunInterval, Alarm &EndAlarm);

int OcnFinalize(TimeInstant &CurrTime);

} // end namespace OMEGA

#endif


