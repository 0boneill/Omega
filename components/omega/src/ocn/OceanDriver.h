#ifndef OMEGA_DRIVER_H
#define OMEGA_DRIVER_H
//===-- ocn/OceanDriver.h ---------------------------------------*- C++ -*-===//
//
/// \file
/// \brief Defines ocean driver methods
///
/// This Header defines methods to drive Omega. These methods are designed to
/// run Omega as either a standalone ocean model or as a component of E3SM.
/// This process is divided into three phases: init, run, and finalize.
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "OceanState.h"
#include "TimeMgr.h"

#include "mpi.h"

namespace OMEGA {

int ocnInit(MPI_Comm Comm, Calendar &OmegaCal, TimeInstant &StartTime, 
            TimeInterval &RunInterval, TimeInterval &TimeStep, Alarm &EndAlarm);

int ocnRun(TimeInstant &CurrTime, TimeInterval &RunInterval, TimeInterval &TimeStep,
           Alarm &EndAlarm);

int ocnFinalize(TimeInstant &CurrTime);

int initTimeManagement(Calendar &OmegaCal, TimeInstant &StartTime, 
                       TimeInterval &RunInterval, Alarm &EndAlarm,
                       Config *OmegaConfig);

int initTimeIntegration(TimeInterval &TimeStep, Config *OmegaConfig);

} // end namespace OMEGA

#endif
