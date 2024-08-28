//===--  -------------*- C++ -*-===//
//
//
//===----------------------------------------------------------------------===//

#include "OceanDriver.h"
#include "TimeMgr.h"

namespace OMEGA {

int ocnRun(
   TimeInstant &CurrTime, ///< [inout] current sim time
   TimeInterval &RunInterval, ///< [in] interval to advance model
   TimeInterval &TimeStep, ///< [in] initial time step
   Alarm &EndAlarm//, ///< [out] alarm to end simulation
   //OceanState &CurrState ///< [inout] current model state
) {

   I4 RetVal = 0;
   I4 Err = 0;

//   TimeInterval TimeStep(RunInterval);
//   Config *OmegaConfig = Config::getOmegaConfig();
//   Config TimeIntConfig("TimeIntegration");
//   if (OmegaConfig->existsGroup("TimeIntegration")) {
//      Err = OmegaConfig->get(TimeIntConfig);
//      std::string TimeStepStr;
//      if ( TimeIntConfig.existsVar("TimeStep") ) {
//         Err = TimeIntConfig.get("TimeStep", TimeStepStr);
//         TimeStep = TimeInterval::TimeInterval(TimeStepStr);
//      }
//   }

   Clock SimClock(CurrTime, TimeStep);
   Err = SimClock.attachAlarm(&EndAlarm);

   I8 IStep = 0;
   // Time Loop
//   while ( !(EndAlarm.isRinging()) ) {
//
//      // advance clock
//      SimClock.advance();
//
////      TimeInstant SimTime = SimClock.getCurrentTime();
////      std::string TimeStr = SimTime.getString(4,4,"-");
//
//
//      ++IStep;
////      std::cout << "Step # " << IStep << " " << TimeStr << std::endl;
//
//      if (IStep == 50) break;
//   }

   CurrTime = SimClock.getCurrentTime();

   std::cout << "NSteps:  " << IStep << std::endl;
   std::cout << EndAlarm.isRinging() << std::endl;

   return RetVal;
}

} // end namespace OMEGA


//===----------------------------------------------------------------------===//
