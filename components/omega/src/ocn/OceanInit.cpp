//===-- ocn/OceanInit.cpp - Ocean Initialization ----------------*- C++ -*-===//
//
// This file contians ocnInit and associated methods which initialize Omega.
// The ocnInit process reads the config file and uses the config options to
// initialize time management and call all the individual initialization
// routines for each module in Omega.
//
//===----------------------------------------------------------------------===//

#include "OceanDriver.h"
#include "AuxiliaryState.h"
#include "Config.h"
#include "DataTypes.h"
#include "Decomp.h"
#include "Halo.h"
#include "HorzMesh.h"
#include "IO.h"
#include "IOField.h"
#include "Logging.h"
#include "MachEnv.h"
#include "OceanState.h"
#include "TendencyTerms.h"
#include "TimeMgr.h"
#include "TimeStepper.h"

#include "mpi.h"
#include <iostream>

namespace OMEGA {

int ocnInit(
   MPI_Comm Comm, ///< [in] ocean MPI communicator
   Calendar &OmegaCal, ///< [out] sim calendar
   TimeInstant &StartTime, ///< [out] sim start time
   TimeInterval &RunInterval, ///< [out] interval for run method
   TimeInterval &TimeStep, ///< [out] initial time step
   Alarm &EndAlarm//, ///< [out] alarm to end simulation
) {

   I4 RetErr = 0;
   I4 Err = 0;

   // Init the default machine environment based on input MPI communicator
   MachEnv::init(Comm);
   MachEnv *DefEnv = MachEnv::getDefault();

   initLogging(DefEnv);

   // Read config file into Config object
   Config("omega");
   Err = Config::readAll("omega.yml");
   if (Err != 0) {
      LOG_ERROR("ocnInit: Error reading config file");
      ++RetErr;
   }
   Config *OmegaConfig = Config::getOmegaConfig();

   // read and save time management options from Config
   Err = initTimeManagement(OmegaCal, StartTime, RunInterval, EndAlarm,
                            OmegaConfig);
   if (Err != 0) {
      LOG_ERROR("ocnInit: Error initializing time management");
      ++RetErr;
   }

//   R8 Rlength;
//   Err = RunInterval.get(Rlength, TimeUnits::Seconds);
//   std::cout << Rlength << std::endl;

//   CalendarKind Kind0 = CalendarNoCalendar;
//   I4 ID0             = 1;
//   std::string Name0(" ");
//   I4 DaysPerMonth0[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
//   I4 MonthsPerYear0  = 12;
//   I4 SecondsPerDay0  = 0;
//   I4 SecondsPerYear0 = 0;
//   I4 DaysPerYear0    = 0;
//
//   I4 Err1 = OmegaCal.get(&ID0, &Name0, &Kind0, DaysPerMonth0, &MonthsPerYear0,
//                       &SecondsPerDay0, &SecondsPerYear0, &DaysPerYear0);
//
//   std::cout << ID0 << Name0 << Kind0 << DaysPerMonth0[0] << MonthsPerYear0 <<
//             SecondsPerDay0 << SecondsPerYear0 << DaysPerYear0 << std::endl;
//
//   std::cout << StartTime.getString(4, 4, "_") << std::endl;


   // initialize remaining Omega modules
   Err = IO::init(Comm);
   if (Err != 0) {
      LOG_ERROR("ocnInit: Error initializing parallel IO");
      ++RetErr;
   }

   Err = Decomp::init();
   if (Err != 0) {
      LOG_ERROR("ocnInit: Error initializing default decomposition");
      ++RetErr;
   }

   Err = Halo::init();
   if (Err != 0) {
      LOG_ERROR("ocnInit: Error initializing default halo");
      ++RetErr;
   }

   Err = HorzMesh::init();
   if (Err != 0) {
      ++RetErr;
      LOG_ERROR("ocnInit: Error initializing default mesh");
   }

   // TODO: retrieve NVertLevels
   const I4 NVertLevels = 60;
   const auto &DefHorzMesh = HorzMesh::getDefault();
   MetaDim::create("NCells", DefHorzMesh->NCellsSize);
   MetaDim::create("NEdges", DefHorzMesh->NEdgesSize);
   MetaDim::create("NVertices", DefHorzMesh->NVerticesSize);
   MetaDim::create("NVertLevels", NVertLevels);

   Err = OceanState::init();
   if (Err != 0) {
      ++RetErr;
      LOG_ERROR("ocnInit: Error initializing default state");
   }

   Err = AuxiliaryState::init();
   if (Err != 0) {
      ++RetErr;
      LOG_ERROR("ocnInit: Error initializing default aux state");
   }

   Err = Tendencies::init();
   if (Err != 0) {
      ++RetErr;
      LOG_ERROR("ocnInit: Error initializing default tendencies");
   }

   Err = TimeStepper::init();
   if (Err != 0) {
      ++RetErr;
      LOG_ERROR("ocnInit: Error initializing default time stepper");
   }

   Err = initTimeIntegration(TimeStep, OmegaConfig);
   if (Err != 0) {
      ++RetErr;
      LOG_ERROR("ocnInit: Error initializing time integration");
   }


   return RetErr;
}

// read time management options from config
int initTimeManagement(Calendar &OmegaCal, TimeInstant &StartTime, 
                       TimeInterval &RunInterval, Alarm &EndAlarm,
                       Config *OmegaConfig) {

   I4 RetErr = 0;

   // create zero length interval for comparison
   TimeInterval ZeroInterval;

   // extract variables for time management group
   Config TimeMgmtConfig("TimeManagement");
   if (OmegaConfig->existsGroup("TimeManagement")) {
      int Err1 = OmegaConfig->get(TimeMgmtConfig);
      // check requested calendar is a valid option, return error if not found
      if ( TimeMgmtConfig.existsVar("CalendarType") ) {
         std::string ConfigCalStr;
         CalendarKind ConfigCalKind = CalendarUnknown;
         I4 Err1 = TimeMgmtConfig.get("CalendarType", ConfigCalStr);
         I4 ICalType = 9;
         for ( I4 I = 0; I < NUM_SUPPORTED_CALENDARS; ++I ) {
            std::size_t match = CalendarKindName[I].find(ConfigCalStr);
            if ( match != std::string::npos ) {
               ICalType = I;
               ConfigCalKind = (CalendarKind) (ICalType + 1);
               std::cout << ICalType << " " << ConfigCalStr << std::endl;
               break;
            }
         }
         if ( ICalType == 9 ) {
            LOG_ERROR("ocnInit: Requested Calendar type not found");
            ++RetErr;
            return RetErr;
         }
         OmegaCal.~Calendar();
         OmegaCal = Calendar::Calendar(ConfigCalStr,
                                       ConfigCalKind);
      } 

      // check for start time and set if found
      if ( TimeMgmtConfig.existsVar("StartTime") ) {
         std::string StartTimeStr;
         I4 Err1 = TimeMgmtConfig.get("StartTime", StartTimeStr);

         StartTime = TimeInstant::TimeInstant(&OmegaCal, StartTimeStr); 

      }

      // set RunInterval by checkint for StopTime and RunDuration in Config,
      // if both are present, use shortest non-zero interval
      if ( TimeMgmtConfig.existsVar("StopTime") ) {
         std::string StopTimeStr;
         std::string NoneStr("none");
         I4 Err1 = TimeMgmtConfig.get("StopTime", StopTimeStr);
         if (StopTimeStr.compare(NoneStr) != 0) {
            TimeInstant StopTime(&OmegaCal, StopTimeStr);
            RunInterval = StopTime - StartTime;
         }
      } 
      if ( TimeMgmtConfig.existsVar("RunDuration") ) {
         std::string RunDurationStr;
         I4 Err1 = TimeMgmtConfig.get("RunDuration", RunDurationStr);
         TimeInterval IntervalFromStr(RunDurationStr);
         if ((IntervalFromStr > ZeroInterval and IntervalFromStr < RunInterval)
             or RunInterval == ZeroInterval) {
            RunInterval = IntervalFromStr;
         }
      }
   }

   // return error if RunInterval set to zero
   if (RunInterval == ZeroInterval) {
      LOG_ERROR("ocnInit: Simulation run duration set to zero");
      ++RetErr;
   }

   // set EndAlarm based on length of RunInterval
   TimeInstant EndTime = StartTime + RunInterval;
   EndAlarm = Alarm::Alarm("End Alarm", EndTime);

   return RetErr;
}

int initTimeIntegration(TimeInterval &TimeStep, Config *OmegaConfig) {

   I4 RetErr = 0;
   I4 Err = 0;

   auto *DefMesh     = HorzMesh::getDefault();
   auto *DefAuxState = AuxiliaryState::getDefault();
   auto *DefHalo     = Halo::getDefault();
   auto *DefTend     = Tendencies::getDefault();

   Config TimeIntConfig("TimeIntegration");
   if (OmegaConfig->existsGroup("TimeIntegration")) {
      Err = OmegaConfig->get(TimeIntConfig);
      std::string TimeStepStr;
      if ( TimeIntConfig.existsVar("TimeStep") ) {
         Err = TimeIntConfig.get("TimeStep", TimeStepStr);
         TimeStep = TimeInterval::TimeInterval(TimeStepStr);
      }
      std::string StepperStr;
      if ( TimeIntConfig.existsVar("TimeStepper")) {
         Err = TimeIntConfig.get("TimeStepper", StepperStr);
         if (StepperStr == "RungeKutta2") {
//            TimeStepper::erase("Default");
//            TimeStepper::DefaultTimeStepper = TimeStepper::create("Default",
//                TimeStepperType::RungeKutta2, DefTend, DefAuxState,
//                DefMesh, DefHalo);
         } else if (StepperStr == "RungeKutta4") {
//            TimeStepper::erase("Default");
//            TimeStepper::DefaultTimeStepper = TimeStepper::create("Default",
//                TimeStepperType::RungeKutta4, DefTend, DefAuxState,
//                DefMesh, DefHalo);
         }

      }
   }

   return RetErr;
}

} // end namespace OMEGA

//===----------------------------------------------------------------------===//
