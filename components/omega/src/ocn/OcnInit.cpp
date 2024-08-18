//===--  -------------*- C++ -*-===//
//
//
//===----------------------------------------------------------------------===//

#include "DataTypes.h"
#include "Decomp.h"
#include "Halo.h"
#include "HorzMesh.h"
#include "IO.h"
#include "Logging.h"
#include "MachEnv.h"
#include "TimeMgr.h"

#include "mpi.h"

namespace OMEGA {

int OcnInit(
   MPI_Comm Comm, ///< [in] ocean MPI communicator
   TimeInstant &StartTime, ///< [out] sim start time
   TimeInterval &RunInterval ///< [out] interval for run method
//   State &CurrState ///< [out] current model state
) {

   I4 RetErr = 0;
   I4 IErr = 0;

   MachEnv::init(Comm);
   MachEnv *DefEnv = MachEnv::getDefault();

   initLogging(DefEnv);

   IErr = IO::init(DefComm);
   if (IErr != 0) {
      LOG_ERROR("OcnInit: Error initializing parallel IO");
      ++RetErr;
   }

   IErr = Decomp::init();
   if (IErr != 0) {
      LOG_ERROR("OcnInit: Error initializing default decomposition");
      ++RetErr;
   }

   IErr = Halo::init();
   if (IErr != 0) {
      LOG_ERROR("OcnInit: Error initializing default halo");
      ++RetErr;
   }

   IErr = HorzMesh::init();
   if (IErr != 0) {
      ++RetErr;
      LOG_ERROR("OcnInit: Error initializing default mesh");
   }

   IErr = OceanState::init();
   if (StateErr != 0) {
      ++RetErr;
      LOG_ERROR("OcnInit: Error initializing default state");
   }

   IErr = AuxiliaryState::init();
   if (AuxStateErr != 0) {
      ++RetErr;
      LOG_ERROR("OcnInit: Error initializing default aux state");
   }



   return RetErr;
}

} // end namespace OMEGA

//===----------------------------------------------------------------------===//
