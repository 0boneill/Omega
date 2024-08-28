//===-- ocn/OceanFinalize.cpp -----------------------------------*- C++ -*-===//
//
// The ocnFinalize method writes a restart file if necessary, and then cleans
// up all objects Omega objects
//
//===----------------------------------------------------------------------===//

#include "OceanDriver.h"
#include "AuxiliaryState.h"
#include "Decomp.h"
#include "Halo.h"
#include "HorzMesh.h"
#include "IO.h"
#include "MachEnv.h"
#include "OceanState.h"
#include "TendencyTerms.h"
#include "TimeMgr.h"
#include "TimeStepper.h"

namespace OMEGA {

int ocnFinalize(
   TimeInstant &CurrTime//, ///< [in] current sim time
//   OceanState &CurrState  ///< [in] current model state
) {

   I4 RetVal = 0;
   I4 Err = 0;

   // Write restart file
   

   // clean up all objects
   TimeStepper::clear();
   Tendencies::clear();
   AuxiliaryState::clear();
   OceanState::clear();
   HorzMesh::clear();
   Halo::clear();
   Decomp::clear();
   MachEnv::removeAll();

   return RetVal;
}

} // end namespace OMEGA

//===----------------------------------------------------------------------===//
