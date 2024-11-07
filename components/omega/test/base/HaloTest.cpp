//===-- Test driver for OMEGA Halo -------------------------------*- C++ -*-===/
//
/// \file
/// \brief Test driver for OMEGA Halo class
///
/// This driver tests the OMEGA model Halo class, which collects and stores
/// everything needed to perform halo exchanges on any supported Kokkos array
/// defined on a mesh in OMEGA with a given parallel decomposition. This
/// unit test driver tests functionality by creating Kokkos arrays of every
/// type and dimensionality supported in OMEGA, initializing each array based
/// on global IDs of the mesh elememts, performing halo exchanges, and
/// confirming the exchanged arrays are identical to the initial arrays.
///
//
//===-----------------------------------------------------------------------===/

#include "Halo.h"
#include "Config.h"
#include "DataTypes.h"
#include "Decomp.h"
#include "IO.h"
#include "Logging.h"
#include "MachEnv.h"
#include "OmegaKokkos.h"
#include "mpi.h"

#include <iostream>
#include <type_traits>

using namespace OMEGA;



//------------------------------------------------------------------------------
// This function template performs a single test on a Kokkos array type in a
// given index space. Two Kokkos arrays of the same type and size are input,
// InitArray contains the global IDs of the mesh elements for all the owned and
// halo elements of the array, while TestArray contains the global IDs only in
// the owned elements. The Halo class object, a label describing the test for
// output, an integer to accumulate errors, and optionally the index space of
// the input arrays (default is OnCell) are also input. A halo exchange is
// performed on TestArray, and then TestArray is compared to InitArray. If any
// elements differ the test is a failure and an error is returned.

template <typename T>
void haloExchangeTest(
    Halo *MyHalo,
    T InitArray,  /// Array initialized based on global IDs of mesh elements
    T &TestArray, /// Array only initialized in owned elements
    const char *Label,                          /// Unique label for test
    I4 &TotErr,                          /// Integer to track errors
    MeshElement ThisElem = OnCell /// index space, cell by default
) {

   I4 IErr{0}; // error code

   // Set total array size and ensure arrays are of same size
   I4 NTot = InitArray.size();
   if (NTot != TestArray.size()) {
      LOG_ERROR("HaloTest: {} arrays must be of same size", Label);
      LOG_INFO("HaloTest: {} exchange test FAIL", Label);
      TotErr += -1;
      return;
   }

   // Perform halo exchange
   IErr = MyHalo->exchangeFullArrayHalo(TestArray, ThisElem);
   if (IErr != 0) {
      LOG_ERROR("HaloTest: Error during {} halo exchange", Label);
      LOG_INFO("HaloTest: {} exchange test FAIL", Label);
      TotErr += -1;
      return;
   }

   // Collapse arrays to 1D for easy iteration
   Kokkos::View<typename T::value_type *, typename T::array_layout,
                typename T::memory_space>
       CollapsedInit(InitArray.data(), InitArray.size());
   Kokkos::View<typename T::value_type *, typename T::array_layout,
                typename T::memory_space>
       CollapsedTest(TestArray.data(), TestArray.size());

   // Confirm all elements are identical, if not set error code
   // and break out of loop
   for (int N = 0; N < NTot; ++N) {
      if (CollapsedInit(N) != CollapsedTest(N)) {
         IErr = -1;
         break;
      }
   }

   if (IErr == 0) {
      LOG_INFO("HaloTest: {} exchange test PASS", Label);
   } else {
      LOG_INFO("HaloTest: {} exchange test FAIL", Label);
      TotErr += -1;
   }

   return;

} // end haloExchangeTest

template <typename T>
void haloExchangeTestD(
    HaloD *MyHalo,
    T InitArray,  /// Array initialized based on global IDs of mesh elements
    T &TestArray, /// Array only initialized in owned elements
    const char *Label,                          /// Unique label for test
    I4 &TotErr,                          /// Integer to track errors
    MeshElement ThisElem = OnCell /// index space, cell by default
) {

   I4 IErr{0}; // error code

   // Set total array size and ensure arrays are of same size
   I4 NTot = InitArray.size();
   if (NTot != TestArray.size()) {
      LOG_ERROR("HaloTest: {} arrays must be of same size", Label);
      LOG_INFO("HaloTest: {} exchange test FAIL", Label);
      TotErr += -1;
      return;
   }

   // Perform halo exchange
   IErr = MyHalo->exchangeFullArrayHalo(TestArray, ThisElem);
   if (IErr != 0) {
      LOG_ERROR("HaloTest: Error during {} halo exchange", Label);
      LOG_INFO("HaloTest: {} exchange test FAIL", Label);
      TotErr += -1;
      return;
   }
//   std::cout << "exchange complete" << std::endl;

   auto TestArrayH = createHostMirrorCopy(TestArray);
   auto InitArrayH = createHostMirrorCopy(InitArray);
//
//   std::cout << "create host copies" << std::endl;
   // Collapse arrays to 1D for easy iteration
   Kokkos::View<typename T::value_type *, typename T::array_layout,
                HostMemSpace>
       CollapsedInit(InitArrayH.data(), InitArrayH.size());
   Kokkos::View<typename T::value_type *, typename T::array_layout,
                HostMemSpace>
       CollapsedTest(TestArrayH.data(), TestArrayH.size());

//   std::cout << "arrays collapsed" << std::endl;
   // Confirm all elements are identical, if not set error code
   // and break out of loop
   I4 NDiff = 0;
   for (int N = 0; N < NTot; ++N) {
//      parallelFor({NTot}, KOKKOS_LAMBDA(int N) {
//   parallelReduce({NTot}, KOKKOS_LAMBDA(int N, int &Accum) {
      if (CollapsedInit(N) != CollapsedTest(N)) {
//         std::cout << N << " " << CollapsedInit(N) << " " << CollapsedTest(N) << std::endl;
         IErr = -1;
         ++NDiff;
////         ++Accum;
////         break;
      }
   }
//   }, NDiff);
//   std::cout << "ndiff: " << NDiff << std::endl;
   if (NDiff != 0) IErr = -1;
//   std::cout << " compare complete " << std::endl;
   if (IErr == 0) {
      LOG_INFO("HaloTest: {} exchange test PASS", Label);
   } else {
      LOG_INFO("HaloTest: {} exchange test FAIL", Label);
      TotErr += -1;
   }

   return;

} // end haloExchangeTest

//------------------------------------------------------------------------------
// Initialization routine for Halo tests. Calls all the init routines needed
// to create the default Halo.

int initHaloTest() {

   I4 IErr{0};

   // Initialize the machine environment and fetch the default environment
   // pointer and the MPI communicator
   MachEnv::init(MPI_COMM_WORLD);
   MachEnv *DefEnv = MachEnv::getDefault();
   MPI_Comm DefComm       = DefEnv->getComm();

   // Initialize the logging system
   initLogging(DefEnv);

   // Open config file
   Config("Omega");
   IErr = Config::readAll("omega.yml");
   if (IErr != 0) {
      LOG_CRITICAL("HaloTest: Error reading config file");
      return IErr;
   }

   // Initialize the IO system
   IErr = IO::init(DefComm);
   if (IErr != 0)
      LOG_ERROR("HaloTest: error initializing parallel IO");

   // Create the default decomposition (initializes the decomposition)
   IErr = Decomp::init();
   if (IErr != 0)
      LOG_ERROR("HaloTest: error initializing default decomposition");

   // Initialize the default halo
   IErr = Halo::init();
   if (IErr != 0)
      LOG_ERROR("HaloTest: error initializing default halo");

//   std::cout << " begin init " << std::endl;
   // Initialize the default halo
   IErr = HaloD::init();
   if (IErr != 0)
      LOG_ERROR("HaloTest: error initializing default halo");
//   std::cout << " end init " << std::endl;

   return IErr;

} // end initHaloTest

//------------------------------------------------------------------------------
// The test driver. Performs halo exchange tests of all index spaces and all
// supported Kokkos array types. For each test, an initial array is set based on
// Global IDs of the mesh elements in the given index space for all owned and
// halo elements, and is copied into the test array. The test array halo
// elements are then set to junk values and a halo exchange is performed, which
// if successful will fetch the proper values from neighboring test arrays such
// that the test array and initial array are equivalent.

int main(int argc, char *argv[]) {

   // Error tracking variables
   I4 TotErr{0};
   I4 IErr{0};

   // Initialize global MPI environment and Kokkos
   MPI_Init(&argc, &argv);
   Kokkos::initialize();
   {

      // Call Halo test initialization routine
      IErr = initHaloTest();
      if (IErr != 0)
         LOG_ERROR("HaloTest: initHaloTest error");

      // Retrieve pointer to default halo
      Halo *DefHalo = Halo::getDefault();
      HaloD *DefHaloD = HaloD::getDefault();

      // Retrieve pointer to default decomposition
      Decomp *DefDecomp = Decomp::getDefault();

      I4 NumOwned;
      I4 NumAll;

//      MachEnv *DefEnv = MachEnv::getDefault();
//      I4 TaskID = DefEnv->getMyTask();
//      MPI_Comm Comm_ = DefEnv->getComm();
//      I4 NTot = 10;
//
//      Array1DI4 TestArray("", NTot);
//      if (TaskID == 0) {
//         deepCopy(TestArray, -1);
//      } else {
//         deepCopy(TestArray, 999);
//      }
//
//      if (TaskID ==0) {
//         MPI_Status status;
//         IErr = MPI_Recv(TestArray.data(), NTot, MPI_INT, 1, MPI_ANY_TAG, Comm_, &status);
//      } else {
//         IErr = MPI_Send(TestArray.data(), NTot, MPI_INT, 0, 0, Comm_);
//      }
//
//      if (TaskID == 0 ) {
//         auto RecvArray = createHostMirrorCopy(TestArray);
//         for (int I = 0; I < NTot; ++I) {
//            std::cout << I << " " << RecvArray(I) << std::endl;
//         }
//      }

      // Perform 1DI4 array tests for each index space (cell, edge, and vertex)

      HostArray1DI4 Init1DI4Cell("Init1DI4Cell", DefDecomp->NCellsSize);
      HostArray1DI4 Test1DI4Cell("Test1DI4Cell", DefDecomp->NCellsSize);
      Array1DI4 Init1DI4CellD("Init1DI4CellD", DefDecomp->NCellsSize);
      Array1DI4 Test1DI4CellD("Test1DI4CellD", DefDecomp->NCellsSize);

      NumOwned     = DefDecomp->NCellsOwned;
      NumAll       = DefDecomp->NCellsAll;
      Init1DI4Cell = DefDecomp->CellIDH;
      deepCopy(Test1DI4Cell, Init1DI4Cell);

      for (int ICell = NumOwned; ICell < NumAll; ++ICell) {
         Test1DI4Cell(ICell) = -1;
      }

      deepCopy(Init1DI4CellD, Init1DI4Cell);
      deepCopy(Test1DI4CellD, Test1DI4Cell);

//      haloExchangeTest(DefHalo, Init1DI4Cell, Test1DI4Cell, "1DI4 Cell",
//                       TotErr);
//      std::cout << " halo test start " << std::endl;
      haloExchangeTestD(DefHaloD, Init1DI4Cell, Test1DI4Cell, "1DI4 Cell",
                       TotErr);
//      haloExchangeTestD(DefHaloD, Init1DI4CellD, Test1DI4CellD, "1DI4 CellD",
//                       TotErr);
//      if(DefHaloD->checkArrayMemLoc<HostArray1DI4>() == ArrayMemLoc::Host) {
//         std::cout << "Array on Host" << std::endl;
//      }


//      std::cout << " halo test end " << std::endl;

      HostArray1DI4 Init1DI4Edge("Init1DI4Edge", DefDecomp->NEdgesSize);
      HostArray1DI4 Test1DI4Edge("Test1DI4Edge", DefDecomp->NEdgesSize);
      Array1DI4 Init1DI4EdgeD("Init1DI4EdgeD", DefDecomp->NEdgesSize);
      Array1DI4 Test1DI4EdgeD("Test1DI4EdgeD", DefDecomp->NEdgesSize);

      NumOwned     = DefDecomp->NEdgesOwned;
      NumAll       = DefDecomp->NEdgesAll;
      Init1DI4Edge = DefDecomp->EdgeIDH;
      deepCopy(Test1DI4Edge, Init1DI4Edge);

      for (int IEdge = NumOwned; IEdge < NumAll; ++IEdge) {
         Test1DI4Edge(IEdge) = -1;
      }

      deepCopy(Init1DI4EdgeD, Init1DI4Edge);
      deepCopy(Test1DI4EdgeD, Test1DI4Edge);
//      haloExchangeTestD(DefHaloD, Init1DI4Edge, Test1DI4Edge, "1DI4 Edge", TotErr,
//                       OnEdge);
//      haloExchangeTestD(DefHaloD, Init1DI4EdgeD, Test1DI4EdgeD, "1DI4 EdgeD", TotErr,
//                       OnEdge);

      HostArray1DI4 Init1DI4Vertex("Init1DI4Vertex",
                                          DefDecomp->NVerticesSize);
      HostArray1DI4 Test1DI4Vertex("Test1DI4Vertex",
                                          DefDecomp->NVerticesSize);
      Array1DI4 Init1DI4VertexD("Init1DI4VertexD",
                                      DefDecomp->NVerticesSize);
      Array1DI4 Test1DI4VertexD("Test1DI4VertexD",
                                      DefDecomp->NVerticesSize);

      NumOwned       = DefDecomp->NVerticesOwned;
      NumAll         = DefDecomp->NVerticesAll;
      Init1DI4Vertex = DefDecomp->VertexIDH;
      deepCopy(Test1DI4Vertex, Init1DI4Vertex);

      for (int IVertex = NumOwned; IVertex < NumAll; ++IVertex) {
         Test1DI4Vertex(IVertex) = -1;
      }

      deepCopy(Init1DI4VertexD, Init1DI4Vertex);
      deepCopy(Test1DI4VertexD, Test1DI4Vertex);
//      haloExchangeTestD(DefHaloD, Init1DI4Vertex, Test1DI4Vertex, "1DI4 Vertex",
//                       TotErr, OnVertex);
//      haloExchangeTestD(DefHaloD, Init1DI4VertexD, Test1DI4VertexD, "1DI4 VertexD",
//                       TotErr, OnVertex);

      // Declaration of variables for remaining tests

      // Random dimension sizes
      I4 N2{20};
      I4 N3{10};
      I4 N4{3};
      I4 N5{2};

      NumOwned = DefDecomp->NCellsOwned;
      NumAll   = DefDecomp->NCellsAll;

      // Declare init and test arrays for all the remaining array types
      HostArray1DI8 Init1DI8("Init1DI8", NumAll);
      HostArray1DR4 Init1DR4("Init1DR4", NumAll);
      HostArray1DR8 Init1DR8("Init1DR8", NumAll);
      Array1DI8 Init1DI8D("Init1DI8D", NumAll);
      Array1DR4 Init1DR4D("Init1DR4D", NumAll);
      Array1DR8 Init1DR8D("Init1DR8D", NumAll);
      HostArray2DI4 Init2DI4("Init2DI4", NumAll, N2);
      HostArray2DI8 Init2DI8("Init2DI8", NumAll, N2);
      HostArray2DR4 Init2DR4("Init2DR4", NumAll, N2);
      HostArray2DR8 Init2DR8("Init2DR8", NumAll, N2);
      HostArray3DI4 Init3DI4("Init3DI4", N3, NumAll, N2);
      HostArray3DI8 Init3DI8("Init3DI8", N3, NumAll, N2);
      HostArray3DR4 Init3DR4("Init3DR4", N3, NumAll, N2);
      HostArray3DR8 Init3DR8("Init3DR8", N3, NumAll, N2);
      HostArray4DI4 Init4DI4("Init4DI4", N4, N3, NumAll, N2);
      HostArray4DI8 Init4DI8("Init4DI8", N4, N3, NumAll, N2);
      HostArray4DR4 Init4DR4("Init4DR4", N4, N3, NumAll, N2);
      HostArray4DR8 Init4DR8("Init4DR8", N4, N3, NumAll, N2);
      HostArray5DI4 Init5DI4("Init5DI4", N5, N4, N3, NumAll, N2);
      HostArray5DI8 Init5DI8("Init5DI8", N5, N4, N3, NumAll, N2);
      HostArray5DR4 Init5DR4("Init5DR4", N5, N4, N3, NumAll, N2);
      HostArray5DR8 Init5DR8("Init5DR8", N5, N4, N3, NumAll, N2);
      HostArray1DI8 Test1DI8("Test1DI8", NumAll);
      HostArray1DR4 Test1DR4("Test1DR4", NumAll);
      HostArray1DR8 Test1DR8("Test1DR8", NumAll);
      Array1DI8 Test1DI8D("Test1DI8D", NumAll);
      Array1DR4 Test1DR4D("Test1DR4D", NumAll);
      Array1DR8 Test1DR8D("Test1DR8D", NumAll);
      HostArray2DI4 Test2DI4("Test2DI4", NumAll, N2);
      HostArray2DI8 Test2DI8("Test2DI8", NumAll, N2);
      HostArray2DR4 Test2DR4("Test2DR4", NumAll, N2);
      HostArray2DR8 Test2DR8("Test2DR8", NumAll, N2);
      HostArray3DI4 Test3DI4("Test3DI4", N3, NumAll, N2);
      HostArray3DI8 Test3DI8("Test3DI8", N3, NumAll, N2);
      HostArray3DR4 Test3DR4("Test3DR4", N3, NumAll, N2);
      HostArray3DR8 Test3DR8("Test3DR8", N3, NumAll, N2);
      HostArray4DI4 Test4DI4("Test4DI4", N4, N3, NumAll, N2);
      HostArray4DI8 Test4DI8("Test4DI8", N4, N3, NumAll, N2);
      HostArray4DR4 Test4DR4("Test4DR4", N4, N3, NumAll, N2);
      HostArray4DR8 Test4DR8("Test4DR8", N4, N3, NumAll, N2);
      HostArray5DI4 Test5DI4("Test5DI4", N5, N4, N3, NumAll, N2);
      HostArray5DI8 Test5DI8("Test5DI8", N5, N4, N3, NumAll, N2);
      HostArray5DR4 Test5DR4("Test5DR4", N5, N4, N3, NumAll, N2);
      HostArray5DR8 Test5DR8("Test5DR8", N5, N4, N3, NumAll, N2);

      Array2DR8 InitD2DR8("Test2DR8", NumAll, N2);
      Array2DR8 TestD2DR8("Test2DR8", NumAll, N2);

      // Initialize and run remaining 1D tests
      for (int ICell = 0; ICell < NumAll; ++ICell) {
         I4 NewVal = DefDecomp->CellIDH(ICell);
         Init1DI8(ICell)  = static_cast<I8>(NewVal);
         Init1DR4(ICell)  = static_cast<R4>(NewVal);
         Init1DR8(ICell)  = static_cast<R8>(NewVal);
      }

      deepCopy(Test1DI8, Init1DI8);
      deepCopy(Test1DR4, Init1DR4);
      deepCopy(Test1DR8, Init1DR8);

      for (int ICell = NumOwned; ICell < NumAll; ++ICell) {
         Test1DI8(ICell) = -1;
         Test1DR4(ICell) = -1;
         Test1DR8(ICell) = -1;
      }

      deepCopy(Init1DI8D, Init1DI8);
      deepCopy(Init1DR4D, Init1DR4);
      deepCopy(Init1DR8D, Init1DR8);
      deepCopy(Test1DI8D, Test1DI8);
      deepCopy(Test1DR4D, Test1DR4);
      deepCopy(Test1DR8D, Test1DR8);

//      haloExchangeTest(DefHalo, Init1DI8, Test1DI8, "1DI8", TotErr);
//      haloExchangeTest(DefHalo, Init1DR4, Test1DR4, "1DR4", TotErr);
//      haloExchangeTest(DefHalo, Init1DR8, Test1DR8, "1DR8", TotErr);
//      haloExchangeTestD(DefHaloD, Init1DI8D, Test1DI8D, "1DI8D", TotErr);
//      haloExchangeTestD(DefHaloD, Init1DR4D, Test1DR4D, "1DR4D", TotErr);
//      haloExchangeTestD(DefHaloD, Init1DR8D, Test1DR8D, "1DR8D", TotErr);

      // Initialize and run 2D tests
      for (int ICell = 0; ICell < NumAll; ++ICell) {
         for (int J = 0; J < N2; ++J) {
            I4 NewVal   = (J + 1) * DefDecomp->CellIDH(ICell);
            Init2DI4(ICell, J) = NewVal;
            Init2DI8(ICell, J) = static_cast<I8>(NewVal);
            Init2DR4(ICell, J) = static_cast<R4>(NewVal);
            Init2DR8(ICell, J) = static_cast<R8>(NewVal);
         }
      }

      deepCopy(Test2DI4, Init2DI4);
      deepCopy(Test2DI8, Init2DI8);
      deepCopy(Test2DR4, Init2DR4);
      deepCopy(Test2DR8, Init2DR8);

//      deepCopy(InitD2DR8, Init2DR8);

      for (int ICell = NumOwned; ICell < NumAll; ++ICell) {
         for (int J = 0; J < N2; ++J) {
            Test2DI4(ICell, J) = -1;
            Test2DI8(ICell, J) = -1;
            Test2DR4(ICell, J) = -1;
            Test2DR8(ICell, J) = -1;
         }
      }
//      deepCopy(TestD2DR8, Test2DR8);

//      haloExchangeTest(DefHalo, Init2DI4, Test2DI4, "2DI4", TotErr);
//      haloExchangeTest(DefHalo, Init2DI8, Test2DI8, "2DI8", TotErr);
//      haloExchangeTest(DefHalo, Init2DR4, Test2DR4, "2DR4", TotErr);
//      haloExchangeTest(DefHalo, Init2DR8, Test2DR8, "2DR8", TotErr);

//      haloDExchangeTest(DefHaloD, InitD2DR8, TestD2DR8, "D2DR8", TotErr);

      // Initialize and run 3D tests
      for (int K = 0; K < N3; ++K) {
         for (int ICell = 0; ICell < NumAll; ++ICell) {
            for (int J = 0; J < N2; ++J) {
               I4 NewVal = (K + 1) * (J + 1) * DefDecomp->CellIDH(ICell);
               Init3DI4(K, ICell, J) = NewVal;
               Init3DI8(K, ICell, J) = static_cast<I8>(NewVal);
               Init3DR4(K, ICell, J) = static_cast<R4>(NewVal);
               Init3DR8(K, ICell, J) = static_cast<R8>(NewVal);
            }
         }
      }

      deepCopy(Test3DI4, Init3DI4);
      deepCopy(Test3DI8, Init3DI8);
      deepCopy(Test3DR4, Init3DR4);
      deepCopy(Test3DR8, Init3DR8);

      for (int K = 0; K < N3; ++K) {
         for (int ICell = NumOwned; ICell < NumAll; ++ICell) {
            for (int J = 0; J < N2; ++J) {
               Test3DI4(K, ICell, J) = -1;
               Test3DI8(K, ICell, J) = -1;
               Test3DR4(K, ICell, J) = -1;
               Test3DR8(K, ICell, J) = -1;
            }
         }
      }

//      haloExchangeTest(DefHalo, Init3DI4, Test3DI4, "3DI4", TotErr);
//      haloExchangeTest(DefHalo, Init3DI8, Test3DI8, "3DI8", TotErr);
//      haloExchangeTest(DefHalo, Init3DR4, Test3DR4, "3DR4", TotErr);
//      haloExchangeTest(DefHalo, Init3DR8, Test3DR8, "3DR8", TotErr);

      // Initialize and run 4D tests
      for (int L = 0; L < N4; ++L) {
         for (int K = 0; K < N3; ++K) {
            for (int ICell = 0; ICell < NumAll; ++ICell) {
               for (int J = 0; J < N2; ++J) {
                  I4 NewVal =
                      (L + 1) * (K + 1) * (J + 1) * DefDecomp->CellIDH(ICell);
                  Init4DI4(L, K, ICell, J) = NewVal;
                  Init4DI8(L, K, ICell, J) = static_cast<I8>(NewVal);
                  Init4DR4(L, K, ICell, J) = static_cast<R4>(NewVal);
                  Init4DR8(L, K, ICell, J) = static_cast<R8>(NewVal);
               }
            }
         }
      }

      deepCopy(Test4DI4, Init4DI4);
      deepCopy(Test4DI8, Init4DI8);
      deepCopy(Test4DR4, Init4DR4);
      deepCopy(Test4DR8, Init4DR8);

      for (int L = 0; L < N4; ++L) {
         for (int K = 0; K < N3; ++K) {
            for (int ICell = NumOwned; ICell < NumAll; ++ICell) {
               for (int J = 0; J < N2; ++J) {
                  Test4DI4(L, K, ICell, J) = -1;
                  Test4DI8(L, K, ICell, J) = -1;
                  Test4DR4(L, K, ICell, J) = -1;
                  Test4DR8(L, K, ICell, J) = -1;
               }
            }
         }
      }

//      haloExchangeTest(DefHalo, Init4DI4, Test4DI4, "4DI4", TotErr);
//      haloExchangeTest(DefHalo, Init4DI8, Test4DI8, "4DI8", TotErr);
//      haloExchangeTest(DefHalo, Init4DR4, Test4DR4, "4DR4", TotErr);
//      haloExchangeTest(DefHalo, Init4DR8, Test4DR8, "4DR8", TotErr);

      // Initialize and run 5D tests
      for (int M = 0; M < N5; ++M) {
         for (int L = 0; L < N4; ++L) {
            for (int K = 0; K < N3; ++K) {
               for (int ICell = 0; ICell < NumAll; ++ICell) {
                  for (int J = 0; J < N2; ++J) {
                     I4 NewVal = (M + 1) * (L + 1) * (K + 1) * (J + 1) *
                                        DefDecomp->CellIDH(ICell);
                     Init5DI4(M, L, K, ICell, J) = NewVal;
                     Init5DI8(M, L, K, ICell, J) =
                         static_cast<I8>(NewVal);
                     Init5DR4(M, L, K, ICell, J) =
                         static_cast<R4>(NewVal);
                     Init5DR8(M, L, K, ICell, J) =
                         static_cast<R8>(NewVal);
                  }
               }
            }
         }
      }

      deepCopy(Test5DI4, Init5DI4);
      deepCopy(Test5DI8, Init5DI8);
      deepCopy(Test5DR4, Init5DR4);
      deepCopy(Test5DR8, Init5DR8);

      for (int M = 0; M < N5; ++M) {
         for (int L = 0; L < N4; ++L) {
            for (int K = 0; K < N3; ++K) {
               for (int ICell = NumOwned; ICell < NumAll; ++ICell) {
                  for (int J = 0; J < N2; ++J) {
                     Test5DI4(M, L, K, ICell, J) = -1;
                     Test5DI8(M, L, K, ICell, J) = -1;
                     Test5DR4(M, L, K, ICell, J) = -1;
                     Test5DR8(M, L, K, ICell, J) = -1;
                  }
               }
            }
         }
      }

//      haloExchangeTest(DefHalo, Init5DI4, Test5DI4, "5DI4", TotErr);
//      haloExchangeTest(DefHalo, Init5DI8, Test5DI8, "5DI8", TotErr);
//      haloExchangeTest(DefHalo, Init5DR4, Test5DR4, "5DR4", TotErr);
//      haloExchangeTest(DefHalo, Init5DR8, Test5DR8, "5DR8", TotErr);

      // Memory clean up
      Halo::clear();
      HaloD::clear();
      Decomp::clear();
      MachEnv::removeAll();

//      std::cout << "success" <<std::endl;
      if (TotErr == 0) {
         LOG_INFO("HaloTest: Successful completion");
      } else {
         LOG_INFO("HaloTest: Failed");
      }
   }
   Kokkos::finalize();
   MPI_Finalize();

   if (TotErr >= 256)
      TotErr = 255;

   return TotErr;

} // end of main
//===-----------------------------------------------------------------------===/
