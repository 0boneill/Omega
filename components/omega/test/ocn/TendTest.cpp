#include "Tend.h"
#include "DataTypes.h"
#include "Decomp.h"
#include "Halo.h"
#include "HorzMesh.h"
#include "HorzOperators.h"
#include "IO.h"
#include "Logging.h"
#include "OceanTestCommon.h"
#include "OmegaKokkos.h"
#include "mpi.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdio.h>

using namespace OMEGA;

struct TestSetupPlane {
   Real Pi = M_PI;

   Real Lx = 1;
   Real Ly = std::sqrt(3) / 2;

   Real ExpectedDivErrorLInf     = 0.00124886886594427027;
   Real ExpectedDivErrorL2       = 0.00124886886590974385;
   Real ExpectedGradErrorLInf    = 0.00125026071878537952;
   Real ExpectedGradErrorL2      = 0.00134354611117262204;
   Real ExpectedLaplaceErrorLInf = 0.00113090174765822192;
   Real ExpectedLaplaceErrorL2   = 0.00134324628763667661;

   KOKKOS_FUNCTION Real vectorX(Real X, Real Y) const {
      return std::sin(2 * Pi * X / Lx) * std::cos(2 * Pi * Y / Ly);
   }

   KOKKOS_FUNCTION Real vectorY(Real X, Real Y) const {
      return std::cos(2 * Pi * X / Lx) * std::sin(2 * Pi * Y / Ly);
   }

   KOKKOS_FUNCTION Real divergence(Real X, Real Y) const {
      return 2 * Pi * (1. / Lx + 1. / Ly) * std::cos(2 * Pi * X / Lx) *
             std::cos(2 * Pi * Y / Ly);
   }

   KOKKOS_FUNCTION Real scalar(Real X, Real Y) const {
      return std::sin(2 * Pi * X / Lx) * std::sin(2 * Pi * Y / Ly);
   }

   KOKKOS_FUNCTION Real gradX(Real X, Real Y) const {
      return 2 * Pi / Lx * std::cos(2 * Pi * X / Lx) *
             std::sin(2 * Pi * Y / Ly);
   }
   KOKKOS_FUNCTION Real gradY(Real X, Real Y) const {
      return 2 * Pi / Ly * std::sin(2 * Pi * X / Lx) *
             std::cos(2 * Pi * Y / Ly);
   }

   KOKKOS_FUNCTION Real curl(Real X, Real Y) const {
      return 2 * Pi * (-1. / Lx + 1. / Ly) * std::sin(2 * Pi * X / Lx) *
             std::sin(2 * Pi * Y / Ly);
   }

   KOKKOS_FUNCTION Real laplaceVecX(Real X, Real Y) const {
      return -4 * Pi * Pi * (1. / Lx / Lx + 1. / Ly / Ly) *
             std::sin(2 * Pi * X / Lx) * std::cos(2 * Pi * Y / Ly);
   }

   KOKKOS_FUNCTION Real laplaceVecY(Real X, Real Y) const {
      return -4 * Pi * Pi * (1. / Lx / Lx + 1. / Ly / Ly) *
             std::cos(2 * Pi * X / Lx) * std::sin(2 * Pi * Y / Ly);
   }

   KOKKOS_FUNCTION Real layerThick(Real X, Real Y) const {
      return 2. + std::sin(2 * Pi * X / Lx) * std::cos(2 * Pi * Y / Ly);
   }

   KOKKOS_FUNCTION Real planetaryVort(Real X, Real Y) const {
      return std::cos(2 * Pi * X / Lx) * std::cos(2 * Pi * Y / Ly);
   }

   KOKKOS_FUNCTION Real normRelVort(Real X, Real Y) const {
      return curl(X, Y) / layerThick(X, Y);
   }

   KOKKOS_FUNCTION Real normPlanetVort(Real X, Real Y) const {
      return planetaryVort(X, Y) / layerThick(X, Y);
   }

   KOKKOS_FUNCTION Real thickFluxX(Real X, Real Y) const {
      return layerThick(X, Y) * vectorX(X, Y);
   }

   KOKKOS_FUNCTION Real thickFluxY(Real X, Real Y) const {
      return layerThick(X, Y) * vectorY(X, Y);
   }
/*
   KOKKOS_FUNCTION Real layerThickness(Real X, Real Y) const {
      return 2 + std::cos(2 * Pi * X / Lx) * std::cos(2 * Pi * Y / Ly);
   }

   KOKKOS_FUNCTION Real velocityX(Real X, Real Y) const {
      return std::sin(2 * Pi * X / Lx) * std::cos(2 * Pi * Y / Ly);
   }

   KOKKOS_FUNCTION Real velocityY(Real X, Real Y) const {
      return std::cos(2 * Pi * X / Lx) * std::sin(2 * Pi * Y / Ly);
   }
   
   KOKKOS_FUNCTION Real thickFluxDiv(Real X, Real Y) const {
      return 2 * Pi * (2 * (1 / Lx + 1 / Ly) * std::cos(2 * Pi * X / Lx) *
             std::cos(2 * Pi * Y / Ly) + (1 / Lx + 1 / Ly) * 
             std::pow(std::cos(2 * Pi * X / Lx), 2) *
             std::pow(std::cos(2 * Pi * Y / Ly), 2) -
             std::pow(std::sin(2 * Pi * X / Lx), 2) *
             std::pow(std::cos(2 * Pi * Y / Ly), 2) / Lx -
             std::pow(std::sin(2 * Pi * Y / Ly), 2) *
             std::pow(std::cos(2 * Pi * X / Lx), 2) / Ly);
   }
*/
}; // end TestSetupPlane

struct TestSetupSphere {
   // radius of spherical mesh
   // TODO: get this from the mesh
   Real Radius = 6371220;

   KOKKOS_FUNCTION Real vectorX(Real Lon, Real Lat) const {
      return -Radius * std::pow(std::sin(Lon), 2) * std::pow(std::cos(Lat), 3);
   }

   KOKKOS_FUNCTION Real vectorY(Real Lon, Real Lat) const {
      return -4 * Radius * std::sin(Lon) * std::cos(Lon) *
             std::pow(std::cos(Lat), 3) * std::sin(Lat);
   }

   KOKKOS_FUNCTION Real divergence(Real Lon, Real Lat) const {
      return std::sin(Lon) * std::cos(Lon) * std::pow(std::cos(Lat), 2) *
             (20 * std::pow(std::sin(Lat), 2) - 6);
   }

   KOKKOS_FUNCTION Real scalar(Real Lon, Real Lat) const {
      return Radius * std::cos(Lon) * std::pow(std::cos(Lat), 4);
   }

   KOKKOS_FUNCTION Real gradX(Real Lon, Real Lat) const {
      return -std::sin(Lon) * std::pow(std::cos(Lat), 3);
   }

   KOKKOS_FUNCTION Real gradY(Real Lon, Real Lat) const {
      return -4 * std::cos(Lon) * std::pow(std::cos(Lat), 3) * std::sin(Lat);
   }


   KOKKOS_FUNCTION Real curl(Real Lon, Real Lat) const {
      return -4 * std::pow(std::cos(Lon), 2) * std::pow(std::cos(Lat), 2) *
             std::sin(Lat);
   }

   KOKKOS_FUNCTION Real laplaceVecX(Real Lon, Real Lat) const {
      return 2 * std::cos(Lat) * (10 * std::pow(std::sin(Lat), 2) *
             (std::pow(std::cos(Lon), 2) - std::pow(std::sin(Lon), 2)) -
             6 * std::pow(std::cos(Lat), 2) * std::pow(std::cos(Lon), 2) +
             2 * std::pow(std::sin(Lon), 2) + 1) / Radius;
   }

   KOKKOS_FUNCTION Real laplaceVecY(Real Lon, Real Lat) const {
      return 4 * std::sin(Lon) * std::cos(Lon) * std::sin(Lat) *
            std::cos(Lon) * (20 * std::pow(std::cos(Lat), 2) - 9) / Radius;
   }

}; // end TestSetupSphere

#ifdef TEND_TEST_PLANE
constexpr Geometry Geom          = Geometry::Planar;
constexpr char DefaultMeshFile[] = "OmegaPlanarMesh.nc";
using TestSetup                  = TestSetupPlane;
#else
constexpr Geometry Geom          = Geometry::Spherical;
constexpr char DefaultMeshFile[] = "OmegaSphereMesh.nc";
using TestSetup                  = TestSetupSphere;
#endif

/*
int initState(const Array2DReal &LayerThickCell,
              const Array2DReal &NormalVelEdge, HorzMesh *Mesh,
              int NVertLevels) {
   int Err = 0;

   TestSetup Setup;

   Err += setScalar(
       KOKKOS_LAMBDA(Real X, Real Y) { return Setup.layerThickness(X, Y); },
       LayerThickCell, Geom, Mesh, OnCell, NVertLevels);

   Err += setVectorEdge(
       KOKKOS_LAMBDA(Real(&VecField)[2], Real X, Real Y) {
          VecField[0] = Setup.velocityX(X, Y);
          VecField[1] = Setup.velocityY(X, Y);
       },
       NormalVelEdge, EdgeComponent::Normal, Geom, Mesh, NVertLevels);


   // need to override FVertex with prescribed values
   // cannot use setScalar because it doesn't support setting 1D arrays
   const auto &FVertex = Mesh->FVertex;

   auto XVertex = createDeviceMirrorCopy(Mesh->XVertexH);
   auto YVertex = createDeviceMirrorCopy(Mesh->YVertexH);

   auto LonVertex = createDeviceMirrorCopy(Mesh->LonVertexH);
   auto LatVertex = createDeviceMirrorCopy(Mesh->LatVertexH);

   parallelFor(
       {Mesh->NVerticesOwned}, KOKKOS_LAMBDA(int IVertex) {
          if (Geom == Geometry::Planar) {
             const Real XV    = XVertex(IVertex);
             const Real YV    = YVertex(IVertex);
             FVertex(IVertex) = Setup.planetaryVorticity(XV, YV);
          } else {
             const Real XV    = LonVertex(IVertex);
             const Real YV    = LatVertex(IVertex);
             FVertex(IVertex) = Setup.planetaryVorticity(XV, YV);
          }
       });

   auto MyHalo    = Halo::getDefault();
   auto &FVertexH = Mesh->FVertexH;
   Err += MyHalo->exchangeFullArrayHalo(FVertexH, OnVertex);
   deepCopy(FVertex, FVertexH);


   return Err;
}
*/

int testThickFluxDiv(int NVertLevels, Real RTol) {

   int Err = 0;
   TestSetup Setup;

   const auto Mesh = HorzMesh::getDefault();
   // TODO: implement config, dummy config for now
   Config *TendConfig;

   // Compute exact result
   Array2DReal ExactThickFluxDiv("ExactThickFluxDiv", Mesh->NCellsOwned,
                                 NVertLevels);

   Err += setScalar(
       KOKKOS_LAMBDA(Real X, Real Y) { return -Setup.divergence(X,Y); },
       ExactThickFluxDiv, Geom, Mesh, OnCell, NVertLevels, false);

   // Set input array
   Array2DR8 ThickFluxEdge("ThickFluxEdge", Mesh->NEdgesSize, NVertLevels);

   Err += setVectorEdge(
       KOKKOS_LAMBDA(Real(&VecField)[2], Real X, Real Y) {
          VecField[0] = Setup.vectorX(X, Y);
          VecField[1] = Setup.vectorY(X, Y);
       },
       ThickFluxEdge, EdgeComponent::Normal, Geom, Mesh, NVertLevels);

   // Compute numerical result
   Array2DReal NumThickFluxDiv("NumThickFluxDiv", Mesh->NCellsOwned,
                               NVertLevels);
   ThicknessFluxDivOnCell ThickFluxDivOnC(Mesh, TendConfig);
   parallelFor(
      {Mesh->NCellsOwned, NVertLevels}, KOKKOS_LAMBDA(int ICell, int KLevel) {
         ThickFluxDivOnC(NumThickFluxDiv, ICell, KLevel, ThickFluxEdge);
      });

   // Compute errors
   ErrorMeasures TFDivErrors;
   Err += computeErrors(TFDivErrors, NumThickFluxDiv, ExactThickFluxDiv, Mesh,
                        OnCell, NVertLevels);

   // Check error values
//   if (!isApprox(TFDivErrors.LInf, Setup.ExpectedDivErrorLInf, RTol)) {
//      Err++;
//      LOG_ERROR("TendTest: ThickFluxDiv LInf FAIL");
//   }
//
//   if (!isApprox(TFDivErrors.L2, Setup.ExpectedDivErrorL2, RTol)) {
//      Err++;
//      LOG_ERROR("TendTest: ThickFluxDiv L2 FAIL");
//   }
//
//   if (Err == 0) {
//      LOG_INFO("TendTest: ThickFluxDiv PASS");
//   }

   std::cout << std::setprecision(20);
   std::cout << "TFErrors:  ";
   std::cout << TFDivErrors.L2 << " " << TFDivErrors.LInf << std::endl;

   return Err;
} // end testThickFluxDiv

int testPotVortFlux(int NVertLevels, Real RTol) {

   int Err = 0;
   TestSetup Setup;

   const auto Mesh = HorzMesh::getDefault();
   // TODO: implement config, dummy config for now
   Config *TendConfig;
/*
   // Compute exact result
   Array2DReal ExactPotVortFlux("ExactPotVortFlux", Mesh->NEdgesOwned,
                                NVertLevels);

   Err += setVectorEdge(
       KOKKOS_LAMBDA(Real(&VecField)[2], Real X, Real Y) {
          VecField[0] = (Setup.normRelVort(X, Y) + Setup.normPlanetVort(X, Y)) *
                        Setup.thickFluxX(X, Y);
          VecField[1] = (Setup.normRelVort(X, Y) + Setup.normPlanetVort(X, Y)) *
                        Setup.thickFluxY(X, Y);
       },
       ExactPotVortFlux, EdgeComponent::Normal, Geom, Mesh, NVertLevels, false);

   // Set input arrays
   Array2DR8 NormRelVortEdge("NormRelVortEdge", Mesh->NEdgesSize, NVertLevels);

   Err += setScalar(
       KOKKOS_LAMBDA(Real X, Real Y) { return Setup.normRelVort(X, Y); },
       NormRelVortEdge, Geom, Mesh, OnEdge, NVertLevels);

   Array2DR8 NormPlanetVortEdge("NormPlanetVortEdge", Mesh->NEdgesSize,
                                NVertLevels);
   Err += setScalar(
       KOKKOS_LAMBDA(Real X, Real Y) { return Setup.normPlanetVort(X,Y); },
       NormPlanetVortEdge, Geom, Mesh, OnEdge, NVertLevels);

   Array2DR8 ThickFluxEdge("ThickFluxEdge", Mesh->NEdgesSize, NVertLevels);

   Err += setVectorEdge(
       KOKKOS_LAMBDA(Real(&VecField)[2], Real X, Real Y) {
          VecField[0] = Setup.thickFluxX(X, Y);
          VecField[1] = Setup.thickFluxY(X, Y);
       },
       ThickFluxEdge, EdgeComponent::Normal, Geom, Mesh, NVertLevels);

   Array2DR8 NormVelEdge("NormVelEdge", Mesh->NEdgesSize, NVertLevels);

   Err += setVectorEdge(
       KOKKOS_LAMBDA(Real(&VecField)[2], Real X, Real Y) {
          VecField[0] = Setup.vectorX(X, Y);
          VecField[1] = Setup.vectorY(X, Y);
       },
       NormVelEdge, EdgeComponent::Normal, Geom, Mesh, NVertLevels);

   // Compute numerical result
   Array2DReal NumPotVortFlux("NumPotVortFlux", Mesh->NEdgesOwned, NVertLevels);

   PotentialVortFluxOnEdge PotVortFluxOnE(Mesh, TendConfig);
   parallelFor(
      {Mesh->NEdgesOwned, NVertLevels}, KOKKOS_LAMBDA(int IEdge, int KLevel) {
         PotVortFluxOnE(NumPotVortFlux, IEdge, KLevel, NormRelVortEdge,
                        NormPlanetVortEdge, ThickFluxEdge, NormVelEdge);
      });

   // Compute errors
   ErrorMeasures PotVortFluxErrors;
   Err += computeErrors(PotVortFluxErrors, NumPotVortFlux, ExactPotVortFlux,
                        Mesh, OnEdge, NVertLevels);


   std::cout << std::setprecision(20);
   std::cout << "PotVortFluxErrors:  ";
   std::cout << PotVortFluxErrors.L2 << " " << PotVortFluxErrors.LInf <<
                std::endl;
*/
} // end testPotVortFlux

int testKEGrad(int NVertLevels, Real RTol) {

//   FILE *ofptr1, *ofptr2, *ofptr3;
//   char outname[24];
//
//   const auto *DefEnv = MachEnv::getDefaultEnv();
//   I4 MyTask = DefEnv->getMyTask();
//   sprintf(outname, "exout_%d.txt", MyTask);
//   ofptr1 = fopen(outname, "w");
//   sprintf(outname, "keout_%d.txt", MyTask);
//   ofptr2 = fopen(outname, "w");
//   sprintf(outname, "nuout_%d.txt", MyTask);
//   ofptr3 = fopen(outname, "w");

   int Err = 0;
   TestSetup Setup;

   const auto Mesh = HorzMesh::getDefault();
   // TODO: implement config, dummy config for now
   Config *TendConfig;

   // Compute exact result
   Array2DReal ExactKEGrad("ExactKEGrad", Mesh->NEdgesOwned, NVertLevels);

   Err += setVectorEdge(
       KOKKOS_LAMBDA(Real(&VecField)[2], Real X, Real Y) {
          VecField[0] = -Setup.gradX(X, Y);
          VecField[1] = -Setup.gradY(X, Y);
       },
       ExactKEGrad, EdgeComponent::Normal, Geom, Mesh, NVertLevels, false);


//   auto KEGradH = createHostMirrorCopy(ExactKEGrad);
//   for(int IEdge = 0; IEdge < Mesh->NEdgesOwned; ++IEdge) {
//      for(int K = 0; K < NVertLevels; ++K) {
//         fprintf(ofptr1, "%8d     %24.15e", IEdge, KEGradH(IEdge, K));
//      }
//      fprintf(ofptr1, "\n");
//   }
//   fclose(ofptr1);


   // Set input array
   Array2DReal KECell("KECell", Mesh->NCellsSize, NVertLevels);

   Err += setScalar(
       KOKKOS_LAMBDA(Real X, Real Y) { return Setup.scalar(X,Y); },
       KECell, Geom, Mesh, OnCell, NVertLevels);

//   auto KECellH = createHostMirrorCopy(KECell);
//   for(int ICell = 0; ICell < Mesh->NCellsSize; ++ICell) {
//      for(int K = 0; K < NVertLevels; ++K) {
//         fprintf(ofptr2, "%8d     %24.15e", ICell, KECellH(ICell, K));
//      }
//      fprintf(ofptr2,"\n");
//   }
//   fclose(ofptr2);

   // Compute numerical result
   Array2DReal NumKEGrad("NumKEGrad", Mesh->NEdgesOwned, NVertLevels);

   KEGradOnEdge KEGradOnE(Mesh, TendConfig);
   parallelFor(
      {Mesh->NEdgesOwned, NVertLevels}, KOKKOS_LAMBDA(int IEdge, int KLevel) {
         KEGradOnE(NumKEGrad, IEdge, KLevel, KECell);
      });

//   auto NumGradH = createHostMirrorCopy(NumKEGrad);
//   for(int IEdge = 0; IEdge < Mesh->NEdgesOwned; ++IEdge) {
//      for(int K = 0; K < NVertLevels; ++K) {
//         fprintf(ofptr3, "%8d     %24.15e", IEdge, NumGradH(IEdge, K));
//      }
//      fprintf(ofptr3, "\n");
//   }
//   fclose(ofptr3);

   // Compute errors
   ErrorMeasures KEGradErrors;
   Err += computeErrors(KEGradErrors, NumKEGrad, ExactKEGrad, Mesh, OnEdge,
                        NVertLevels);

   // Check error values
//   if (!isApprox(KEGradErrors.LInf, Setup.ExpectedGradErrorLInf, RTol)) {
//      Err++;
//      LOG_ERROR("TendTest: KEGrad LInf FAIL");
//   }
//
//   if (!isApprox(KEGradErrors.L2, Setup.ExpectedGradErrorL2, RTol)) {
//      Err++;
//      LOG_ERROR("TendTest: KEGrad L2 FAIL");
//   }
//
//   if (Err == 0) {
//      LOG_INFO("TendTest: KEGrad PASS");
//   }

   std::cout << std::setprecision(20);
   std::cout << "KEErrors:  ";
   std::cout << KEGradErrors.L2 << " " << KEGradErrors.LInf << std::endl;

   return Err;
} // end testKEGrad 

int testSSHGrad(int NVertLevels, Real RTol) {

   int Err = 0;
   TestSetup Setup;

   const auto Mesh = HorzMesh::getDefault();
   // TODO: implement config, dummy config for now
   Config *TendConfig;

   // Compute exact result
   Array2DReal ExactSSHGrad("ExactSSHGrad", Mesh->NEdgesOwned, NVertLevels);

   Err += setVectorEdge(
       KOKKOS_LAMBDA(Real(&VecField)[2], Real X, Real Y) {
          VecField[0] = -Setup.gradX(X, Y);
          VecField[1] = -Setup.gradY(X, Y);
       },
       ExactSSHGrad, EdgeComponent::Normal, Geom, Mesh, NVertLevels, false);

   // Set input array
   Array2DReal SSHCell("SSHCell", Mesh->NCellsSize, NVertLevels);

   Err += setScalar(
       KOKKOS_LAMBDA(Real X, Real Y) { return Setup.scalar(X,Y); },
       SSHCell, Geom, Mesh, OnCell, NVertLevels);

   // Compute numerical result
   Array2DReal NumSSHGrad("NumSSHGrad", Mesh->NEdgesOwned, NVertLevels);

   SSHGradOnEdge SSHGradOnE(Mesh, TendConfig);
   parallelFor(
      {Mesh->NEdgesOwned, NVertLevels}, KOKKOS_LAMBDA(int IEdge, int KLevel) {
         SSHGradOnE(NumSSHGrad, IEdge, KLevel, SSHCell);
      });

   // Compute errors
   ErrorMeasures SSHGradErrors;
   Err += computeErrors(SSHGradErrors, NumSSHGrad, ExactSSHGrad, Mesh, OnEdge,
                        NVertLevels);

   // Check error values
//   if (!isApprox(SSHGradErrors.LInf, Setup.ExpectedGradErrorLInf, RTol)) {
//      Err++;
//      LOG_ERROR("TendTest: SSHGrad LInf FAIL");
//   }
//
//   if (!isApprox(SSHGradErrors.L2, Setup.ExpectedGradErrorL2, RTol)) {
//      Err++;
//      LOG_ERROR("TendTest: SSHGrad L2 FAIL");
//   }
//
//   if (Err == 0) {
//      LOG_INFO("TendTest: SSHGrad PASS");
//   }

   std::cout << std::setprecision(20);
   std::cout << "SSHErrors:  ";
   std::cout << SSHGradErrors.L2 << " " << SSHGradErrors.LInf << std::endl;

   return Err;
} // end testSSHGrad 

int testVelDiff(int NVertLevels, Real RTol) {

   int Err = 0;
   TestSetup Setup;

   const auto Mesh = HorzMesh::getDefault();
   // TODO: implement config, dummy config for now
   Config *TendConfig;

   // TODO: move to Mesh constructor
   Mesh->setMasks(NVertLevels);

   // Compute exact result
   Array2DReal ExactVelDiff("ExactVelDiff", Mesh->NEdgesOwned, NVertLevels);

   Err += setVectorEdge(
       KOKKOS_LAMBDA(Real(&VecField)[2], Real X, Real Y) {
          VecField[0] = Setup.laplaceVecX(X, Y);
          VecField[1] = Setup.laplaceVecY(X, Y);
       },
       ExactVelDiff, EdgeComponent::Normal, Geom, Mesh, NVertLevels, false);

   // Set input arrays
   Array2DReal DivCell("DivCell", Mesh->NCellsSize, NVertLevels);

   Err += setScalar(
       KOKKOS_LAMBDA(Real X, Real Y) { return Setup.divergence(X,Y); },
       DivCell, Geom, Mesh, OnCell, NVertLevels);

   Array2DReal RVortVertex("RVortVertex", Mesh->NVerticesSize, NVertLevels);

   Err += setScalar(
       KOKKOS_LAMBDA(Real X, Real Y) { return Setup.curl(X,Y); },
       RVortVertex, Geom, Mesh, OnVertex, NVertLevels);

   // Compute numerical result
   Array2DReal NumVelDiff("NumVelDiff", Mesh->NEdgesOwned, NVertLevels);

   VelocityDiffusionOnEdge VelDiffOnE(Mesh, TendConfig);
   parallelFor(
      {Mesh->NEdgesOwned, NVertLevels}, KOKKOS_LAMBDA(int IEdge, int KLevel) {
         VelDiffOnE(NumVelDiff, IEdge, KLevel, DivCell, RVortVertex);
      });

   // Compute errors
   ErrorMeasures VelDiffErrors;
   Err += computeErrors(VelDiffErrors, NumVelDiff, ExactVelDiff, Mesh, OnEdge,
                        NVertLevels);

   // Check error values
//   if (!isApprox(VelDiffErrors.LInf, Setup.ExpectedLaplaceErrorLInf, RTol)) {
//      Err++;
//      LOG_ERROR("TendTest: VelDiff LInf FAIL");
//   }
//
//   if (!isApprox(VelDiffErrors.L2, Setup.ExpectedLaplaceErrorL2, RTol)) {
//      Err++;
//      LOG_ERROR("TendTest: VelDiff L2 FAIL");
//   }
//
//   if (Err == 0) {
//      LOG_INFO("TendTest: VelDiff PASS");
//   }

   std::cout << std::setprecision(20);
   std::cout << "VelDiffErrors:  ";
   std::cout << VelDiffErrors.L2 << " " << VelDiffErrors.LInf << std::endl;

   return Err;
} // end testVelDiff

int testVelHyperDiff(int NVertLevels, Real RTol) {

   int Err = 0;
   TestSetup Setup;

   const auto Mesh = HorzMesh::getDefault();
   // TODO: implement config, dummy config for now
   Config *TendConfig;
/*
   // TODO: move to Mesh constructor
   Mesh->setMasks(NVertLevels);

   // Compute exact result
   Array2DReal ExactVelHyperDiff("ExactVelHyperDiff", Mesh->NEdgesOwned,
                                 NVertLevels);

   Err += setVectorEdge(
       KOKKOS_LAMBDA(Real(&VecField)[2], Real X, Real Y) {
          VecField[0] = -Setup.laplaceVecX(X, Y);
          VecField[1] = -Setup.laplaceVecY(X, Y);
       },
       ExactVelHyperDiff, EdgeComponent::Normal, Geom, Mesh, NVertLevels,
       false);

   // Set input arrays
   Array2DReal DivCell("DivCell", Mesh->NCellsSize, NVertLevels);

   Err += setScalar(
       KOKKOS_LAMBDA(Real X, Real Y) { return Setup.divergence(X,Y); },
       DivCell, Geom, Mesh, OnCell, NVertLevels);

   Array2DReal RVortVertex("RVortVertex", Mesh->NVerticesSize, NVertLevels);

   Err += setScalar(
       KOKKOS_LAMBDA(Real X, Real Y) { return Setup.curl(X,Y); },
       RVortVertex, Geom, Mesh, OnVertex, NVertLevels);

   // Compute numerical result
   Array2DReal NumVelHyperDiff("NumVelHyperDiff", Mesh->NEdgesOwned,
                               NVertLevels);

   VelocityHyperDiffOnEdge VelHyperDiffOnE(Mesh, TendConfig);
   parallelFor(
      {Mesh->NEdgesOwned, NVertLevels}, KOKKOS_LAMBDA(int IEdge, int KLevel) {
         VelHyperDiffOnE(NumVelHyperDiff, IEdge, KLevel, DivCell, RVortVertex);
      });

   // Compute errors
   ErrorMeasures VelHyperDiffErrors;
   Err += computeErrors(VelHyperDiffErrors, NumVelHyperDiff, ExactVelHyperDiff,
                        Mesh, OnEdge, NVertLevels);

   // Check error values
//   if (!isApprox(VelHyperDiffErrors.LInf, Setup.ExpectedLaplaceErrorLInf,
//       RTol)) {
//      Err++;
//      LOG_ERROR("TendTest: VelHyperDiff LInf FAIL");
//   }
//
//   if (!isApprox(VelHyperDiffErrors.L2, Setup.ExpectedLaplaceErrorL2, RTol)) {
//      Err++;
//      LOG_ERROR("TendTest: VelHyperDiff L2 FAIL");
//   }
//
//   if (Err == 0) {
//      LOG_INFO("TendTest: VelHyperDiff PASS");
//   }

   std::cout << std::setprecision(20);
   std::cout << "VelHyperDiffErrors:  ";
   std::cout << VelHyperDiffErrors.L2 << " " << VelHyperDiffErrors.LInf << std::endl;
*/
   return Err;
} // end testVelHyperDiff


int initTendTest(const std::string &mesh) {

   I4 Err = 0;

   MachEnv::init(MPI_COMM_WORLD);
   MachEnv *DefEnv  = MachEnv::getDefaultEnv();
   MPI_Comm DefComm = DefEnv->getComm();

   I4 IOErr = IO::init(DefComm);
   if (IOErr != 0) {
      Err++;
      LOG_ERROR("AuxVarsTest: error initializing parallel IO");
   }

   int DecompErr = Decomp::init(mesh);
   if (DecompErr != 0) {
      Err++;
      LOG_ERROR("AuxVarsTest: error initializing default decomposition");
   }

   int HaloErr = Halo::init();
   if (HaloErr != 0) {
      Err++;
      LOG_ERROR("AuxVarsTest: error initializing default halo");
   }

   int MeshErr = HorzMesh::init();
   if (MeshErr != 0) {
      Err++;
      LOG_ERROR("AuxVarsTest: error initializing default mesh");
   }

   return Err;
} // end initTendTest

void finalizeTendTest() {
   HorzMesh::clear();
   Halo::clear();
   Decomp::clear();
   MachEnv::removeAll();
} // end finalizeTendTest

void tendencyTermsTest(const std::string &mesh = DefaultMeshFile) {
   int Err = initTendTest(mesh);
   if (Err != 0) {
      LOG_CRITICAL("TendTest: Error initializing");
   }

   const auto &Mesh = HorzMesh::getDefault();
   int NVertLevels  = 16;

//   Array2DReal LayerThickCell("LayerThickCell", Mesh->NCellsSize, NVertLevels);
//   Array2DReal NormalVelEdge("NormalVelEdge", Mesh->NEdgesSize, NVertLevels);

//   Err += initState(LayerThickCell, NormalVelEdge, Mesh, NVertLevels);

   const Real RTol = sizeof(Real) == 4 ? 1e-2 : 1e-10;

   Err += testThickFluxDiv(NVertLevels, RTol);

   Err += testPotVortFlux(NVertLevels, RTol);

   Err += testKEGrad(NVertLevels, RTol);

   Err += testSSHGrad(NVertLevels, RTol);

   Err += testVelDiff(NVertLevels, RTol);

   Err += testVelHyperDiff(NVertLevels, RTol);

   if (Err == 0) {
      LOG_INFO("TendTest: Successful completion");
   }
   finalizeTendTest();
} // end tendencyTermsTest

int main(int argc, char *argv[]) {
   MPI_Init(&argc, &argv);
   Kokkos::initialize(argc, argv);

   tendencyTermsTest();

   Kokkos::finalize();
   MPI_Finalize();

} // end of main
//===-----------------------------------------------------------------------===/
