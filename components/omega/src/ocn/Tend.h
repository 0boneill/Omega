#ifndef OMEGA_TEND_H
#define OMEGA_TEND_H

#include "Config.h"
#include "HorzMesh.h"
#include "MachEnv.h"
#include "OceanState.h"

namespace OMEGA {

class ThicknessFluxDivergenceOnCell {
 public:
 
   bool Enabled = false;
 
   ThicknessFluxDivergenceOnCell(const HorzMesh *Mesh, Config *Options);
  
   KOKKOS_FUNCTION void operator()(Array2DReal &Tend,
                                   I4 ICell,
                                   I4 KChunk,
                                   const Array2DR8 &ThicknessFlux
                                  ) const {

      const Real InvAreaCell = 1. / AreaCell(ICell);
      for (int J = 0; J < NEdgesOnCell(ICell); ++J) {
         const I4 JEdge = EdgesOnCell(ICell, J);
         for (int K = KChunk * VecLength; K < (KChunk + 1) * VecLength; ++K) {
            Tend(ICell, K) -= DvEdge(JEdge) * EdgeSignOnCell(ICell, J) *
                 ThicknessFlux(JEdge, K) * InvAreaCell;
         }
      }

   }

 private:
   Array1DI4 NEdgesOnCell;
   Array2DI4 EdgesOnCell;
   Array1DR8 DvEdge;
   Array1DR8 AreaCell;
   Array2DR8 EdgeSignOnCell;

};

class PotentialVortFluxOnEdge {
 public:

   bool Enabled = false;

   PotentialVortFluxOnEdge(const HorzMesh *Mesh, Config *Options);

   KOKKOS_FUNCTION void operator()(Array2DReal &Tend,
                                   I4 IEdge,
                                   I4 KChunk,
                                   const Array2DR8 &NormRVortEdge,
                                   const Array2DR8 &NormFEdge,
                                   const Array2DR8 &HFluxEdge,
                                   const Array2DR8 &VNEdge
                                   ) const {

      for (int J = 0; J < NEdgesOnEdge(IEdge); ++J) {
      
         I4 JEdge = EdgesOnEdge(IEdge, J);
         for (int K = KChunk * VecLength; K < (KChunk + 1) * VecLength; ++K) {
             Real NormVort = (NormRVortEdge(IEdge, K) + NormFEdge(IEdge, K) +
                              NormRVortEdge(JEdge, K) + NormFEdge(JEdge, K)) * 0.5;

             Tend(IEdge, K) += WeightsOnEdge(IEdge, K) * HFluxEdge(JEdge, K) *
                               VNEdge(JEdge, K) * NormVort;

         }
      }
   }

 private:
   Array1DI4 NEdgesOnEdge;
   Array2DI4 EdgesOnEdge;
   Array2DR8 WeightsOnEdge;

};

class KineticEnergyGradOnEdge {
 public:

   bool Enabled = false;

   KineticEnergyGradOnEdge(const HorzMesh *Mesh, Config *Options);

   KOKKOS_FUNCTION void operator()(Array2DReal &Tend,
                                   I4 IEdge,
                                   I4 KChunk,
                                   const Array2DR8 &KECell
                                   ) const {

      const I4 ICell0 = CellsOnEdge(IEdge, 0);
      const I4 ICell1 = CellsOnEdge(IEdge, 1);
      const Real InvDcEdge = 1. / DcEdge(IEdge);

      for (int K = KChunk * VecLength; K < (KChunk + 1) * VecLength; ++K) {
         Tend(IEdge, K) -= (KECell(ICell1, K) - KECell(ICell0, K)) * InvDcEdge;
      }

   }

 private:

   Array2DI4 CellsOnEdge;
   Array1DR8 DcEdge;


};

class SSHGradOnEdge {
 public:

   bool Enabled = false;

   SSHGradOnEdge(const HorzMesh *Mesh, Config *Options);

   KOKKOS_FUNCTION void operator()(Array2DReal &Tend,
                                   I4 IEdge,
                                   I4 KChunk,
                                   const Array2DR8 &HCell
                                   ) const {
      
      const I4 ICell0 = CellsOnEdge(IEdge, 0);
      const I4 ICell1 = CellsOnEdge(IEdge, 1);
      const Real InvDcEdge = 1. / DcEdge(IEdge);

      for (int K = KChunk * VecLength; K < (KChunk + 1) * VecLength; ++K) {
         Tend(IEdge, K) -= Grav * (HCell(ICell1, K) - HCell(ICell0, K)) * InvDcEdge;
      }
}

 private:
   R8 Grav;
   Array2DI4 CellsOnEdge;
   Array1DR8 DcEdge;
    

};

class VelocityDiffusionOnEdge {
 public:

   bool Enabled = false;

   VelocityDiffusionOnEdge(const HorzMesh *Mesh, Config *Options);

   KOKKOS_FUNCTION void operator()(Array2DReal &Tend,
                                   I4 IEdge,
                                   I4 KChunk,
                                   const Array2DR8 &DivCell,
                                   const Array2DR8 &RVortVertex
                                   ) const {

      const I4 ICell0 = CellsOnEdge(IEdge, 0);
      const I4 ICell1 = CellsOnEdge(IEdge, 1);

      const I4 IVertex0 = VerticesOnEdge(IEdge, 0);
      const I4 IVertex1 = VerticesOnEdge(IEdge, 1);

      const Real DcEdgeInv = 1. / DcEdge(IEdge);
      const Real DvEdgeInv = 1. / DvEdge(IEdge);

      for (int K = KChunk * VecLength; K < (KChunk + 1) * VecLength; ++K) {
         const Real Del2U = ((DivCell(ICell1, K) - DivCell(ICell0, K)) * DcEdgeInv -
                            (RVortVertex(IVertex1, K) - RVortVertex(IVertex0, K)) 
                            * DvEdgeInv);

         Tend(IEdge, K) += EdgeMask(IEdge, K) * ViscDel2 * MeshScalingDel2(IEdge) *
                          Del2U;

      }

   }

 private:
   R8 ViscDel2;
   Array2DI4 CellsOnEdge;
   Array2DI4 VerticesOnEdge;
   Array1DR8 DcEdge;
   Array1DR8 DvEdge;
   Array1DR8 MeshScalingDel2;
   Array2DR8 EdgeMask;

};

class VelocityHyperDiffusionOnEdge {
 public: 

   bool Enabled = false;

   VelocityHyperDiffusionOnEdge(const HorzMesh *Mesh, Config *Options);

   KOKKOS_FUNCTION void operator()(Array2DReal &Tend,
                                   I4 IEdge,
                                   I4 KChunk,
                                   const Array2DR8 &Del2DivCell,
                                   const Array2DR8 &Del2RVortVertex
                                   ) const {

      const I4 ICell0 = CellsOnEdge(IEdge, 0);
      const I4 ICell1 = CellsOnEdge(IEdge, 1);

      const I4 IVertex0 = VerticesOnEdge(IEdge, 0);
      const I4 IVertex1 = VerticesOnEdge(IEdge, 1);

      const Real DcEdgeInv = 1. / DcEdge(IEdge);
      const Real DvEdgeInv = 1. / DvEdge(IEdge);

      for (int K = KChunk * VecLength; K < (KChunk + 1) * VecLength; ++K) {
         const Real Del2U = ((Del2DivCell(ICell1, K) - Del2DivCell(ICell0, K))
                             * DcEdgeInv -
                            (Del2RVortVertex(IVertex1, K)
                              - Del2RVortVertex(IVertex0, K)) 
                            * DvEdgeInv);

         Tend(IEdge, K) -= EdgeMask(IEdge, K) * ViscDel4 * MeshScalingDel4(IEdge) *
                          Del2U;

      }

   }

 private:
   Real ViscDel4;
   Array2DI4 CellsOnEdge;
   Array2DI4 VerticesOnEdge;
   Array1DR8 DcEdge;
   Array1DR8 DvEdge;
   Array1DR8 MeshScalingDel4;
   Array2DR8 EdgeMask;

};

class TracerHorzAdvOnCell {
 public:
   bool Enabled = false;

   TracerHorzAdvOnCell(const HorzMesh *Mesh, Config *Options);

   KOKKOS_FUNCTION void operator()(Array3DReal &Tend,
                                   I4 L,
                                   I4 ICell,
                                   I4 KChunk,
                                   const Array2DR8 &VEdge,
                                   const Array3DR8 &NormTrCell,
                                   const Array2DR8 &HFluxEdge
                                   ) const {

      Real InvAreaCell = 1. / AreaCell(ICell);

      for (int J = 0; J < NEdgesOnCell(ICell); ++J) {
         const I4 JEdge = EdgesOnCell(ICell, J);

         const I4 JCell0 = CellsOnEdge(JEdge, 0);
         const I4 JCell1 = CellsOnEdge(JEdge, 1);

         for (int K = KChunk * VecLength; K < (KChunk + 1) * VecLength; ++K) {
            const Real NormTrEdge = 
                (NormTrCell(L, JCell0, K) + NormTrCell(L, JCell1, K)) * 0.5;
            Tend(L, ICell, K) -= DvEdge(JEdge) * EdgeSignOnCell(ICell, J) *
                HFluxEdge(JEdge, K) * NormTrEdge * VEdge(JEdge, K) * InvAreaCell;
         }
      }
   }

 private:
   Array1DI4 NEdgesOnCell;
   Array2DI4 EdgesOnCell;
   Array2DI4 CellsOnEdge;
   Array2DR8 EdgeSignOnCell;
   Array1DR8 DvEdge;
   Array1DR8 AreaCell;

};

class TracerDiffusionOnCell {
 public:
   bool Enabled = false;

   TracerDiffusionOnCell(const HorzMesh *Mesh, Config *Options);

   KOKKOS_FUNCTION void operator()(Array3DReal &Tend,
                                   I4 L,
                                   I4 ICell,
                                   I4 KChunk,
                                   const Array3DR8 &NormTrCell,
                                   const Array2DR8 &HMeanEdge
//                                   const OceanAuxState *AuxState
                                   ) const {

      Real InvAreaCell = 1. / AreaCell(ICell);

      for (int J = 0; J < NEdgesOnCell(ICell); ++J) {
         const I4 JEdge = EdgesOnCell(ICell, J);

         const I4 JCell0 = CellsOnEdge(JEdge, 0);
         const I4 JCell1 = CellsOnEdge(JEdge, 1);

         const Real InvDcEdge = 1. / DcEdge(JEdge);

         for (int K = KChunk * VecLength; K < (KChunk + 1) * VecLength; ++K) {
            const Real GradTrEdge =
                (NormTrCell(L, JCell1, K) - NormTrCell(L, JCell0, K)) *
                InvDcEdge;

            Tend(L, ICell, K) += EddyDiff2 * DvEdge(JEdge) * EdgeSignOnCell(ICell, J) *
                                 HMeanEdge(JEdge, K) * MeshScalingDel2(JEdge) *
                                 GradTrEdge * InvAreaCell;

         }
      }
   }

 private:
   Real EddyDiff2;
   Array1DI4 NEdgesOnCell;
   Array2DI4 EdgesOnCell;
   Array2DI4 CellsOnEdge;
   Array2DR8 EdgeSignOnCell;
   Array1DR8 DvEdge;
   Array1DR8 DcEdge;
   Array1DR8 AreaCell;
   Array1DR8 MeshScalingDel2;

};

class TracerHyperDiffusionOnCell {
 public:
   bool Enabled = false;

   TracerHyperDiffusionOnCell(const HorzMesh *Mesh, Config *Options);

   KOKKOS_FUNCTION void operator()(Array3DReal &Tend,
                                   I4 L,
                                   I4 ICell,
                                   I4 KChunk,
                                   const Array3DR8 &TrDel2Cell
                                   ) const {

      Real InvAreaCell = 1. / AreaCell(ICell);

      for (int J = 0; J < NEdgesOnCell(ICell); ++J) {
         const I4 JEdge = EdgesOnCell(ICell, J);

         const I4 JCell0 = CellsOnEdge(JEdge, 0);
         const I4 JCell1 = CellsOnEdge(JEdge, 1);

         const Real InvDcEdge = 1. / DcEdge(JEdge);

         for (int K = KChunk * VecLength; K < (KChunk + 1) * VecLength; ++K) {
            const Real GradTrDel2Edge =
                (TrDel2Cell(L, JCell1, K) - TrDel2Cell(L, JCell0, K)) *
                InvDcEdge;

         Tend(L, ICell, K) -= EddyDiff4 * DvEdge(JEdge) * EdgeSignOnCell(ICell, J) *
                              MeshScalingDel4(JEdge) * GradTrDel2Edge * InvAreaCell;

         }
      }
   }

 private:
   Real EddyDiff4;
   Array1DI4 NEdgesOnCell;
   Array2DI4 EdgesOnCell;
   Array2DI4 CellsOnEdge;
   Array2DR8 EdgeSignOnCell;
   Array1DR8 DvEdge;
   Array1DR8 DcEdge;
   Array1DR8 AreaCell;
   Array1DR8 MeshScalingDel4;

};

} // namespace OMEGA
#endif
