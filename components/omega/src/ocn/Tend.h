#ifndef OMEGA_TEND_H
#define OMEGA_TEND_H

#include "Config.h"
#include "HorzMesh.h"
#include "OceanState.h"

namespace OMEGA {

class ThicknessFluxDivergenceOnCell {
 public:
 
   bool Enabled = false;
 
   ThicknessFluxDivergenceOnCell(const HorzMesh *Mesh, Config *Options);
  
   KOKKOS_INLINE_FUNCTION void operator()(I4 ICell,
                                          I4 KChunk,
                                          const OceanState *State,
//                                          const OceanAuxState *AuxState,
                                          Array2DReal &Tend) const {
      // dummy vars 
      Array2DR8 ThicknessFlux;
      I4 LevelsPerChunk;

      const Real InvAreaCell = 1. / AreaCell(ICell);
      for (int J = 0; J < NEdgesOnCell(ICell); ++J) {
         const I4 JEdge = EdgesOnCell(ICell, J);
         for (int K = KChunk * LevelsPerChunk; K < (KChunk + 1) * LevelsPerChunk; ++K) {
            Tend(ICell, K) -= DvEdge(JEdge) * EdgeSignOnCell(ICell, J) *
                 ThicknessFlux(JEdge, K) * InvAreaCell;
    // AuxState->ThicknessFlux(JEdge, K) * InvAreaCell;
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

   KOKKOS_INLINE_FUNCTION void operator()(I4 IEdge,
                                          I4 KChunk,
                                          const OceanState *State,
//                                         const OceaAuxState *AuxState,
                                          Array2DReal &Tend) {
      // dummy vars
      I4 LevelsPerChunk;
      Array2DR8 NormRVortEdge;
      Array2DR8 NormFEdge;
      Array2DR8 HFluxEdge;
      Array2DR8 VNEdge;

      for (int J = 0; J < NEdgesOnEdge(IEdge); ++J) {
      
         I4 JEdge = EdgesOnEdge(IEdge, J);
         for (int K = KChunk * LevelsPerChunk; K < (KChunk + 1) * LevelsPerChunk; ++K) {
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

   KOKKOS_INLINE_FUNCTION void operator()(I4 IEdge,
                                          I4 KChunk,
                                          const OceanState *State,  
//                                         const OceaAuxState *AuxState,
                                          Array2DReal &Tend) {
      // dummy vars
      I4 LevelsPerChunk;
      Array2DR8 KECell;

      const I4 ICell0 = CellsOnEdge(IEdge, 0);
      const I4 ICell1 = CellsOnEdge(IEdge, 1);
      const Real InvDcEdge = 1. / DcEdge(IEdge);

      for (int K = KChunk * LevelsPerChunk; K < (KChunk + 1) * LevelsPerChunk; ++K) {
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

   KOKKOS_INLINE_FUNCTION void operator()(I4 IEdge,
                                          I4 KChunk,
                                          const OceanState *State,
//                                          const OceanAuxState *AuxState,
                                          Array2DReal &Tend) {
      // dummy vars
      Array2DR8 HCell;
      I4 LevelsPerChunk;

      const I4 ICell0 = CellsOnEdge(IEdge, 0);
      const I4 ICell1 = CellsOnEdge(IEdge, 1);
      const Real InvDcEdge = 1. / DcEdge(IEdge);

      for (int K = KChunk * LevelsPerChunk; K < (KChunk + 1) * LevelsPerChunk; ++K) {
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

   KOKKOS_INLINE_FUNCTION void operator()(I4 IEdge,
                                          I4 KChunk,
                                          const OceanState *State,
//                                          const OceanAuxState *AuxState,
                                          Array2DReal &Tend) {
      // dummy vars
      I4 LevelsPerChunk;
      Array2DR8 DivCell;
      Array2DR8 RVortVertex;

      const I4 ICell0 = CellsOnEdge(IEdge, 0);
      const I4 ICell1 = CellsOnEdge(IEdge, 1);

      const I4 IVertex0 = VerticesOnEdge(IEdge, 0);
      const I4 IVertex1 = VerticesOnEdge(IEdge, 1);

      const Real DcEdgeInv = 1. / DcEdge(IEdge);
      const Real DvEdgeInv = 1. / DvEdge(IEdge);

      for (int K = KChunk * LevelsPerChunk; K < (KChunk + 1) * LevelsPerChunk; ++K) {
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

   KOKKOS_INLINE_FUNCTION void operator()(I4 IEdge,
                                          I4 KChunk,
                                          const OceanState *State,
//                                          const OceanAuxState *AuxState,
                                          Array2DReal &Tend) {
      // dummy vars
      I4 LevelsPerChunk;
      Array2DR8 Del2DivCell;
      Array2DR8 Del2RVortVertex;

      const I4 ICell0 = CellsOnEdge(IEdge, 0);
      const I4 ICell1 = CellsOnEdge(IEdge, 1);

      const I4 IVertex0 = VerticesOnEdge(IEdge, 0);
      const I4 IVertex1 = VerticesOnEdge(IEdge, 1);

      const Real DcEdgeInv = 1. / DcEdge(IEdge);
      const Real DvEdgeInv = 1. / DvEdge(IEdge);

      for (int K = KChunk * LevelsPerChunk; K < (KChunk + 1) * LevelsPerChunk; ++K) {
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

   KOKKOS_INLINE_FUNCTION void operator()(I4 L,
                                          I4 ICell,
                                          I4 KChunk,
                                          const OceanState *State,
//                                          const OceanAuxState *AuxState,
                                          Array3DReal &Tend) {
      // dummy vars
      I4 LevelsPerChunk;
      Array2DR8 VEdge;
      Array3DR8 NormTrCell;
      Array2DR8 HFluxEdge;

      Real InvAreaCell = 1. / AreaCell(ICell);
      Real Accum = 0;

      for (int J = 0; J < NEdgesOnCell(ICell); ++J) {
         const I4 JEdge = EdgesOnCell(ICell, J);

         const I4 JCell0 = CellsOnEdge(JEdge, 0);
         const I4 JCell1 = CellsOnEdge(JEdge, 1);

         for (int K = KChunk * LevelsPerChunk; K < (KChunk + 1) * LevelsPerChunk; ++K) {
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

} // namespace OMEGA
#endif
