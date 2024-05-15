#include "Tend.h"
#include "Config.h"
#include "DataTypes.h"
#include "HorzMesh.h"

namespace OMEGA {

ThicknessFluxDivergenceOnCell::ThicknessFluxDivergenceOnCell(
    const HorzMesh *Mesh, Config *Options)
    : NEdgesOnCell(Mesh->NEdgesOnCell), EdgesOnCell(Mesh->EdgesOnCell),
      DvEdge(Mesh->DvEdge), AreaCell(Mesh->AreaCell),
      EdgeSignOnCell(Mesh->EdgeSignOnCell) {

    Options->get("ThicknessFluxTendencyEnable", Enabled);
}

PotentialVortFluxOnEdge::PotentialVortFluxOnEdge(
    const HorzMesh *Mesh, Config *Options)
    : NEdgesOnEdge(Mesh->NEdgesOnEdge), EdgesOnEdge(Mesh->EdgesOnEdge),
      WeightsOnEdge(Mesh->WeightsOnEdge) {

    Options->get("PVTendencyEnable", Enabled);

}

KineticEnergyGradOnEdge::KineticEnergyGradOnEdge(
    const HorzMesh *Mesh, Config *Options)
    : CellsOnEdge(Mesh->CellsOnEdge), DcEdge(Mesh->DcEdge) {

    Options->get("KETendencyEnable", Enabled);
}

SSHGradOnEdge::SSHGradOnEdge(const HorzMesh *Mesh, Config *Options)
    : CellsOnEdge(Mesh->CellsOnEdge), DcEdge(Mesh->DcEdge) {

    Options->get("SSHTendencyEnable", Enabled);
    Options->get("Gravity", Grav);
}

VelocityDiffusionOnEdge::VelocityDiffusionOnEdge(
    const HorzMesh *Mesh, Config *Options)
    : CellsOnEdge(Mesh->CellsOnEdge), VerticesOnEdge(Mesh->VerticesOnEdge),
      DcEdge(Mesh->DcEdge), DvEdge(Mesh->DvEdge) { //,
//      MeshScalingDel2(Mesh->MeshScalingDel2), EdgeMask(Mesh->EdgeMask) {

    Options->get("VelDiffTendencyEnable", Enabled);
    Options->get("ViscDel2", ViscDel2);

}

VelocityHyperDiffusionOnEdge::VelocityHyperDiffusionOnEdge(
    const HorzMesh *Mesh, Config *Options)
    : CellsOnEdge(Mesh->CellsOnEdge), VerticesOnEdge(Mesh->VerticesOnEdge),
      DcEdge(Mesh->DcEdge), DvEdge(Mesh->DvEdge) { //,
//      MeshScalingDel4(Mesh->MeshScalingDel4), EdgeMask(Mesh->EdgeMask) {

    Options->get("VelHyperDiffTendencyEnable", Enabled);
    Options->get("ViscDel4", ViscDel4);
}

TracerHorzAdvOnCell::TracerHorzAdvOnCell(
    const HorzMesh *Mesh, Config *Options)
    : NEdgesOnCell(Mesh->NEdgesOnCell), EdgesOnCell(Mesh->EdgesOnCell),
      CellsOnEdge(Mesh->CellsOnEdge), EdgeSignOnCell(Mesh->EdgeSignOnCell),
      DvEdge(Mesh->DvEdge), AreaCell(Mesh->AreaCell) {

    Options->get("TracerHAdvTendency", Enabled);

}

} // end namespace OMEGA

//===----------------------------------------------------------------------===//
