#include "Tend.h"
#include "Config.h"
#include "DataTypes.h"
#include "HorzMesh.h"

namespace OMEGA {

ThickFluxDivOnC::ThickFluxDivOnC(
    const HorzMesh *Mesh, Config *Options)
    : NEdgesOnCell(Mesh->NEdgesOnCell), EdgesOnCell(Mesh->EdgesOnCell),
      DvEdge(Mesh->DvEdge), AreaCell(Mesh->AreaCell),
      EdgeSignOnCell(Mesh->EdgeSignOnCell) {

    Options->get("ThicknessFluxTendencyEnable", Enabled);
}

PotVortFluxOnE::PotVortFluxOnE(
    const HorzMesh *Mesh, Config *Options)
    : NEdgesOnEdge(Mesh->NEdgesOnEdge), EdgesOnEdge(Mesh->EdgesOnEdge),
      WeightsOnEdge(Mesh->WeightsOnEdge) {

    Options->get("PVTendencyEnable", Enabled);

}

KEGradOnE::KEGradOnE(
    const HorzMesh *Mesh, Config *Options)
    : CellsOnEdge(Mesh->CellsOnEdge), DcEdge(Mesh->DcEdge) {

    Options->get("KETendencyEnable", Enabled);
}

SSHGradOnEdge::SSHGradOnEdge(const HorzMesh *Mesh, Config *Options)
    : CellsOnEdge(Mesh->CellsOnEdge), DcEdge(Mesh->DcEdge) {

    Options->get("SSHTendencyEnable", Enabled);
    Options->get("Gravity", Grav);
}

VelDiffOnE::VelDiffOnE(
    const HorzMesh *Mesh, Config *Options)
    : CellsOnEdge(Mesh->CellsOnEdge), VerticesOnEdge(Mesh->VerticesOnEdge),
      DcEdge(Mesh->DcEdge), DvEdge(Mesh->DvEdge) { //,
//      MeshScalingDel2(Mesh->MeshScalingDel2), EdgeMask(Mesh->EdgeMask) {

    Options->get("VelDiffTendencyEnable", Enabled);
    Options->get("ViscDel2", ViscDel2);

}

VelHyperDiffOnE::VelHyperDiffOnE(
    const HorzMesh *Mesh, Config *Options)
    : CellsOnEdge(Mesh->CellsOnEdge), VerticesOnEdge(Mesh->VerticesOnEdge),
      DcEdge(Mesh->DcEdge), DvEdge(Mesh->DvEdge) { //,
//      MeshScalingDel4(Mesh->MeshScalingDel4), EdgeMask(Mesh->EdgeMask) {

    Options->get("VelHyperDiffTendencyEnable", Enabled);
    Options->get("ViscDel4", ViscDel4);
}

TracerHAdvOnC::TracerHAdvOnC(
    const HorzMesh *Mesh, Config *Options)
    : NEdgesOnCell(Mesh->NEdgesOnCell), EdgesOnCell(Mesh->EdgesOnCell),
      CellsOnEdge(Mesh->CellsOnEdge), EdgeSignOnCell(Mesh->EdgeSignOnCell),
      DvEdge(Mesh->DvEdge), AreaCell(Mesh->AreaCell) {

    Options->get("TracerHAdvTendencyEnable", Enabled);

}

TracerDiffOnC::TracerDiffOnC(
    const HorzMesh *Mesh, Config *Options)
    : NEdgesOnCell(Mesh->NEdgesOnCell), EdgesOnCell(Mesh->EdgesOnCell),
      CellsOnEdge(Mesh->CellsOnEdge), EdgeSignOnCell(Mesh->EdgeSignOnCell),
      DvEdge(Mesh->DvEdge), DcEdge(Mesh->DcEdge), AreaCell(Mesh->AreaCell) {
//      MeshScalingDel2(Mesh->MeshScalingDel2) {

    Options->get("TracerDiffTendencyEnable", Enabled);
    Options->get("EddyDiff2", EddyDiff2);

}

TracerHyperDiffOnC::TracerHyperDiffOnC(
    const HorzMesh *Mesh, Config *Options)
    : NEdgesOnCell(Mesh->NEdgesOnCell), EdgesOnCell(Mesh->EdgesOnCell),
      CellsOnEdge(Mesh->CellsOnEdge), EdgeSignOnCell(Mesh->EdgeSignOnCell),
      DvEdge(Mesh->DvEdge), DcEdge(Mesh->DcEdge), AreaCell(Mesh->AreaCell) {
//      MeshScalingDel4(Mesh->MeshScalingDel2) {

    Options->get("TracerHyperDiffTendencyEnable", Enabled);
    Options->get("EddyDiff4", EddyDiff4);

}

} // end namespace OMEGA

//===----------------------------------------------------------------------===//
