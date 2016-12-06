isEmpty(MedVis_MeshCompare_PRI_INCLUDED) {
  message ( loading MedVis_MeshCompare.pri )
}
# **InsertLicense** code
# -----------------------------------------------------------------------------
# MedVis_MeshCompare prifile
#
# \file    MedVis_MeshCompare.pri
# \author     Samuel
# \date    2016-11-27
#
# 
#
# -----------------------------------------------------------------------------

# include guard against multiple inclusion
isEmpty(MedVis_MeshCompare_PRI_INCLUDED) {

MedVis_MeshCompare_PRI_INCLUDED = 1

# -- System -------------------------------------------------------------

include( $(MLAB_MeVis_Foundation)/Configuration/SystemInit.pri )

# -- Define local PACKAGE variables -------------------------------------

PACKAGE_ROOT    = $$(MLAB_MedVis_MeshCompare)
PACKAGE_SOURCES = "$${PACKAGE_ROOT}"/Sources
PACKAGE_PROJECTS = "$${PACKAGE_ROOT}"/Projects

# Add package library path
LIBS          += -L"$${PACKAGE_ROOT}"/lib

# -- Projects -------------------------------------------------------------

# NOTE: Add projects below to make them available to other projects via the CONFIG mechanism

# -----------------------------------------------------------------------------
# You can use these example templates for typical projects #
# -----------------------------------------------------------------------------

# --------------------------------
# For self-contained projects (located in the 'Projects' folder of the package):

#MLMySelfContainedProject {
#  CONFIG_FOUND += MLMySelfContainedProject
#  $include ( $${PACKAGE_PROJECTS}/MLMySelfContainedProject/Sources/MLMySelfContainedProject.pri ) 
#}

# and within $${PACKAGE_PROJECTS}/MLMySelfContainedProject/Sources/MLMySelfContainedProject.pri: 
#
#  INCLUDEPATH += $${PACKAGE_PROJECTS}/MLMySelfContainedProject/Sources
#  win32:LIBS += MLMySelfContainedProject$${d}.lib
#  unix:LIBS += -lMLMySelfContainedProject$${d}
#  CONFIG += ProjectMLMySelfContainedProjectDependsOn1 ProjectMLMySelfContainedProjectDependsOn2 ...

# --------------------------------
# For non-self-contained projects (old-style, distributed over Sources/Modules/... ):

#MLMyProject {
#  CONFIG_FOUND += MLMyProject
#  INCLUDEPATH += $${PACKAGE_SOURCES}/ML/MLMyProject
#  win32:LIBS += MLMyProject$${d}.lib
#  unix:LIBS += -lMLMyProject$${d}
#  CONFIG += ProjectMLMyProjectDependsOn1 ProjectMLMyProjectDependsOn2 ...
#}

# -----------------------------------------------------------------------------

# -- ML Projects -------------------------------------------------------------

# -- Inventor Projects -------------------------------------------------------

# -- Shared Projects ---------------------------------------------------------

# End of projects ------------------------------------------------------------

}