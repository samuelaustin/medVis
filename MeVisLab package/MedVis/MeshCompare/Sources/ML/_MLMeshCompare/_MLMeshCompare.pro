# -----------------------------------------------------------------------------
# _MLMeshCompare project profile
#
# \file
# \author     Samuel
# \date    2016-11-27
# -----------------------------------------------------------------------------


TEMPLATE   = lib
TARGET     = _MLMeshCompare

DESTDIR    = $$(MLAB_CURRENT_PACKAGE_DIR)/lib
DLLDESTDIR = $$(MLAB_CURRENT_PACKAGE_DIR)/lib

# Set high warn level (warn 4 on MSVC)
WARN = HIGH

# Add used projects here (see included pri files below for available projects)
CONFIG += dll ML

MLAB_PACKAGES += MedVis_MeshCompare \
                 MeVisLab_Standard

# make sure that this file is included after CONFIG and MLAB_PACKAGES
include ($(MLAB_MeVis_Foundation)/Configuration/IncludePackages.pri)

DEFINES += _MLMESHCOMPARE_EXPORTS

# Enable ML deprecated API warnings. To completely disable the deprecated API, change WARN to DISABLE.
DEFINES += ML_WARN_DEPRECATED

HEADERS += \
    _MLMeshCompareInit.h \
    _MLMeshCompareSystem.h \
    mlMeshDeviation.h \

SOURCES += \
    _MLMeshCompareInit.cpp \
    mlMeshDeviation.cpp \