//----------------------------------------------------------------------------------
//! Dynamic library and runtime type system initialization.
/*!
// \file    
// \author     Samuel
// \date    2016-11-27
*/
//----------------------------------------------------------------------------------

#include "_MLMeshCompareSystem.h"

// Include definition of ML_INIT_LIBRARY.
#include <mlLibraryInitMacros.h>

// Include all module headers ...
#include "mlMeshDeviation.h"


ML_START_NAMESPACE

//----------------------------------------------------------------------------------
//! Calls init functions of all modules to add their types to the runtime type
//! system of the ML.
//----------------------------------------------------------------------------------
int _MLMeshCompareInit()
{
  // Add initClass calls from modules here.
  MeshDeviation::initClass();

  return 1;
}

ML_END_NAMESPACE


//! Calls the init method implemented above during load of shared library.
ML_INIT_LIBRARY(_MLMeshCompareInit)