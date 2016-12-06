//----------------------------------------------------------------------------------
//! Project global and OS specific declarations.
/*!
// \file    
// \author     Samuel
// \date    2016-11-27
*/
//----------------------------------------------------------------------------------


#pragma once


// DLL export macro definition.
#ifdef _MLMESHCOMPARE_EXPORTS
  // Use the _MLMESHCOMPARE_EXPORT macro to export classes and functions.
  #define _MLMESHCOMPARE_EXPORT ML_LIBRARY_EXPORT_ATTRIBUTE
#else
  // If included by external modules, exported symbols are declared as import symbols.
  #define _MLMESHCOMPARE_EXPORT ML_LIBRARY_IMPORT_ATTRIBUTE
#endif
