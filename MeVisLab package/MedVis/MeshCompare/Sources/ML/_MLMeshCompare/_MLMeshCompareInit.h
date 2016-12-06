//----------------------------------------------------------------------------------
//! Dynamic library and runtime type system initialization.
/*!
// \file    
// \author     Samuel
// \date    2016-11-27
*/
//----------------------------------------------------------------------------------


#pragma once


ML_START_NAMESPACE

//! Calls init functions of all modules to add their types to the runtime type
//! system of the ML.
int _MLMeshCompareInit();

ML_END_NAMESPACE
