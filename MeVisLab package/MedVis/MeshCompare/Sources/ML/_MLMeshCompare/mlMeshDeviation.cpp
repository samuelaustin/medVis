//----------------------------------------------------------------------------------
//! The ML module class MeshDeviation.
/*!
// \file   
// \author     Samuel
// \date    2016-11-27
//
// Calculates the difference between two meshes
*/
//----------------------------------------------------------------------------------

#include "mlMeshDeviation.h"


ML_START_NAMESPACE

//! Implements code for the runtime type system of the ML
ML_MODULE_CLASS_SOURCE(MeshDeviation, Module);

//----------------------------------------------------------------------------------

MeshDeviation::MeshDeviation() : Module(2, 1)
{
  // Suppress calls of handleNotification on field changes to
  // avoid side effects during initialization phase.
  handleNotificationOff();

  // Reactivate calls of handleNotification on field changes.
  handleNotificationOn();


  // Activate inplace data buffers for output outputIndex and input inputIndex.
  // setOutputImageInplace(outputIndex, inputIndex);

  // Activate page data bypass from input inputIndex to output outputIndex.
  // Note that the module must still be able to calculate the output image.
  // setBypass(outputIndex, inputIndex);

}

//----------------------------------------------------------------------------------

void MeshDeviation::handleNotification(Field* field)
{
  // Handle changes of module parameters and input image fields here.
  bool touchOutputs = false;
  if (isInputImageField(field))
  {
    touchOutputs = true;
  }

  if (touchOutputs) 
  {
    // Touch all output image fields to notify connected modules.
    touchOutputImageFields();
  }
}

//----------------------------------------------------------------------------------

void MeshDeviation::activateAttachments()
{
  // Update members to new field state here.
  // Call super class functionality to enable notification handling again.
  Module::activateAttachments();
}

//----------------------------------------------------------------------------------

void MeshDeviation::calculateOutputImageProperties(int /*outputIndex*/, PagedImage* outputImage)
{
  // Change properties of output image outputImage here whose
  // defaults are inherited from the input image 0 (if there is one).

  // Set output image to a fixed type.
  //outputImage->setDataType(MLdoubleType);

  // Specify input sub-image data types (otherwise the above output data type is used for all input sub-images).
  //outputImage->setInputSubImageDataType(0, getInputImage(0)->getDataType());
  //outputImage->setInputSubImageDataType(1, getInputImage(1)->getDataType());
}

//----------------------------------------------------------------------------------

SubImageBox MeshDeviation::calculateInputSubImageBox(int inputIndex, const SubImageBox& outputSubImageBox, int outputIndex)
{
  // Return region of input image inputIndex needed to compute region
  // outSubImgBox of output image outputIndex.
  return outputSubImageBox;
}


//----------------------------------------------------------------------------------

ML_CALCULATEOUTPUTSUBIMAGE_NUM_INPUTS_2_SCALAR_TYPES_CPP(MeshDeviation);

template <typename T>
void MeshDeviation::calculateOutputSubImage(TSubImage<T>* outputSubImage, int outputIndex
                                     , TSubImage<T>* inputSubImage0
                                     , TSubImage<T>* inputSubImage1
                                     )
{
  // Compute sub-image of output image outputIndex from input sub-images.

  // Clamp box of output image against image extent to avoid that unused areas are processed.
  const SubImageBox validOutBox = outputSubImage->getValidRegion();

  // Process all voxels of the valid region of the output page.
  ImageVector p;
  for (p.u=validOutBox.v1.u;  p.u<=validOutBox.v2.u;  ++p.u) {
    for (p.t=validOutBox.v1.t;  p.t<=validOutBox.v2.t;  ++p.t) {
      for (p.c=validOutBox.v1.c;  p.c<=validOutBox.v2.c;  ++p.c) {
        for (p.z=validOutBox.v1.z;  p.z<=validOutBox.v2.z;  ++p.z) {
          for (p.y=validOutBox.v1.y;  p.y<=validOutBox.v2.y;  ++p.y) {

            p.x = validOutBox.v1.x;
            // Get pointers to row starts of input and output sub-images.
            const T* inVoxel0 = inputSubImage0->getImagePointer(p);

            T*  outVoxel = outputSubImage->getImagePointer(p);

            const MLint rowEnd   = validOutBox.v2.x;

            // Process all row voxels.
            for (; p.x <= rowEnd;  ++p.x, ++outVoxel, ++inVoxel0)
            {
              *outVoxel = *inVoxel0;
            }
          }
        }
      }
    }
  }
}

ML_END_NAMESPACE