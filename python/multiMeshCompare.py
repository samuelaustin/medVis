import sys
import numpy as np
import scipy.spatial as spatial
from scipy.spatial import distance
from vtk import *
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy

def GetData(fileName):
    reader = vtkPolyDataReader()
    reader.SetFileName(fileName)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()

    return reader.GetOutput()

def GetActor(data):
    numOrgans = 8
    lut = MakeLUT(numOrgans)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(data)
    mapper.SetLookupTable(lut)
    mapper.SetScalarVisibility(1)
    mapper.SetScalarRange(0, numOrgans-1)
    mapper.SetScalarModeToUseCellData()
    mapper.Update()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(20)

    return actor

def PointToPointDistance(referencePolyData, polyData):
    referencePoints = vtk_to_numpy(referencePolyData.GetPoints().GetData())
    dataPoints = vtk_to_numpy(polyData.GetPoints().GetData())

    referenceTree = spatial.cKDTree(referencePoints, leafsize=10)

    numDataPoints = polyData.GetNumberOfPoints()
    print("Data points: " + str(numDataPoints))

    mean = 0
    for point in dataPoints:
        (dist, index) = referenceTree.query(point, k=1, distance_upper_bound=100)
        mean = mean + dist

    mean = mean / numDataPoints
    return mean



def MakeLUT(tableSize):
    '''
    Make a lookup table from a set of named colors.
    :param: tableSize - The table size
    :return: The lookup table.
    '''
    nc = vtk.vtkNamedColors()
 
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(tableSize)
    lut.Build()
 
    # Fill in a few known colors, the rest will be generated if needed
    lut.SetTableValue(0,nc.GetColor4d("Mint"))
    lut.SetTableValue(1,nc.GetColor4d("Banana"))
    lut.SetTableValue(2,nc.GetColor4d("Tomato"))
    lut.SetTableValue(3,nc.GetColor4d("Wheat"))
    lut.SetTableValue(4,nc.GetColor4d("Lavender"))
    lut.SetTableValue(5,nc.GetColor4d("Flesh"))
    lut.SetTableValue(6,nc.GetColor4d("Raspberry"))
    lut.SetTableValue(7,nc.GetColor4d("Salmon"))
    #lut.SetTableValue(8,nc.GetColor4d("Black"))
    #lut.SetTableValue(9,nc.GetColor4d("Peacock"))
 
    return lut



referenceData = GetData("../model.vtk")
patientData = GetData("../pat1.vtk")

print("Calculating point-to-point distance.")
dist = PointToPointDistance(referenceData, patientData)
print("Mean distance to reference: " + str(dist))

referenceActor = GetActor(referenceData)
patientActor = GetActor(patientData)

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
 
renderer.AddActor(referenceActor)
renderer.AddActor(patientActor)
 
renderWindow.Render()
renderWindowInteractor.Start()
