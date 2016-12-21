import sys
sys.path.insert(0, '/opt/VTK-7.0.0/lib/python3.5/site-packages')
import numpy as np
import scipy.spatial as spatial
from scipy.spatial import distance
from vtk import *
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy

g_minimumDistance = 0
g_maximumDistance = 0
g_visualDistance = 0
g_patientActors = []
distances = []


class ClickInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, parent=None):
            self.AddObserver("KeyPressEvent",self.onKeyPressEvent)

    def UpdatePatientOpacities(self):
        global g_patientActors
        global g_minimumDistance
        global g_maximumDistance
        global g_visualDistance
        global renderWindow

        diff = g_maximumDistance - g_minimumDistance
        for i in range(0,len(distances)):
            dist = abs(g_visualDistance - distances[i])
            relative = dist / diff
            opacity = max(0, (1-(20*relative)))
            g_patientActors[i].GetProperty().SetOpacity(opacity)
        renderWindow.Render()


    def onKeyPressEvent(self, renderer, event):        
        key = self.GetInteractor().GetKeySym()

        global g_minimumDistance
        global g_maximumDistance
        global g_visualDistance

        diff = g_maximumDistance - g_minimumDistance
        stepSize = diff / 200

        if(key == "Right" or key == "Up"):
            g_visualDistance = min([g_maximumDistance, g_visualDistance+stepSize])
        elif(key == "Left" or key == "Down"):
            g_visualDistance = max([g_minimumDistance, g_visualDistance-stepSize])
        print(g_visualDistance)
        self.UpdatePatientOpacities()

def GetData(fileName):
    reader = vtkPolyDataReader()
    reader.SetFileName(fileName)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()

    return reader.GetOutput()

def GetReferenceActor(data):
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

def GetActor(data):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(data)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(20)

    return actor

def PointToPointDistance(referenceTree, polyData):
    
    dataPoints = vtk_to_numpy(polyData.GetPoints().GetData())
    numDataPoints = polyData.GetNumberOfPoints()

    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("colors")

    mean = 0
    for point in dataPoints:
        (dist, index) = referenceTree.query(point, k=1, distance_upper_bound=100)
        mean = mean + dist
        redness = min(255, 10*int(dist))
        otherColors = 255 - redness
        colors.InsertNextTuple3(255,otherColors,otherColors)

    mean = mean / numDataPoints
    return (mean, colors)


def MakeLUT(tableSize):
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
 
    return lut

def ImportPatientData():
    patientData = []
    patientData.append(GetData("../pat1.vtk"))
    patientData.append(GetData("../pat3.vtk"))
    patientData.append(GetData("../pat4.vtk"))
    patientData.append(GetData("../pat5.vtk"))
    patientData.append(GetData("../pat7.vtk"))
    patientData.append(GetData("../pat8.vtk"))
    patientData.append(GetData("../pat9.vtk"))
    patientData.append(GetData("../pat10.vtk"))
    patientData.append(GetData("../pat11.vtk"))
    return patientData


def CreateActors(patientData):
    global g_patientActors
    for i in range(0, len(patientData)):
        g_patientActors.append(GetActor(patientData[i]))


print(vtk.vtkVersion.GetVTKSourceVersion())

referenceData = GetData("../model.vtk")
patientData = ImportPatientData()

referencePoints = vtk_to_numpy(referenceData.GetPoints().GetData())
referenceTree = spatial.cKDTree(referencePoints, leafsize=10)

print("Calculating point-to-point distance.")

patientColors = []
for i in range(0, len(patientData)):
    (dist, colors) = PointToPointDistance(referenceTree, patientData[i])
    distances.append(dist)
    patientData[i].GetPointData().SetScalars(colors)
print("Distances:")
print(distances)

g_minimumDistance = min(distances)
g_maximumDistance = max(distances)
g_visualDistance = g_minimumDistance

referenceActor = GetReferenceActor(referenceData)
referenceActor.GetProperty().SetRepresentationToWireframe()

CreateActors(patientData)

renderer = vtk.vtkRenderer()
#renderer.SetBackground(1,1,1);

renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

style = ClickInteractorStyle()
style.UpdatePatientOpacities()
renderWindowInteractor.SetInteractorStyle(style)
 
renderer.AddActor(referenceActor)
for actor in g_patientActors:
    actor.GetProperty().SetOpacity(0.2)
    renderer.AddActor(actor)

renderWindow.Render()
renderWindowInteractor.Start()
