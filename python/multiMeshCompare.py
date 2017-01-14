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
g_referenceActor = 0
g_patientActors = []
g_lineActors = []
distances = []
g_linePolygons = []


class ClickInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    
    visibleIndex = 0

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

    def MakeLinesVisible(self, index):
        global g_patientActors
        global g_lineActors

        for i in range(0,len(g_patientActors)):
            g_patientActors[i].GetProperty().SetOpacity(0)

        for i in range(0,len(g_lineActors)):
            g_lineActors[i].GetProperty().SetOpacity(0)

        g_lineActors[index].GetProperty().SetOpacity(1)
        renderWindow.Render()

    def MakePatientVisible(self, index):
        global g_patientActors
        global g_lineActors

        for i in range(0,len(g_patientActors)):
            g_patientActors[i].GetProperty().SetOpacity(0)

        for i in range(0,len(g_lineActors)):
            g_lineActors[i].GetProperty().SetOpacity(0)

        g_patientActors[index].GetProperty().SetOpacity(1)
        renderWindow.Render()

    def TogglePatientVisibility(self, index):
        global g_patientActors
        global renderWindow

        opacity = g_patientActors[index].GetProperty().GetOpacity()
        if opacity > 0:
            g_patientActors[index].GetProperty().SetOpacity(0)
        else:
            g_patientActors[index].GetProperty().SetOpacity(1)

        renderWindow.Render()


    def onKeyPressEvent(self, renderer, event):    
        key = self.GetInteractor().GetKeySym()

        global g_minimumDistance
        global g_maximumDistance
        global g_visualDistance
        global g_patientActors
        global g_referenceActor
        global g_lineActors
        global renderWindow

        diff = g_maximumDistance - g_minimumDistance
        stepSize = diff / 200

        if(key == "Right"):
            #g_visualDistance = min([g_maximumDistance, g_visualDistance+stepSize])
            #print(g_visualDistance)
            #self.UpdatePatientOpacities()
            self.visibleIndex = min(self.visibleIndex+1, len(g_patientActors)-1)
            self.MakePatientVisible(self.visibleIndex)
        elif(key == "Left"):
            #g_visualDistance = max([g_minimumDistance, g_visualDistance-stepSize])
            #print(g_visualDistance)
            #self.UpdatePatientOpacities()
            self.visibleIndex = max(self.visibleIndex-1, 0)
            self.MakePatientVisible(self.visibleIndex)
        elif(key == "Up"):
            self.visibleIndex = min(self.visibleIndex+1, len(g_patientActors)-1)
            self.MakeLinesVisible(self.visibleIndex)
        elif(key == "Down"):
            self.visibleIndex = max(self.visibleIndex-1, 0)
            self.MakeLinesVisible(self.visibleIndex)
        elif(key == "m"):
            opacity = g_referenceActor.GetProperty().GetOpacity()
            g_referenceActor.GetProperty().SetOpacity(1-opacity)
            renderWindow.Render()
        elif(key == "p"):
            opacity = g_patientActors[self.visibleIndex].GetProperty().GetOpacity()
            g_patientActors[self.visibleIndex].GetProperty().SetOpacity(1-opacity)
            renderWindow.Render()
        elif(key == "l"):
            opacity = g_lineActors[self.visibleIndex].GetProperty().GetOpacity()
            opacity = g_lineActors[self.visibleIndex].GetProperty().SetOpacity(1-opacity)
            renderWindow.Render()
        elif(key == "1"):
            self.TogglePatientVisibility(0)
        elif(key == "2"):
            self.TogglePatientVisibility(1)
        elif(key == "3"):
            self.TogglePatientVisibility(2)
        elif(key == "4"):
            self.TogglePatientVisibility(3)
        elif(key == "5"):
            self.TogglePatientVisibility(4)
        elif(key == "6"):
            self.TogglePatientVisibility(5)
        elif(key == "7"):
            self.TogglePatientVisibility(6)
        elif(key == "8"):
            self.TogglePatientVisibility(7)
        elif(key == "9"):
            self.TogglePatientVisibility(8)
     

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
    actor.GetProperty().SetPointSize(1)

    return actor

def PointToPointDistance(referenceTree, polyData):
    global g_linePolygons

    dataPoints = vtk_to_numpy(polyData.GetPoints().GetData())
    numDataPoints = polyData.GetNumberOfPoints()

    points = vtk.vtkPoints()
    distances = []

    mean = 0
    for point in dataPoints:
        (dist, index) = referenceTree.query(point, k=1, distance_upper_bound=100)
        distances.append(dist)
        mean = mean + dist

        modelPoint = referenceTree.data[index]
        points.InsertNextPoint(point[0], point[1], point[2])
        points.InsertNextPoint(modelPoint[0], modelPoint[1], modelPoint[2])

    distances = np.array(distances)
    indices = distances.argsort()
    sortedDistances = distances[indices]

    length = len(sortedDistances)
    quarter = int(length/4)
    half = int(length/2)
    three_quarter = half+quarter

    distanceColorValues = [sortedDistances[three_quarter], sortedDistances[half], sortedDistances[quarter]]

    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("colors")
    for dist in distances:
        if dist > distanceColorValues[0]:
            colors.InsertNextTuple3(204,76,2) #Red
        elif dist > distanceColorValues[1]:
            colors.InsertNextTuple3(254,153,41) #Orange
        elif dist > distanceColorValues[2]:
            colors.InsertNextTuple3(254,217,142) #Yellow
        else:
            colors.InsertNextTuple3(255,255,212) #White

    lines = vtk.vtkCellArray()
    for i in range(0, numDataPoints):
        index = i*2
        lines.InsertNextCell(2)
        lines.InsertCellPoint(index)
        lines.InsertCellPoint(index+1)

    polygon = vtk.vtkPolyData()
    polygon.SetPoints(points)
    polygon.SetLines(lines)
    polygon.GetCellData().SetScalars(colors);

    g_linePolygons.append(polygon)

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
    actors = []
    for i in range(0, len(patientData)):
        actors.append(GetActor(patientData[i]))

    return actors


print(vtk.vtkVersion.GetVTKSourceVersion())

referenceData = GetData("../model.vtk")
patientData = ImportPatientData()

referencePoints = vtk_to_numpy(referenceData.GetPoints().GetData())
referenceTree = spatial.cKDTree(referencePoints, leafsize=10)

print("Calculating point-to-point distance.")

for i in range(0, len(patientData)):
    (dist, colors) = PointToPointDistance(referenceTree, patientData[i])
    distances.append(dist)
    patientData[i].GetPointData().SetScalars(colors)
print("Distances:")
print(distances)

g_minimumDistance = min(distances)
g_maximumDistance = max(distances)
g_visualDistance = g_minimumDistance

g_referenceActor = GetReferenceActor(referenceData)
g_referenceActor.GetProperty().SetRepresentationToWireframe()

renderer = vtk.vtkRenderer()
renderer.SetBackground(255,255,255);

renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

g_patientActors = CreateActors(patientData)
g_lineActors = CreateActors(g_linePolygons)

dist_patientActors = zip(distances, g_patientActors)
g_patientActors = [p for d, p in sorted(dist_patientActors)]

dist_lineActors = zip(distances, g_lineActors)
g_lineActors = [l for d, l in sorted(dist_lineActors)]

style = ClickInteractorStyle()
style.UpdatePatientOpacities()
renderWindowInteractor.SetInteractorStyle(style)

renderer.AddActor(g_referenceActor)
for actor in g_patientActors:
    actor.GetProperty().SetOpacity(0.2)
    renderer.AddActor(actor)

for actor in g_lineActors:
    actor.GetProperty().SetOpacity(0.0)
    renderer.AddActor(actor)

renderer.ResetCamera()
renderWindow.Render()
renderWindowInteractor.Start()
