import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import scipy.spatial as spatial
from scipy.spatial import distance
import copy as cp


class MyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
 
    def __init__(self,parent=None):
        self.AddObserver("KeyPressEvent",self.onKeyPressEvent)
        #self.AddObserver("LeftButtonPressEvent",self.leftButtonPressEvent)
 
    def onKeyPressEvent(self,ren, event):
        key = self.GetInteractor().GetKeySym()
        
        #renderer = vtk.vtkRenderer()
        global renderer
        global path
        global renderWindow
        global OrganActors
        global OrganRefActors
        global OrganActorsMean
        global mark
        global count
	
        #OrganRefList = Initial(path,model)[2]
          
        if(key == "b"):
            renderer.RemoveAllViewProps()
            renderer.AddActor(Bladders[count])
            renderer.AddActor(OrganRefActors[0])
            mark = "Bladder"
            renderWindow.Render() 
			
        elif(key == "s"):
            renderer.RemoveAllViewProps()
            renderer.AddActor(SeminalVesicles[count])
            OrganRefActors[2].GetProperty().SetRepresentationToWireframe()
            renderer.AddActor(OrganRefActors[2])
            mark = "SV"

        elif(key == "r"):
            renderer.RemoveAllViewProps()
            renderer.AddActor(Rectums[count])
            renderer.AddActor(OrganRefActors[3])
            mark = "Rectum"

        elif(key == "p"):
            renderer.RemoveAllViewProps()
            renderer.AddActor(Prostates[count])
            renderer.AddActor(OrganRefActors[1])
            mark = "Prostate"
            
        elif(key =='m'):
            renderer.RemoveAllViewProps()
			
            for i in OrganActorsMean:
                renderer.AddActor(i)
                renderWindow.Render()
        elif(key == "Right"):
            if count != (9):
                count +=1
                print ('Patient',count)
				
            if(mark == "Bladder"):
                renderer.RemoveAllViewProps()
                renderer.AddActor(Bladders[count])
                renderer.AddActor(OrganRefActors[0])				
                mark = "Bladder"
                renderWindow.Render()				

            elif(mark == "SV"):
                renderer.RemoveAllViewProps()
                renderer.AddActor(SeminalVesicles[count])
                renderer.AddActor(OrganRefActors[2])            
                mark = "SV"
                renderWindow.Render()
				
            elif(mark == "Rectum"):
                renderer.RemoveAllViewProps()
                renderer.AddActor(Rectums[count])
                renderer.AddActor(OrganRefActors[3])				
                mark = "Rectum"
                renderWindow.Render()

            elif(mark == "Prostate"):
                renderer.RemoveAllViewProps()
                renderer.AddActor(Prostates[count])
                renderer.AddActor(OrganRefActors[1])				
                mark = "Prostate"				
                renderWindow.Render()
				
        elif(key == "Left"):
            if count != 0:
                count -=1
                print ('Patient',count)				
                if(mark == "Bladder"):
                    renderer.RemoveAllViewProps()
                    renderer.AddActor(Bladders[count])
                    renderer.AddActor(OrganRefActors[0])
                    mark = "Bladder"
                    renderWindow.Render()
					
                elif(mark == "SV"):
                    renderer.RemoveAllViewProps()
                    renderer.AddActor(SeminalVesicles[count])
                    renderer.AddActor(OrganRefActors[2])					
                    mark = "SV"
                    renderWindow.Render()
					
                elif(mark == "Rectum"):
                    renderer.RemoveAllViewProps()
                    renderer.AddActor(Rectums[count])
                    renderer.AddActor(OrganRefActors[3])					
                    mark = "Rectum"
                    renderWindow.Render()

                elif(mark == "Prostate"):
                    renderer.RemoveAllViewProps()
                    renderer.AddActor(Prostates[count])
                    renderer.AddActor(OrganRefActors[1])					
                    mark = "Prostate"
                    renderWindow.Render()

        return

def ExtractOrgan(Path,FileName,OrganLabel):

	reader = vtk.vtkPolyDataReader()
	reader.SetFileName(path+FileName)
	reader.ReadAllVectorsOn()
	reader.ReadAllScalarsOn()
	reader.Update()

	##---Building the vector containing the delimitatio indexes for the different biological structures---##
	
	ReaderOutput = reader.GetOutput()
	RefArray = vtk_to_numpy(ReaderOutput.GetCellData().GetArray(0))

	index_vec = []
	unique = np.unique(RefArray); unique = unique.astype(int)

	for i in unique:
		if i==1:
			vec=np.where(RefArray==i); vec = vec[-1];
			
			##--For the first group of 1--##
			ind_vec = np.where(vec<6000); ind_vec = ind_vec[-1]
			ini = vec[ind_vec[0]]; fini = vec[ind_vec[-1]];
			index_vec.append(ini); index_vec.append(fini);
			##--For the second group of 2--##
			ind_vec = np.where(vec>6000); ind_vec = ind_vec[-1]
			ini = vec[ind_vec[0]]; fini = vec[ind_vec[-1]];
			index_vec.extend([ini,fini])
		
		else:
			vec = np.where(RefArray==i); vec = vec[-1];
			ini = vec[0]; fini = vec[-1] ; 
			index_vec.extend([ini,fini])


	

	##---Call a cell in order not to kill the Python Kernel---##
	reader.GetOutput().GetCell(1)
	lastV = reader.GetOutput().GetNumberOfCells()
	
	##---Extract the desired Organ---##
	
	if OrganLabel == 1: #Bladder + BP
		for i in range(index_vec[4],index_vec[6]):
			reader.GetOutput().DeleteCell(i)
		for i in range(index_vec[8],index_vec[2]):
			reader.GetOutput().DeleteCell(i)
		for i in range(index_vec[12],lastV):
			reader.GetOutput().DeleteCell(i)
		
	elif OrganLabel == 2: #Prostate + BP + SVP
		for i in range(0,index_vec[4]):
			reader.GetOutput().DeleteCell(i)
		for i in range(index_vec[8],index_vec[10]):
			reader.GetOutput().DeleteCell(i)
		for i in range(index_vec[2],lastV):
			reader.GetOutput().DeleteCell(i)

	elif OrganLabel == 3: #Seminal Vesicles + SVP
		for i in range(0,index_vec[8]):
			reader.GetOutput().DeleteCell(i)
		for i in range(index_vec[2],lastV):
			reader.GetOutput().DeleteCell(i)
	
	elif OrganLabel == 4: #Rectum
		for i in range(0,index_vec[12]):
			reader.GetOutput().DeleteCell(i)
		for i in range(index_vec[14],lastV):
			reader.GetOutput().DeleteCell(i)
			
	reader.Update()
	ReaderOutput.RemoveDeletedCells()
	reader.Update()
	ReaderOutput = CleanUnusedPoints(ReaderOutput)
	#Points = np.asarray(vtk_to_numpy(ReaderOutput.GetPoints().GetData())).reshape(-1)
	Points = vtk_to_numpy(ReaderOutput.GetPoints().GetData())
	##---Set Mapper---##
	mapper = vtk.vtkPolyDataMapper()
	mapper.SetInputData(reader.GetOutput())
	mapper.Update()
	
	Mass = vtk.vtkMassProperties()
	Mass.SetInputData(ReaderOutput)
	Mass.Update()
	
	Volume = Mass.GetVolume() 
	Surface = Mass.GetSurfaceArea()
	
	##---Set Actor---##
	actor = vtk.vtkActor()
	actor.SetMapper(mapper)
	
	return actor,Points,ReaderOutput,Volume,Surface
	
def ExtractBladders(path,patientFile,number):
	#Bladder: OrganLabel = 1
	#Bladder-Prostate Interface: OrganLabel = 3
	#Extract all the organs: number = 1 (patientFile should be a list)
	#Extract just one organ: number = 0 (patientFile should be a single file)
	ActorList = []
	Points = []
	Outputs = []
	if number ==0:
		ActorList.append(ExtractOrgan(path,patientFile,1)[0])
		Points.append(ExtractOrgan(path,patientFile,1)[1])
		Outputs.append(ExtractOrgan(path,patientFile,1)[2])

				
	else:
		for i in patientFile:
			ActorList.append(ExtractOrgan(path,i,1)[0])
			Points.append(ExtractOrgan(path,i,1)[1])
			Outputs.append(ExtractOrgan(path,i,1)[2])
	
		
	return ActorList,Points,Outputs
		
def ExtractProstates(path,patientFile,number):
	#Prostate: OrganLabel = 2
	#Bladder-Prostate Interface: OrganLabel = 3
	#SeminalVesicles-Prostate Interface: OrganLabel = 5
	#Extract all the organs: number = 1 (patientFile should be a list)
	#Extract just one organ: number = 0 (patientFile should be a single file)
	ActorList = []
	Outputs = []
	Points = []
	if number ==0:
		ActorList.append(ExtractOrgan(path,patientFile,2)[0])
		Points.append(ExtractOrgan(path,patientFile,2)[1])
		Outputs.append(ExtractOrgan(path,patientFile,2)[2])

		
	else:
		for i in patientFile:
			ActorList.append(ExtractOrgan(path,i,2)[0])
			Points.append(ExtractOrgan(path,i,2)[1])
			Outputs.append(ExtractOrgan(path,i,2)[2])
			
		
	return ActorList, Points, Outputs

def ExtractSeminalVesicles(path,patientFile,number):
	#SeminalVesicles: OrganLabel = 5
	#SeminalVesicles-Prostate Interface: OrganLabel = 4
	#Extract all the organs: number = 1 (patientFile should be a list)
	#Extract just one organ: number = 0 (patientFile should be a single file)
	
	ActorList = []
	Points = []
	Outputs = []
	if number ==0:
		ActorList.append(ExtractOrgan(path,patientFile,3)[0])
		Points.append(ExtractOrgan(path,patientFile,3)[1])
		Outputs.append(ExtractOrgan(path,patientFile,3)[2])

	else:
		for i in patientFile:
			ActorList.append(ExtractOrgan(path,i,3)[0])
			Points.append(ExtractOrgan(path,i,3)[1])
			Outputs.append(ExtractOrgan(path,i,3)[2])

		
	return ActorList,Points,Outputs
	
def ExtractRectums(path,patientFile,number):
	#Rectum: OrganLabel = 6
	#Extract all the organs: number = 1 (patientFile should be a list)
	#Extract just one organ: number = 0 (patientFile should be a single file)
	Points = []
	ActorList = []
	Outputs = []
	if number ==0:
		ActorList.append(ExtractOrgan(path,patientFile,4)[0])
		Points.append(ExtractOrgan(path,patientFile,4)[1])
		Outputs.append(ExtractOrgan(path,patientFile,4)[2])
	else:
		for i in patientFile:
			ActorList.append(ExtractOrgan(path,i,4)[0])
			Points.append(ExtractOrgan(path,i,4)[1])
			Outputs.append(ExtractOrgan(path,i,4)[2])
		
	return ActorList,Points,Outputs
	
def Initial(Path,FileName):

    numOrgans = 8
    lut = MakeLUT(numOrgans)
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(Path+FileName)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    
    Points = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
    
    OutPut = reader.GetOutput()
	
    #--- Set Mapper ---#
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(reader.GetOutput())
    mapper.SetLookupTable(lut)
    mapper.SetScalarVisibility(1)
    mapper.SetScalarRange(0, numOrgans-1)
    mapper.SetScalarModeToUseCellData()
    mapper.Update()
    mapper.Update()

    ##---Set Actor---##
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    return actor, Points, OutPut

def ShapeVariability(path,patientList,model,OrganLabel):
	MeanPoints = ExtractOrgan(path,model,Organlabel)[1]
	for idx,i in enumerate(patientList):
		if idx==0:
			Org  = ExtractOrgan(path,i,Organlabel)[1]
			Matrx = Org-MeanPoints
			M = np.dot(Matrx[:,np.newaxis],Matrx[np.newaxis,:])
		else:
			Org  = ExtractOrgan(path,i,Organlabel)[1]
			Matrx = Org-MeanPoints
			Mat = np.dot(Matrx[:,np.newaxis],Matrx[np.newaxis,:])
			M = M + Mat
	
	prod = 1/8
	M = prod*M
	vaps = np.linalg.svd(M, full_matrices=1, compute_uv=0)
	return M ,vaps
def CleanUnusedPoints(InputFile):
	cleaner = vtk.vtkCleanPolyData()
	cleaner.ConvertPolysToLinesOff()
	cleaner.ConvertLinesToPointsOff()
	cleaner.ConvertStripsToPolysOff()
	cleaner.SetInputData(InputFile)
	cleaner.Update()
	return cleaner.GetOutput()
		
def PointToPointDistance(referencePoints,polyData,SpecColor):

    dataPoints = vtk_to_numpy(polyData.GetPoints().GetData())
    numDataPoints = polyData.GetNumberOfPoints()#int(len(dataPoints))
    refP = np.array(referencePoints)[0]
    #colors = vtk.vtkUnsignedCharArray()
    #colors.SetNumberOfComponents(3)
    #colors.SetName("colors")
	
    points = vtk.vtkPoints()
    distances = []

    mean = 0
    for idx,point in enumerate(dataPoints):
        
        dist = np.linalg.norm(point-refP[idx])
        distances.append(dist)
        mean = mean + dist
      

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
            colors.InsertNextTuple3(SpecColor[3][0],SpecColor[3][1],SpecColor[3][2]) #Red
        elif dist > distanceColorValues[1]:
            colors.InsertNextTuple3(SpecColor[2][0],SpecColor[2][1],SpecColor[2][2]) #Orange
        elif dist > distanceColorValues[2]:
            colors.InsertNextTuple3(SpecColor[1][0],SpecColor[1][1],SpecColor[1][2]) #Yellow
        else:
            colors.InsertNextTuple3(SpecColor[0][0],SpecColor[0][1],SpecColor[0][2]) #White

    mean = mean / numDataPoints
    return (distances, colors, distanceColorValues)
def ColorPoints(MeanDistancePerOrgan,SpecColor):
	
	distances = np.array(MeanDistancePerOrgan)
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
			colors.InsertNextTuple3(SpecColor[3][0],SpecColor[3][1],SpecColor[3][2]) #Red
		elif dist > distanceColorValues[1]:
			colors.InsertNextTuple3(SpecColor[2][0],SpecColor[2][1],SpecColor[2][2]) #Orange
		elif dist > distanceColorValues[2]:
			colors.InsertNextTuple3(SpecColor[1][0],SpecColor[1][1],SpecColor[1][2]) #Yellow
		else:
			colors.InsertNextTuple3(SpecColor[0][0],SpecColor[0][1],SpecColor[0][2]) #White
				
		
				
	return colors
def WholeVariationColors(MeanDistanceOrganesList,OrganList):
	vec = MeanDistanceOrganesList
	concatenated = []
	
	for list in vec:
		concatenated = np.concatenate((concatenated, list))	
	
	length = len(concatenated)
	concatenated = np.array(concatenated)
	indices = concatenated.argsort()
	sortedDistances = concatenated[indices]
	

	quarter = int(length/4)
	half = int(length/2)
	three_quarter = half+quarter

	distanceColorValues = [sortedDistances[three_quarter], sortedDistances[half], sortedDistances[quarter]]
	 
	
	for idx, i in enumerate(MeanDistanceOrganesList):
		distances = np.array(i)
		
		colors = vtk.vtkUnsignedCharArray()
		colors.SetNumberOfComponents(3)
		colors.SetName("colors")
		for dist in distances:
			if dist > distanceColorValues[0]:
				colors.InsertNextTuple3(0,76,153) #Red
			elif dist > distanceColorValues[1]:
				colors.InsertNextTuple3(0,128,255) #Orange
			elif dist > distanceColorValues[2]:
				colors.InsertNextTuple3(102,178,255) #Yellow
			else:
				colors.InsertNextTuple3(204,229,255) #White
		
		OrganList[idx].GetPointData().SetScalars(colors)
				
	return OrganList
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
def GetActor(data):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(data)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(1)

    return actor
def VolumesSurfaces(patientFile,model,path):
	Volumes = []
	Surfaces = []
	
	for i in range(1,5):
		Volumes.append(ExtractOrgan(path,model,i)[3])
		Surfaces.append(ExtractOrgan(path,model,i)[4])
		for j in patientFile:
			Volumes.append(ExtractOrgan(path,j,i)[3])
			Surfaces.append(ExtractOrgan(path,j,i)[4])
	
	return Volumes, Surfaces
def CreateActors(patientData):

    actors = []
    for i in range(0, len(patientData)):
        actors.append(GetActor(patientData[i]))

    return actors
def ColorList():
	B = [[255,255,204],[255,255,153],[255,255,0],[204,204,0]]
	P = [[255,229,204],[255,204,153],[255,178,102],[255,128,0]]
	SemV = [[255,255,255],[224,224,224],[192,192,192],[128,128,128]]
	R = [[255,204,229],[255,153,204],[255,102,178],[255,0,127]]
	
	Cl = []
	Cl.append(B); Cl.append(P); Cl.append(SemV); Cl.append(R); 
	return Cl

##---Data Properties ---##
path = "C:\\Users\\Maria\\Documents\\Master\\MedicalVisualization\\proyecto\\data\\"
model = "model.vtk"

##---Global Initial Variables ---##
count = 0

patientFile = []
idspat = [1,3,4,5,7,8,9,10,11]
for i in idspat:
	pat = "pat"+str(i)+".vtk"
	patientFile.append(pat)

#referencePoints = Initial(path,model)[1]
#referenceTree = spatial.cKDTree(referencePoints, leafsize=10)

##--- Reference OutPut DATA --- ##
BladderRef = ExtractBladders(path,model,0)[2]
ProstateRef  = ExtractProstates(path,model,0)[2]
SVRef = ExtractSeminalVesicles(path,model,0)[2]
RectumRef = ExtractRectums(path,model,0)[2]

OrganRefList = []
OrganRefList.append(BladderRef[0])
OrganRefList.append(ProstateRef[0])
OrganRefList.append(SVRef[0])
OrganRefList.append(RectumRef[0])
OrganRefActors = CreateActors(OrganRefList)
for actors in OrganRefActors:
	actors.GetProperty().SetRepresentationToWireframe()
	
BladderPoints = ExtractBladders(path,model,0)[1]
ProstatePoints  = ExtractProstates(path,model,0)[1]
SVPoints = ExtractSeminalVesicles(path,model,0)[1]
RectumPoints = ExtractRectums(path,model,0)[1]	

referencePoints = []
referencePoints.append(BladderPoints)
referencePoints.append(ProstatePoints)
referencePoints.append(SVPoints)
referencePoints.append(RectumPoints)
print (np.array(referencePoints[0])[0])

##--- PATIENT DATA ---##
BladdersOutPut = ExtractBladders(path,patientFile,1)[2]
Bladders = CreateActors(BladdersOutPut)

SeminalVesiclesOutPut = ExtractSeminalVesicles(path,patientFile,1)[2]
SeminalVesicles = CreateActors(SeminalVesiclesOutPut)

ProstatesOutPut = ExtractProstates(path,patientFile,1)[2]
Prostates = CreateActors(ProstatesOutPut)

RectumsOutPut = ExtractRectums(path,patientFile,1)[2]
Rectums = CreateActors(RectumsOutPut)

TotListOrgans = []
TotListOrgans.append(BladdersOutPut)
TotListOrgans.append(ProstatesOutPut)
TotListOrgans.append(SeminalVesiclesOutPut)
TotListOrgans.append(RectumsOutPut)

##--Calculate Volumes and Surfaces for Each Organ, including the Reference --##
(Volumes,Surfaces) = VolumesSurfaces(patientFile,model,path)
#for idx in range(0,len(Volumes)):
	#print ("Volume = ", Volumes[idx]) 
	#print ("Surface = ", Surfaces[idx])


##-- Calculating distances and colouring the cells
meanDistances = []
RGBList = ColorList()

for indx,OrganOutPut in enumerate(TotListOrgans):
	distances = []
	legends = []
	col = RGBList[indx]
	for i in range(0, len(OrganOutPut)):
		(dist, colors,legend) = PointToPointDistance(referencePoints[indx], OrganOutPut[i],col)
		distances.append(dist)
		legends.append(legend)
		OrganOutPut[i].GetPointData().SetScalars(colors)
	print ("for Patient 4 organs legend:",legends)
	meanDistances.append(sum(distances)/len(idspat))

ColoredOrganMeanList = list(OrganRefList) #Color each organ from the model with respect to the mean values
for i in range(0,len(OrganRefList)):
	colors = ColorPoints(meanDistances[i],RGBList[i])
	ColoredOrganMeanList[i].GetPointData().SetScalars(colors)

#print (ColoredOrganMeanList[0])	
OrganActors = CreateActors(ColoredOrganMeanList)

Bladders.append(OrganActors[0])
Prostates.append(OrganActors[1])
SeminalVesicles.append(OrganActors[2])
Rectums.append(OrganActors[3])

#OrganListMean = WholeVariationColors(meanDistances,OrganRefList)
#OrganActorsMean = CreateActors(OrganListMean)
		


#vaps = ShapeVariability(path,patientList,model,1)
#print(vaps)

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderer.SetBackground(1,1,1)
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetInteractorStyle(MyInteractorStyle())
renderWindowInteractor.SetRenderWindow(renderWindow)

renderer.AddActor(Initial(path,model)[0])
renderWindow.Render()
renderWindowInteractor.Start()