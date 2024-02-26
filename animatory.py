import imageio
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

###for plotting
##############################################
def openFields_timemarcher(file_name):
    with open(file_name,'rb') as load_file:
        time = np.load(load_file)
        bFromFile = np.load(load_file)
        uFromFile = np.load(load_file)
        vFromFile = np.load(load_file)
    return time, uFromFile, vFromFile, bFromFile

def makeChebPoints(N):
    points = np.zeros(N)
    points[0] = 1
    points[-1] = -1
    for i in range(1,N-1):
        points[i] = np.cos(((2*i-1)/(2*(N-2)))*np.pi)
    points = np.flip(points)
    return points

def makexPoints(alpha,N):
    return np.linspace(-1*np.pi/alpha,np.pi/alpha,N,endpoint=False)

def makeCoordArrs(alpha,Nx,Nz):
    xArr = makexPoints(alpha,Nx)
    zArr = makeChebPoints(Nz)
    return xArr, zArr
##############################################

def getDataFileNames(dataPath):
    #finds all data files
    dataPath = dataPath + 'fluidData*.npy'
    fileList = glob.glob(dataPath)
    fileNames = []
    times = []
    #creates a list of the filenames and corresponding times
    for fileName in fileList:
        dataFileName = fileName.split('/')[-1]
        fileNames.append(dataFileName)
        time = dataFileName.removesuffix('.npy')
        time = time.removeprefix('fluidData')
        times.append(float(time))
        
    #filenames are ordered by corresponding times
    correctOrder = np.argsort(times)
    times = [times[i] for i in correctOrder]
    fileNames = [fileNames[i] for i in correctOrder]
    return fileNames, times
        

def makePlots(dataPath,framesPath,fileNames):
    #plots each data file and saves it as an image which is a 'frame'
    xArr, zArr = makeCoordArrs(alpha,Nx,Nz)
    X,Z = np.meshgrid(xArr, zArr)
    framecount = 1
    for fileName in fileNames:
        time, uFromFile, vFromFile, bFromFile = openFields_timemarcher(dataPath+fileName)
        plotTitle = 'time = ' + str(time)
        plt.pcolormesh(X.T,Z.T,bFromFile.T, cmap='seismic')
        plt.title(plotTitle)
        plt.xlabel('x')
        plt.ylabel('z')
        axs = plt.gca()
        axs.set_aspect('equal')
        plt.colorbar(orientation='horizontal')
        image_name = framesPath+'frame' + str(framecount).zfill(10) + '.jpg'
        #image_name = 'animations/frame' + str(framecount).zfill(10) + '.jpg'
        plt.savefig(image_name, dpi=300)
        plt.clf()
        framecount += 1
    return True
        
# def makeGifFromFrames(framesPath):
#     #takes all frames and creates a GIF from it
#     image_folder = os.fsencode(framesPath)
#     filenames = []
#     for file in os.listdir(image_folder):
#         filename = os.fsencode(file)
#         if filename.endswith( ('.jpg', '.png', '.gif') ):
#             filename.append(os.path.join(framesPath,filename))
    
#     filename.sort()
    
#     images = [imageio.imread(f) for f in filename]
#     imageio.mimsave(os.path.join('movie.gif'),images,duration=0.04)
#     return True

def makeGifFromFrames(framesPath):
    file_names = sorted((fn for fn in os.listdir(framesPath) if fn.startswith('frame')))
    images = []
    for filename in file_names:
        images.append(imageio.imread(framesPath+filename))
        imageio.mimsave('movie.gif', images, duration = 0.04)

def makeGif(dataPath,framesPath):
    fileNames, times = getDataFileNames(dataPath)
    print("ordered files")
    makePlots(dataPath, framesPath, fileNames)
    print("created frames")
    print("creating GIF")
    makeGifFromFrames(framesPath)
    return True

Nx = 700
Nz = 300
alpha = 0.6283185307179586

##HAS TO END WITH A '/'
dataPath = '/Users/gudibanda/Desktop/Research/expHeating/Ra100000Pr7alpha0.6283185307179586Nx700Nz300_T100_runOutput/'
framesPath = '/Users/gudibanda/Desktop/Research/animation_tools/animations/'


makeGif(dataPath, framesPath)
