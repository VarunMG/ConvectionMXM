import numpy as np
import matplotlib.pyplot as plt

def openFields(file_name):
    with open(file_name,'rb') as load_file:
        uFromFile = np.load(load_file)
        vFromFile = np.load(load_file)
        bFromFile = np.load(load_file)
        phiFromFile = np.load(load_file)
        dt = np.load(load_file)
    return uFromFile, vFromFile, bFromFile, phiFromFile, dt

def openFields_timemarcher(file_name):
    with open(file_name,'rb') as load_file:
        time = np.load(load_file)
        bFromFile = np.load(load_file)
        uFromFile = np.load(load_file)
        vFromFile = np.load(load_file)
    return time, uFromFile, vFromFile, bFromFile

def openNuData(file_name):
    with open(file_name,'rb') as load_file:
        tVals = np.load(load_file)
        NuVals = np.load(load_file)
    return tVals, NuVals

def makeChebPoints(N):
    kList = np.arange(1,N+1)
    points = np.cos(np.pi*((2*kList-1)/(2*N)))
    points = points[::-1]
    return points

def makexPoints(alpha,N):
    return np.linspace(-1*np.pi/alpha,np.pi/alpha,N,endpoint=False)

def makeCoordArrs(alpha,Nx,Nz):
    xArr = makexPoints(alpha,Nx)
    zArr = makeChebPoints(Nz)
    return xArr, zArr

def takeYAverage(verticalSlice,yVals):
    N = len(verticalSlice)
    avg = (1/2)*np.trapz(verticalSlice,yVals)
    return avg

def calcMeans(array,yVals,Nx):
    result = np.zeros(Nx)
    for i in range(Nx):
        result[i] = takeYAverage(array[:,i],yVals)
    return result
    
def calcWaveNums(alpha,Nx):
    kNums = np.zeros(Nx)
    for i in range(int(Nx/2)):
        kNums[i] = i
    for i in range(int(Nx/2),Nx):
        kNums[i] = i-Nx
    kVals = alpha*kNums
    return kVals

def calcSpectra(uArr,vArr,yVals,alpha,Nx):
    uMeans = calcMeans(uArr.T,yVals,Nx)
    vMeans = calcMeans(vArr.T,yVals,Nx)
    bMeans = calcMeans(bArr.T,yVals,Nx)
    
    ufft = np.fft.fft(uMeans)/Nx
    vfft = np.fft.fft(vMeans)/Nx
    bfft = np.fft.fft(bMeans)/Nx
    
    numShells = int(Nx/2)+1
    
    e_spectra = np.zeros(numShells)
    b_spectra = np.zeros(numShells)
    e_spectra[0] = 0.5*np.abs(ufft[0])**2 + 0.5*np.abs(vfft[0])**2
    b_spectra[0] = 0.5*np.abs(bfft[0])**2
    for i in range(1,numShells):
        e_spectra[i] = np.abs(ufft[i])**2 + np.abs(vfft[i])**2
        b_spectra[i] = np.abs(bfft[i])**2
    e_spectra = (2*np.pi/alpha)*e_spectra
    b_spectra = (2*np.pi/alpha)*b_spectra
    shells = np.arange(0,numShells)
    return shells, e_spectra, b_spectra

def plotFields(xArr,zArr, uArr,vArr,bArr):
    X,Z = np.meshgrid(xArr, zArr)
    
    ##just for coolness, cmap='coolwarm' also looks alright
        
    fig, axs = plt.subplots(2, 2)
    p1 = axs[0,0].pcolormesh(X.T,Z.T,bArr,cmap='seismic')
    axs[0,0].quiver(X.T,Z.T,uArr,vArr)
    fig.colorbar(p1,ax=axs[0,0])
    
    p2 = axs[0,1].pcolormesh(X.T,Z.T,bArr,cmap='seismic')
    fig.colorbar(p2,ax=axs[0,1])
    
    p3 = axs[1,0].contourf(X.T,Z.T,bArr,cmap='seismic')
    fig.colorbar(p3,ax=axs[1,0])
    
    speeds = np.sqrt(uArr**2+vArr**2)
    p4 = axs[1,1].pcolormesh(X.T,Z.T,speeds,cmap='RdBu')
    fig.colorbar(p4,ax=axs[1,1])
    

uArr, vArr, bArr, phiArr, dt = openFields('filename')

#put in the inputs used for simulation
Nx = 84
Nz = 120
alpha =  1.5585
xArr, zArr = makeCoordArrs(alpha,Nx,Nz)

plotFields(xArr,zArr,uArr,vArr,bArr)

shells, e_spectra, b_spectra = calcSpectra(uArr,vArr,zArr,alpha,Nx)

plt.figure()
plt.loglog(shells,e_spectra,label='velocity spectra')
plt.loglog(shells,b_spectra,label='thermal spectra')
plt.legend()
