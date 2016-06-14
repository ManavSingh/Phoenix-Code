# -*- coding: utf-8 -*-
import os
import csv
from sys import exit
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as ani
plt.rcParams.update({'font.size':20})
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\msing_000\\Documents\\Grad\\Research\\Python Code\\ffmpeg\\bin\\ffmpeg'
np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth='nan')

Na = 6.022141E23     #Avogadro's Number
K = 1.38065E-16      #Boltzmann Constant
gamma = 5./3.        #Adiabatic Constant
pi = np.pi           #Define pi
Q0 = 1.0             #V.N.R. Artificial Viscosity Constant
G = 6.67E-8          #Universal Gravitational Constant
E0 = 1.4E-27         #Bremsstrahlung Cooling Source Constant, expected 1.E-27
Mns = 1.E33          #Neutron Star Mass
Rns = 1.E6           #Radius of the neutron star
Mdot = 1.E13         #Accretion Rate, originally 1E14
mu = 1.E30           #Magnetic Moment at surface of Neutron Star
x = 1.0              #Luminosity Constant, xE34 ergs/s is the Luminosity. Between 0.1 and 100
AccRad = 2.5E10      #Accretion Radius
RAtemp = 1.E4        #Initial temperature of plasma at accretion radius
ztop = 299           #Subscript of top zone in atmosphere
zbot = 0             #Subscript of bottom zone in atmosphere
zones = [i for i in range(zbot, ztop)]

radius = np.zeros([ztop+1,3])
plasmaVelocity = np.zeros([ztop+1,3])
specificVolume = np.zeros([ztop,3])
internalEnergy = np.zeros([ztop,3])
pressure = np.zeros([ztop,3])
density = np.zeros([ztop,3])
temperature = np.zeros([ztop,3])
artificialViscosity = np.zeros([ztop,3])
interfaceInertia = np.zeros(ztop+1)
mass = np.zeros(ztop)

dtp2 = .01  #dtp2 is timestep 
dtm2 = .01  #dtm2 is previous timestep

spos = 0    #Position of the shock, initialize at 0 in runner
accel = 0.0 #Stores the acceleration of the magnetopause

nGate = 0   #The number of loops the gate is open
gateOpen = 0
stepE = 0.0 #Luminosity in a given time step

Eflux = np.zeros(ztop+1)
Eflow = np.zeros(ztop+1)
TA = []
width = 1.*(AccRad-1.E9)/(ztop+1) #Initial zone width.
atmosphereTop = AccRad

runType = 0 #0: Full run, 1: plasma at rest test, 2: Blast wave test. Assigned in setup()
zoneAppended = 0 #Flag signifying whether a new zone was appended this loop

uMag = 0.0 #Velocity of the magnetopause
massAdded = 0.0
massDropped = 0.0
#We consider the magnetopause and the bottom interface of the plasma to be two separate objects
#These objects are generally at the same location except when the gate is open.

############################
# Status #
# Time-steps need to be very small, don't let a zone move farther than 1/200th of 
# the distance to the next zone. Less than that doesn't change much.
############################

def dynamicAtmosphere(U, R, V, E, P, Q, T, rho, DM, interfaceInertia, spos, nGate, Eflux, Eflow, loop):
    global stepE
    global gateOpen
    global uMag
    global massDropped
    #Accrete mass
    if(runType == 0):
        U,R,P,V,T,rho,Q,E,DM,interfaceInertia = AccreteMass(U,R,P,V,T,rho,Q,E,DM,interfaceInertia)

    for i in range(zbot+1, ztop): #Excludes first and last zones
        Area = 4.*pi*R[i,0]**2
        dVelocity = Area*(-(P[i,0]+Q[i,0])+(P[i-1,0]+Q[i-1,0]))*dtp2/interfaceInertia[i]-(G*Mns*dtp2/R[i,0]**2)
        if (runType == 3): #If Cooling Test
            dVelocity = 0
        U[i,2] = U[i,0] + dVelocity
    
    if (runType != 2 and runType != 5): #If not Blast Wave and not Accretion Test
        #bPressure = ((2.75*mu/R[zbot,0]**3)**2)/(8.*pi) #Magnetic Pressure
        bPressure = ((mu/R[zbot,0]**3)**2)/(8.*pi) #Magnetic Pressure
        Area = 4.*pi*R[zbot,0]**2
        bcrMag = Area*(-(P[zbot,0]+Q[zbot,0])+bPressure)*dtp2/interfaceInertia[zbot]
        uMag = uMag + bcrMag
        U[zbot,2] = uMag
    elif(runType == 5):
        Area = 4.*pi*R[zbot,0]**2
        U[zbot,2] = U[zbot,0] - Area*(P[zbot,0]+Q[zbot,0])*dtp2/interfaceInertia[zbot]
        U[zbot,2] = U[zbot,2] - (G*Mns*dtp2/R[zbot,0]**2)
    else:
        U[zbot,0] = 0

    if (runType != 0 and runType != 5): #If not full run or Accretion Test
        #U[ztop,2] = (4*pi*R[ztop,0]**2)*(P[ztop-1,0]+Q[ztop-1,0])*dtp2/interfaceInertia[ztop] #Open the top
        U[ztop,2] = 0.0
    else:
        #U[ztop,2] = -np.sqrt(G*Mns/R[ztop,0])
        U[ztop,2] = U[ztop,0] - (G*Mns*dtp2/R[zbot,0]**2)
    #Compute position from velocity
    for i in range(zbot, ztop+1): #Runs over all zones
        R[i,2] = R[i,0] + U[i,0]*dtp2
        if(i != 0 and R[i,2] <= R[i-1,2]): exit("Interface "+str(i-1)+" is radially farther than interface "+str(i))    
    
    #Compute density from zone volume
    for i in range(zbot, ztop):
        rho[i,2] = DM[i]/((4*pi/3.)*(R[i+1,2]**3-R[i,2]**3))

#####Check for stability at magnetopause#####

    if (runType == 0): #If full run
        #Compute max velocity of B-field in the plasma
        temporaryVar = np.sqrt(2.0*G*Mns/R[zbot,0])
        temporaryVar2 = 1.0 - P[zbot,0]/(rho[zbot,0]*temporaryVar**2)
        if (temporaryVar2 > 0.0):
            vBmax = 0.5*temporaryVar*np.sqrt(temporaryVar2*(R[spos,0] - R[zbot,0])/R[zbot,0])
        else: vBmax = 0.0
        if (vBmax > np.sqrt(gamma*P[zbot,0]*V[zbot,0])): vBmax = np.sqrt(gamma*P[zbot,0]*V[zbot,0])
        
    maxRho = (1.3445*mu**2)/(4.*pi*G*Mns*R[zbot,2]**5)    
    #maxRho = 0.0
    #Check if the gate is open. If the gate is open, plasma can leak through the magnetopause
    if(rho[zbot,2] > maxRho and runType == 0):
        #Open the gate
        Area = 4.*pi*R[zbot,0]**2
        dUplasma = -Area*(P[zbot,0]+Q[zbot,0])*dtp2/interfaceInertia[zbot]-G*Mns*dtp2/R[zbot,0]**2
        if (not gateOpen): #Gate was not open in the last timestep
            firstInstanceGate = 1 #This is the first timestep with the gate open
            gateOpen = 1
        else: firstInstanceGate = 0 #Gate was open last timestep
        uPlasma = U[zbot,0] + dUplasma

        rPlasma = R[zbot,0] + uPlasma*dtp2

        if (R[zbot,2] < rPlasma):
            rPlasma = R[zbot,2]
        
        #Determining the amount of plasma that fell through the magnetopause
        dRatio = 1.0 #Ratio of density below magnetopause to above magnetopause
        dropVol = 4.0*pi/3.0 * (R[zbot,2]**3 - rPlasma**3) #Volume of plasma that falls through
        keepVol = 4.0*pi/3.0 * (R[zbot+1,2]**3 - R[zbot,2]**3) #Volume of plasma that remains
        if (dropVol > 0.0):
            rho[zbot,2] = DM[zbot]/(keepVol + dRatio*dropVol)
            #Drop plasma below R[zbot,2]
            massBelow = rho[zbot,2]*dRatio*dropVol #Mass of plasma that fell through
            if (massBelow < 0.0): exit("Mass below is negative for loop #",loop)
        else: massBelow = 0.0
        
        if (massBelow > 0.0):
            massDropped = massDropped + massBelow
            nGate = nGate+1
            stepE = stepE + (G*Mns*massBelow/Rns)/dtp2
            if(massBelow < DM[zbot]): #If there is enough mass in the bottom zone.
                DM[zbot] = DM[zbot]-massBelow
                rho[zbot,2] = DM[zbot]/((4*pi/3.)*(R[zbot+1,2]**3-R[zbot,2]**3))
                
                #The bottom zone has fallen below the magnetopause
                #Some of the plasma will now leave the bottom zone
                #That plasma will carry away some momentum and energy
                #We need to solve for the new momemntum and energy of the system
                #We are going from R[zbot+1,0] and rPlasma as our lowest zones to
                #rPlasma, R[zbot,0], and R[zbot+1,0] as our lowest zones
                tempMass = np.array([massBelow,DM[zbot],DM[zbot+1]])
                tempVel = np.zeros(2)
                tempInertia = np.zeros(3)
                Inertia(tempMass,tempInertia,0,1,len(tempMass),ztop)
                momentum = interfaceInertia[zbot]*uPlasma + (interfaceInertia[zbot+1]-tempInertia[2])*U[zbot+1,0]
                kineticEnergy = interfaceInertia[zbot]*uPlasma**2 + (interfaceInertia[zbot+1]-tempInertia[2])*U[zbot+1,0]**2
                Conserve(tempInertia[0],tempVel[0],tempInertia[1],tempVel[1],momentum,kineticEnergy,uPlasma)
                #tempVel[0] should be close to uPlasma
                Inertia(DM,interfaceInertia,zbot,zbot,zbot+3,ztop)
                if(uMag-tempVel[1] > np.sqrt(gamma*P[zbot,0]*V[zbot,0])):
                    uMag = np.sqrt(gamma*P[zbot,0]*V[zbot,0])+tempVel[1]#Interface cannot move faster than the speed limit.
                U[zbot,0] = tempVel[1]
            else: exit("Not enough mass in bottom zone")
        #end if for massBelow>0
    elif(gateOpen):
        #The gate is now closed but was open last time step
        lastInstanceGate = 1 #The gate was open last time step
        U[zbot,2] = U[zbot,0] + bcrMag
        R[zbot,2] = R[zbot,0] + U[zbot,0]*dtp2
        rho[zbot,2] = DM[zbot]/((4*pi/3.0) * (R[zbot+1,2]**3-R[zbot,2]**3))
        gateOpen = 0
    else: lastInstanceGate = 0
##############
    for i in range(zbot, ztop):
        V[i,2] = 1.0/rho[i,2]
        E[i,2] = E[i,0] - (Q[i,0]+P[i,0])*(V[i,2]-V[i,0])
        if(E[i,0] < (Q[i,0]+P[i,0])*(V[i,2]-V[i,0])): exit("Energy going negative in zone " + str(i)) 
        E[i,2] = E[i,2] - E0*np.sqrt(T[i,0])*(rho[i,0]*Na)**2 * dtp2*V[i,0] #Cooling Term
            #Recall E is specific energy, energy per unit mass
            #Cooling term needs the density to be in particles/cm^3 (assume hydrogen)
            #Taking molar mass of hydrogen to be 1.0 g/mol
        if (runType == 3): #If cooling test
            E[i,2] = E[i,0] - E0*np.sqrt(T[i,0])*(rho[i,0]*Na)**2 * dtp2*V[i,0] #Cooling Term
        T[i,2] = E[i,2]/(3.*Na*K)
        P[i,2] = (gamma-1)*E[i,2]*rho[i,2]
        if (T[i,2] <= 0): exit("T is nonsense")
        if (V[i,2] < V[i,0] and U[i,2]-U[i+1,2] > 0): Q[i,2] = Q0*rho[i,2]*(U[i,2]-U[i+1,2])**2
        else: Q[i,2] = 0.0 #Only non-zero when there is compression.
        if (Q[i,2] < 0): Q[i,2] = 0.0
        stabilityParam = Q0*(U[i,2]-U[i+1,2])*(dtp2/(R[i+1,2]-R[i,2]))
        if (stabilityParam > 0.5): exit("Diffusion Instability: " + str(stabilityParam))

    #Update Time Subscripts
    for i in range(zbot, ztop):
        #Current becomes old
        rho[i,1] = rho[i,0]
        R[i,1] = R[i,0]
        P[i,1] = P[i,0]
        V[i,1] = V[i,0]
        E[i,1] = E[i,0]
        T[i,1] = T[i,0]
        U[i,1] = U[i,0]
        Q[i,1] = Q[i,0]
        #New becomes current
        R[i,0] = R[i,2]
        rho[i,0] = rho[i,2]
        P[i,0] = P[i,2]
        V[i,0] = V[i,2]
        E[i,0] = E[i,2]
        T[i,0] = T[i,2]
        U[i,0] = U[i,2]
        Q[i,0] = Q[i,2]
    R[ztop,1] = R[ztop,0]
    R[ztop,0] = R[ztop,2]
    U[ztop,1] = U[ztop,0]
    U[ztop,0] = U[ztop,2]
    
    return U,R,P,V,T,rho,Q,E,DM,interfaceInertia
  #End Dynamic Atmosphere
  
def initTestCase4(U,R,P,V,T,rho,Q,E,DM,DM2):
    #Rest Test
    global G
    global E0
    global width
    G = 0
    E0 = 0
    width = 1.*(2.0E10-R[0,0])/(ztop+1)
    
    #Radius, Temperature
    for i in range(zbot+1, ztop+1):
        R[i,0] = R[i-1,0]+width
    #Density and Specific Volume
    rho[0,0] = 1.E-14
    V[0,0] = 1./rho[0,0]
    for i in range(zbot+1, ztop):
        rho[i,0] = rho[0,0]
        V[i,0] = V[0,0]
    #Viscosity, Energy, Pressure, Mass, Temperature, Velocity
    for i in range(zbot, ztop):
        T[i] = 1.E5
        Q[i] = 0.0
        E[i,0] = 3.*Na*K*T[i,0]
        P[i,0] = (gamma-1)*(E[i,0]*rho[i,0])
        DM[i] = rho[0,0]*((4*pi/3.)*(R[i+1,0]**3-R[i,0]**3))
        U[i,0] = 0
    #Inertia
    Inertia(DM, DM2, zbot, zbot, ztop, ztop)

def initTestCase3(U,R,P,V,T,rho,Q,E,DM,DM2):
    #CoolingTest
    global G, width
    G = 0
    width = 1.*(2.0E10-R[0,0])/(ztop+1)
    #Radius, Temperature
    for i in range(zbot+1, ztop+1):
        R[i,0] = R[i-1,0]+width
    #Density and Specific Volume
    rho[0,0] = 1.E-10
    V[0,0] = 1./rho[0,0]
    for i in range(zbot+1, ztop):
        rho[i,0] = rho[0,0]
        V[i,0] = V[0,0]
    #Viscosity, Energy, Pressure, Mass, Temperature, Velocity
    for i in range(zbot, ztop):
        T[i] = 1.E8
        Q[i] = 0.0
        E[i,0] = 3.*Na*K*T[i,0]
        P[i,0] = (gamma-1)*(E[i,0]*rho[i,0])
        DM[i] = rho[0,0]*((4*pi/3.)*(R[i+1,0]**3-R[i,0]**3))
        U[i,0] = 0
    #Inertia
    Inertia(DM, DM2, zbot, zbot, ztop, ztop)

def initTestCase2(U,R,P,V,T,rho,Q,E,DM,DM2, cooling = 0):
    #Blast Wave Test
    global G
    global E0
    G = 0
    if (cooling == 0): E0 = 0
            
    #Radius, Temperature
    for i in range(zbot+1, ztop+1):
        R[i,0] = R[i-1,0]+width
    #Density and Specific Volume
    rho[0,0] = 1.E3
    V[0,0] = 1./rho[0,0]
    for i in range(zbot+1, ztop):
        rho[i,0] = rho[0,0]
        V[i,0] = V[0,0]
    #Viscosity, Energy, Pressure, Mass, Temperature, Velocity
    for i in range(zbot, ztop):
        T[i] = 1.E1
        Q[i] = 0.0
        E[i,0] = 3.*Na*K*T[i,0]
        P[i,0] = (gamma-1)*(E[i,0]*rho[i,0])
        DM[i] = rho[0,0]*((4*pi/3.)*(R[i+1,0]**3-R[i,0]**3))
        U[i,0] = 0
    T[0:2,0] = 1.E9
    E[0:2,0] = 3.*Na*K*T[0,0]
    P[0:2,0] = (gamma-1)*(E[0,0]*rho[0,0])
    #Inertia
    Inertia(DM, DM2, zbot, zbot, ztop, ztop)

def initTestCase1(U,R,P,V,T,rho,Q,E,DM,DM2, cooling = 1):
    global E0
    global atmosphereTop
    global width
    if (cooling == 0): E0 = 0
    width = 1.*(2.0E10-R[0,0])/(ztop)
    
    #Radius, Temperature
    for i in range(zbot+1, ztop+1):
        R[i,0] = R[i-1,0]+width
    atmosphereTop = R[ztop,0]
    #Temperature, Velocity
    for i in range(zbot, ztop):
        T[i,0] = RAtemp
        U[i,0] = -np.sqrt(G*Mns/R[i,0])
    U[ztop,0] = -np.sqrt(G*Mns/R[ztop,0])
    #Density, Specific Volume, Viscosity, Energy, Pressure, Mass
    for i in range(zbot, ztop):
        rho[i,0] = Mdot/(4.*pi*.25*(R[i,0]+R[i+1,0])**2 * .5*abs(U[i,0])+U[i+1,0])#Needs Radii and Vels
        V[i,0] = 1.0/rho[i,0] #Needs Densities
        Q[i] = 0.0
        E[i,0] = 3.*Na*K*T[i,0]
        P[i,0] = (gamma-1)*(E[i,0]*rho[i,0])
        DM[i] = rho[i,0]*((4*pi/3.)*(R[i+1,0]**3-R[i,0]**3))
    #Inertia
    Inertia(DM, DM2, zbot, zbot, ztop, ztop)

def initTestCase0(U,R,P,V,T,rho,Q,E,DM,DM2):
    #Constant Mass, free fall density
    global atmosphereTop, width #Width defined as space between ztop and ztop-1
    initMass = 1.E13 
    #Radii
    A = np.sqrt(G*Mns)*3.*initMass/Mdot
    #R[0,0] is set by setup routine.
    for i in range(zbot,ztop):
        R[i+1,0] = (A*R[i,0]**(1.5) + R[i,0]**3)**(1./3)
        if (R[i+1,0] > AccRad): exit("Too many zones after zone #" + str(i+1))
        U[i+1,0] = -np.sqrt(G*Mns/R[i+1,0])
    U[zbot,0] = -np.sqrt(G*Mns/R[zbot,0])
    width = radius[ztop,0] - radius[ztop-1,0]
    atmosphereTop = radius[ztop,0]
    for i in range(zbot, ztop):
        T[i,0] = RAtemp
        rho[i,0] = Mdot/(4.*pi*R[i,0]**2 * abs(U[i,0]))
        V[i,0] = 1.0/rho[i,0] #Needs Densities
        Q[i] = 0.0
        E[i,0] = 3.*Na*K*T[i,0]
        P[i,0] = (gamma-1)*(E[i,0]*rho[i,0])
        DM[i] = initMass
    #Inertia
    Inertia(DM, DM2, zbot, zbot, ztop, ztop)

def initTestCase5(U,R,P,V,T,rho,Q,E,DM,DM2):
    #Accretion Test, Constant initial Mass, freefall Density
    global atmosphereTop, width #Width defined as space between ztop and ztop-1
    global mu, E0
    mu = 0
    E0 = 0
    
    initMass = 1.E13 
    #Radii
    A = np.sqrt(G*Mns)*3.*initMass/Mdot
    #R[0,0] is set by setup routine.
    for i in range(zbot,ztop):
        R[i+1,0] = (A*R[i,0]**(1.5) + R[i,0]**3)**(1./3)
        U[i+1,0] = -np.sqrt(G*Mns/R[i+1,0])
    U[zbot,0] = -np.sqrt(G*Mns/R[zbot,0])
    width = radius[ztop,0] - radius[ztop-1,0]
    atmosphereTop = radius[ztop,0]
    for i in range(zbot, ztop):
        T[i,0] = RAtemp
        rho[i,0] = Mdot/(4.*pi*R[i,0]**2 * abs(U[i,0]))
        V[i,0] = 1.0/rho[i,0] #Needs Densities
        Q[i] = 0.0
        E[i,0] = 3.*Na*K*T[i,0]
        P[i,0] = (gamma-1)*(E[i,0]*rho[i,0])
        DM[i] = initMass
    #Inertia
    Inertia(DM, DM2, zbot, zbot, ztop, ztop)
    
def AccreteMass2(U,R,P,V,T,rho,Q,E,DM,DM2):
    global ztop
    global zoneAppended
    global massAdded
    if (atmosphereTop - R[ztop,0] > 1.0*width): #If there is space to fit a new zone
        newR = R[ztop,0] + 1.0*width
        R = np.append(R,np.ones([1,3])*newR,0)
        ztop = ztop+1
        UFreefall = G*Mns*dtp2/R[ztop,0]**2
        U = np.append(U,np.ones([1,3])*UFreefall,0)
        rhoFreefall = Mdot/(4.*pi*R[ztop,0]**2*U[ztop,0])
        rho = np.append(rho,np.ones([1,3])*rhoFreefall,0)
        newV = 1./rho[ztop-1,0]
        V = np.append(V,np.ones([1,3])*newV,0)
        Q = np.append(Q,np.zeros([1,3]),0)
        T = np.append(T,np.ones([1,3])*RAtemp,0)
        newE = 3.*Na*K*RAtemp
        E = np.append(E,np.ones([1,3])*newE,0)
        newP = (gamma-1)*(E[ztop-1,0]*rho[ztop-1,0])
        P = np.append(P,np.ones([1,3])*newP,0)
        newM = rho[ztop-1,0]*(4./3 *pi*(R[ztop,0]**3-R[ztop-1,0]**3))
        DM = np.append(DM,newM)
        DM2 = np.append(DM2,1.0) #Values corrected with Inertia routine.
        zoneAppended = 1
        massAdded = massAdded + newM
    else:
        zoneAppended = 0
                
    Inertia(DM, DM2, zbot, zbot, ztop, ztop) #Needed for velocity   
    #Conserve energy and momentum, regardless of new zone or not.
    momentum = DM2[ztop]*U[ztop,0]+DM2[ztop-1]*U[ztop-1,0]
    KE = DM2[ztop]*U[ztop,0]**2+DM2[ztop-1]*U[ztop-1,0]**2
    Conserve(DM2[ztop-1],U[ztop-1,0],DM2[ztop],U[ztop,0],momentum,KE,U[ztop-1,0])
    
    if (abs(U[zbot-1,0]) > np.sqrt(gamma*P[ztop-2,0]*V[ztop-2,0])):
        U[ztop-1,0] = -np.sqrt(gamma*P[ztop-2,0]*V[ztop-2,0])
    
    if (abs(U[zbot,0]) > np.sqrt(gamma*P[ztop-1,0]*V[ztop-1,0])):
        U[ztop,0] = -np.sqrt(gamma*P[ztop-1,0]*V[ztop-1,0])
        
    return U,R,P,V,T,rho,Q,E,DM,DM2
    
    Inertia(DM, DM2, zbot, zbot, ztop, ztop) #Needed for velocity   
    #Conserve energy and momentum, regardless of new zone or not.
    momentum = DM2[ztop]*U[ztop,0]+DM2[ztop-1]*U[ztop-1,0]
    KE = DM2[ztop]*U[ztop,0]**2+DM2[ztop-1]*U[ztop-1,0]**2
    Conserve(DM2[ztop-1],U[ztop-1,0],DM2[ztop],U[ztop,0],momentum,KE,U[ztop-1,0])
    
    if (abs(U[zbot-1,0]) > np.sqrt(gamma*P[ztop-2,0]*V[ztop-2,0])):
        U[ztop-1,0] = -np.sqrt(gamma*P[ztop-2,0]*V[ztop-2,0])
    
    if (abs(U[zbot,0]) > np.sqrt(gamma*P[ztop-1,0]*V[ztop-1,0])):
        U[ztop,0] = -np.sqrt(gamma*P[ztop-1,0]*V[ztop-1,0])
        
    return U,R,P,V,T,rho,Q,E,DM,DM2

def Inertia(DM,interfaceInertia,zbot,zmag,top,ztop):
    for i in range(zbot+1,top):
        interfaceInertia[i] = 0.5*(DM[i-1] + DM[i])
    if (zmag >= zbot): interfaceInertia[zbot] = DM[zbot]*.5
    else: interfaceInertia[zbot] = .5*(DM[zbot-1]+DM[zbot])
    if (top == ztop): interfaceInertia[top] = .5*(DM[ztop-1])

def Conserve(A, x, B, y, C, D, x0):
    #Finds a solution to Ax + By = C
    #and Ax^2 + By^2 = D
    #nearest to x0
    root = A*B*(D*(A+B)-C**2)
    if (root < 0): exit("root is negative in Conserve")
    root = np.sqrt(root)
    x = (A*C - root)/(A*(A+B))
    x1 = (A*C + root)/(A*(A+B))
    if (abs(x0-x1) < abs(x0-x)): x = x1
    y = (C-A*x)/B

def Tyme3(U,R,P,V,T,rho,Q,E,DM,interfaceInertia,loop, t1, t2):
    #Calculates timestep based on velocities of zone interfaces
    #If bottom zone is too small, combines bottom two zones.
    global ztop
    
    abu = np.zeros(ztop+1)
    zt = np.zeros(ztop+1)
    mintim = np.zeros(ztop)
    soundSpeed = np.zeros(ztop)   
    
    for j in range(zbot, ztop):
        abu[j] = np.abs(U[j,0] - U[j+1,0])
        zt[j] = R[j+1,0] - R[j,0]
        soundSpeed[j] = np.sqrt(gamma*P[j,0]*V[j,0])
        mintim[j] = zt[j]/(abu[j]+soundSpeed[j] + np.sqrt(E[j,0]))
        
    magDt = (R[zbot+1,0]-R[zbot,0])/np.sqrt(gamma*P[0,0]*V[0,0])
    if (np.all(magDt<mintim[1:])):
    #if (np.all(mintim[0]<mintim[1:])):
        #Combine bottom zones
        oldMass = DM[zbot]
        oldInertia = interfaceInertia[zbot]
        DM[zbot] = DM[zbot+1]+oldMass
        Inertia(DM, interfaceInertia, zbot, zbot, zbot+3, ztop)
                
        DM[zbot+1] = 0.0
        rho[zbot+1,0] = DM[zbot]/((4.*pi/3)*(R[zbot+2,0]**3-R[zbot,0]**3))
        Q[zbot+1,0] = (Q[zbot,0]*V[zbot,0]+Q[zbot+1,0]*V[zbot+1,0])*rho[zbot+1,0]
        #Note V[zbot+1,0] has not been updated yet, we want the old value above
        #But rho has been updated, and we want the new value.
        V[zbot+1,0] = 1./rho[zbot+1,0]
        E[zbot+1,0] = (E[zbot,0]*DM[zbot]+E[zbot+1,0]*oldMass)/DM[zbot]
        T[zbot+1,0] = E[zbot+1,0]/(3.*Na*K)
        P[zbot+1,0] = (gamma-1)*E[zbot+1,0]*rho[zbot+1,0]        
        
        #Delete empty zone and second interface
        R = np.delete(R,1,0)
        U = np.delete(U,1,0)
        V = np.delete(V,1,0)
        P = np.delete(P,1,0)
        T = np.delete(T,1,0)
        rho = np.delete(rho,1,0)
        E = np.delete(E,1,0)
        DM = np.delete(DM,1,0)
        interfaceInertia = np.delete(interfaceInertia,1,0)
        Q = np.delete(Q,1,0)
        ztop = ztop-1
        
        #Now that bottom zones have been combined, call Tyme again.
        U,R,P,V,T,rho,Q,E,DM,interfaceInertia,t1,t2 = Tyme3(U,R,P,V,T,rho,Q,E,DM,interfaceInertia,loop,t1,t2)
        return U,R,P,V,T,rho,Q,E,DM,interfaceInertia,t1, t2
        
    mintm = np.amin(mintim)
    if (loop <= 10):
        t1 = .001*mintm
        t2 = t1
    else:    
        t1 = t2
        t2 = .05*mintm
        #We don't want the fastest interface moving more than this much of the distance to the next interface.
    
    return U,R,P,V,T,rho,Q,E,DM,interfaceInertia,t1, t2
    
def setup(case = 0,cooling = 1):
    global Mdot
    global runType
    if (case==2):
        print "Setup for Test Case 2 (Blast Wave Test)."
        Mdot = 0
        Rmatch = 0
        radius[0,0] = Rmatch
        runType = 2
        initTestCase2(plasmaVelocity,radius,pressure,specificVolume,temperature,density,artificialViscosity,internalEnergy,mass,interfaceInertia, cooling)
    
    elif(case==0):
        print "Setup for Full Run."
        radius[0,0] = 4.E9/x
        runType = 0
        #initTestCase1(plasmaVelocity,radius,pressure,specificVolume,temperature,density,artificialViscosity,internalEnergy,mass,interfaceInertia, cooling)
        initTestCase0(plasmaVelocity,radius,pressure,specificVolume,temperature,density,artificialViscosity,internalEnergy,mass,interfaceInertia)
        

    elif(case==3):
        print "Setup for Cooling Test (Test Case 3)"
        radius[0,0] = 4.E9/x
        Mdot = 0
        runType = 3
        initTestCase3(plasmaVelocity,radius,pressure,specificVolume,temperature,density,artificialViscosity,internalEnergy,mass,interfaceInertia)
    
    elif(case==4):
        print "Setup for Rest Test"
        radius[0,0] = 4.E9/x
        Mdot = 0
        runType = 4
        initTestCase4(plasmaVelocity,radius,pressure,specificVolume,temperature,density,artificialViscosity,internalEnergy,mass,interfaceInertia)
    
    elif(case==5):
        print "Setup for Accretion Test"
        radius[0,0] = 10.E9/x
        runType = 5
        initTestCase5(plasmaVelocity,radius,pressure,specificVolume,temperature,density,artificialViscosity,internalEnergy,mass,interfaceInertia)

def runSystem(runName = "test", runTime = 0, maxLoops = 0, shockCheck = 1, shockStop = 1, outputFreq = 100, minTime = 0):
    global dtp2, dtm2
    global spos
    global TA
    global stepE
    global plasmaVelocity,radius,pressure,specificVolume,temperature,density
    global artificialViscosity,internalEnergy,mass,interfaceInertia
    spos = 0
    loop = 0
    output = 0  #Flag indicating whether or not to write data to file
    TA = [0.0]
    
    if (runName != "test"):
        if (os.path.exists(runName)):
            exit("Folder of name " + runName + " already exists!")
        else: #Setup file and write initial conditions
            os.mkdir(runName)
            output = 1
            data = open(runName+"/output.txt", 'w')
            data.write(str(ztop) + "\n")
            data.write(printState(loop, TA[-1], spos, radius, plasmaVelocity, density, temperature, internalEnergy, pressure, artificialViscosity, mass, specificVolume))
            
    while TA[-1] < runTime:
        if (loop % outputFreq == 0 or TA[-1] <= minTime):
            print "\n"  
            print "Loop =", loop
            print "Simulation Time =", TA[-1], "seconds"
            print "Length of Arrays =", ztop
            print "Lowest zone is at", radius[0,0]
        if ztop > 4999: exit("Arrays are getting too large")
    
        plasmaVelocity,radius,pressure,specificVolume,temperature,density,artificialViscosity,internalEnergy,mass,interfaceInertia,dtm2,dtp2 = Tyme3(plasmaVelocity,radius,pressure,specificVolume,temperature,density,artificialViscosity,internalEnergy,mass,interfaceInertia,loop, dtm2, dtp2)
        if np.isnan(dtp2): exit("dtp2 is nan")
        if (dtp2<=0): exit("dtp2 is nonsensible")
        if (loop % outputFreq == 0 or TA[-1] <= minTime): 
            print "dtp2 is",dtp2
            maxRho = (1.3445*mu**2)/(4.*pi*G*Mns*radius[zbot,0]**5)
            print "Density ratio is", density[zbot,0]/maxRho
                
        if (gateOpen and (loop % outputFreq == 0 or TA[-1] <= minTime)): print "The gate is open."
        
        plasmaVelocity,radius,pressure,specificVolume,temperature,density,artificialViscosity,internalEnergy,mass,interfaceInertia = dynamicAtmosphere(plasmaVelocity,radius,specificVolume,internalEnergy,pressure,artificialViscosity,temperature,density,mass,interfaceInertia, spos, nGate, Eflux, Eflow,loop)
             
        if shockCheck:  #Find position of the shock wave
            size = int(ztop/10)
            kbot = spos - int(size/2)
            if (kbot < 0): kbot = 0
            if (kbot+size >= ztop): kbot = ztop-size
            Qsub = np.zeros(size)
            for i in range(0, len(Qsub)):
                Qsub[i] = artificialViscosity[i+kbot,0]
            spos = np.argmax(Qsub) + kbot
                
        if (loop % outputFreq == 0 or TA[-1] <= minTime): 
            print "Shock is at zone",spos
            print "Luminosity is", stepE            
        
        #Update time and write data
        TA.append(TA[-1] + dtp2)
        if (output and (loop % outputFreq == 0 or TA[-1] <= minTime)):
            data.write(printState(loop, TA[-1], spos, radius, plasmaVelocity, density, temperature, internalEnergy, pressure, artificialViscosity, mass, specificVolume))
        
        #Reset stepE
        stepE = 0.0
    
        #Increment loop
        loop = loop + 1
    
        if (radius[spos,0] >= .9*atmosphereTop and shockStop):
            print "Shock is too close to top of atmosphere"
            break
            
        if (radius[zbot,0] <= Rns):
            print "Bottom zone hit surface of neutron star."
            break
  
    print "\nComplete"
    if (output):
        data.write(printState(loop, TA[-1], spos, radius, plasmaVelocity, density, temperature, internalEnergy, pressure, artificialViscosity, mass, specificVolume))
        data.close()
    
def printState(loop, time, spos, R, U, rho, T, E, P, Q, M, V):
    state = "*****" + "\n " #Flag to denote start of new timestep.
    state += str(loop) + ' ' + str(time) + ' ' + str(spos) + " \n "
    state += np.array_str(R[:,0],max_line_width = 10000000).strip("[]") + " \n "
    state += np.array_str(U[:,0],max_line_width = 10000000).strip("[]") + " \n "
    state += np.array_str(rho[:,0],max_line_width = 10000000).strip("[]") + " \n "
    state += np.array_str(T[:,0],max_line_width = 10000000).strip("[]") + " \n "
    state += np.array_str(E[:,0],max_line_width = 10000000).strip("[]") + " \n "
    state += np.array_str(P[:,0],max_line_width = 10000000).strip("[]") + " \n "
    state += np.array_str(Q[:,0],max_line_width = 10000000).strip("[]") + " \n "
    state += np.array_str(mass,max_line_width = 10000000).strip("[]") + " \n "
    state += np.array_str(V[:,0],max_line_width = 10000000).strip("[]") + " \n "
    state += str(stepE) + " \n "
    return state

def readFile(fileName = "output.txt"):
    #Returns ztop, total number of loops, 2 1D arrays and 9 2D arrays.
    #Rows are constant time, Cols are constant zone.
    R = []
    U = []
    rho = []
    T = []
    E = []
    P = []
    Q = []
    mass = []
    V = []
    loops = []
    times = []
    spos = []
    L = []
    ztop = 99
    counter = -1
    with open(fileName, 'r') as data:
        reader = csv.reader(data, delimiter = ' ')
        for line in reader:
            line = filter(None,line)
            if (len(line) == 0): 
                print "EOF"
                break
            if (counter == -1):
                ztop = int(line[0])  #First line should contain 1 element
                counter = 0
                continue
            elif (line[0] == "*****"):
                counter = 0
                continue
            else:
                if (counter == 0):
                    loops.append(float(line[0]))
                    times.append(float(line[1]))
                    spos.append(float(line[2]))
                    counter += 1
                    continue
                elif (counter == 1):
                    R.append(np.array(map(float,line)))
                    counter += 1
                    continue
                elif (counter == 2):
                    U.append(np.array(map(float,line)))
                    counter += 1
                    continue
                elif (counter == 3):
                    rho.append(np.array(map(float,line)))
                    counter += 1
                    continue
                elif (counter == 4):
                    T.append(np.array(map(float,line)))
                    counter += 1
                    continue
                elif (counter == 5):
                    E.append(np.array(map(float,line)))
                    counter += 1
                    continue
                elif (counter == 6):
                    P.append(np.array(map(float,line)))
                    counter += 1
                    continue
                elif (counter == 7):
                    Q.append(np.array(map(float,line)))
                    counter += 1
                    continue
                elif (counter == 8):
                    mass.append(np.array(map(float,line)))
                    counter += 1
                    continue
                elif (counter == 9):
                    V.append(np.array(map(float,line)))
                    counter += 1
                    continue
                elif (counter ==10):
                    L.append(map(float,line))
                    counter += 1
                else: exit("Error: counter > 10")
    return ztop, loops, times, spos, np.array(R), np.array(U), np.array(rho), np.array(T), np.array(E), np.array(P), np.array(Q), np.array(mass), np.array(V), np.array(L)
    
def animatePlot(x, dataSet, times, frameTime, xlabel = '', ylabel = '', title = '', save=0, name = "animation.mp4", rate = 60, log = 0):
    #For when arrays don't vary in size
    fig = plt.figure()
    plt.xlim(np.min(x),np.max(x))
    plt.ylim(np.min(dataSet),np.max(dataSet))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if log:
        line, = plt.semilogy([],[],'k',lw=2)
    else: line, = plt.plot([],[], 'k', lw = 2)
    time = plt.figtext(.15, .85, '')
    
    def init():
        line.set_data([], [])
        return line,   
    def animate(i, dataSet, line, time, times):
        time.set_text("Time = " + str(times[i]) + " s")
        line.set_data(x, dataSet[i,:])
        return line,
    
    anim = ani.FuncAnimation(fig, animate, frames=len(dataSet), interval = frameTime, fargs = (dataSet, line, time, times), repeat = True, init_func = init, blit = False, repeat_delay = 1000)
    plt.draw()
    
    if (save):
        writer = ani.FFMpegWriter()
        anim.save(name, writer = writer, fps=rate, extra_args=['-vcodec', 'libx264'])

def animatePlot2(x, y, times, xlims, ylims, frameTime, flag=1, xlabel = '', ylabel = '', title = '', save=0, name = "animation.mp4"):
    #For when arrays vary in size
    fig = plt.figure()
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    line,=plt.plot([],[],'k',lw=2.0)
    time = plt.figtext(.15,.85,"")
    
    def init():
        line.set_data([],[])
        return line,
    def animate(i,y,line,time,times):
        time.set_text("Time = " + str(times[i]) + " s")
        if flag: line.set_data(x[i], y[i])
        else: line.set_data(x[i][:-1], y[i])
        return line,
        
    anim = ani.FuncAnimation(fig,animate,frames=len(times)-1,interval=frameTime,fargs=(y, line, time, times),repeat=1,init_func = init, blit = 0,repeat_delay = 1000)
    plt.draw()    
    
    if (save):
        writer = ani.FFMpegWriter()
        anim.save(name, writer = writer, extra_args=['-vcodec', 'libx264'])
    
def sposSpeed(spos, time, radii, P, V, U, postOffset = 5, order = 1):
    sposVel = np.zeros(len(spos))
    if (order == 1):
        for i in range(1, len(spos)-1):
            sposVel[i] = (-radii[i-1,spos[i-1]] + radii[i+1,spos[i+1]])/(time[i+1]-time[i-1])
        sposVel[0] = (-radii[0,spos[0]]+radii[1,spos[1]])/(time[1]-time[0])
        sposVel[-1] = (radii[-1,spos[-1]]-radii[-2,spos[-2]])/(time[-1]-time[-2])

    machNumber = [(U[i,spos[i]]-sposVel[i])/np.sqrt(gamma*P[i,spos[i]+postOffset]*V[i,spos[i]+postOffset]) for i in range(0, len(sposVel))]
    return np.array(sposVel), np.array(machNumber)

def jumpConditions(spos, sposVel, times, M, rho, U, T, P, preOffset=0, postOffset=5, plot = 0):
    #Predicted Jump conditions from Dr. Lea's Astr Notes
    theoRhoJump = (gamma+1)/(gamma-1 + 2./(M**2))
    theoVelJump = 1./theoRhoJump
    theoTJump = (5./16)*M**2 #This is for large M, see Lea Fluids notes.
    theoPJump = (1./(gamma+1))*(2*gamma*M**2 - (gamma-1))
    
    #Actual Jumps from data
    actualRhoJump = np.array([np.max(rho[i,:]/rho[i,spos[i]+postOffset]) for i in range(0,len(times))])
    actualVelJump = np.array([(sposVel[i]-np.max(U[i,:]))/(sposVel[i]-U[i,spos[i]+postOffset]) for i in range(0, len(times))])
    actualPJump = np.array([np.max(P[i,:]/P[i,spos[i]+postOffset]) for i in range(0,len(times))])
    actualTJump = []
    for i in range(0,len(times)):
        minTValue = T[0,0] #The Inital Temp of the Blast Zone
        for j in range(0,int(spos[i]-2*Q0)):
            if (np.min(T[i,j]/T[0,-2]) < minTValue): minTValue = np.min(T[i,j]/T[0,-2])
        actualTJump.append(minTValue)
    actualTJump=np.array(actualTJump)
    
    #4 2-D arrays. First index of each array determines predicted vs. actual.
    #Second index determines which time step.    
    rhoJump = np.array([theoRhoJump,actualRhoJump])
    velJump = np.array([theoVelJump,actualVelJump])
    TJump = np.array([theoTJump,actualTJump])
    PJump = np.array([theoPJump,actualPJump])
     
    if plot:
        plt.figure(1)
        plt.plot(times, rhoJump[0], 'k.', times, rhoJump[1], 'r.')
        plt.xlabel(r"Time ($s$)")
        plt.ylabel("Jump Ratio")
        plt.title("Density Jump Ratio")
        plt.legend(["Predicted","Actual"], loc=4)
        
        plt.figure(2)
        plt.plot(times, velJump[0], 'k.', times, velJump[1], 'r.')
        plt.xlabel(r"Time ($s$)")
        plt.ylabel("Jump Ratio")
        plt.title("Velocity Jump Ratio")
        plt.legend(["Predicted","Actual"], loc=1)
        
        plt.figure(3)
        plt.semilogy(times, TJump[0], 'k.', times, TJump[1], 'r.')
        plt.xlabel(r"Time ($s$)")
        plt.ylabel("Jump Ratio")
        plt.title("Temperature Jump Ratio")
        plt.legend(["Predicted ","Actual"], loc=1)
        
        plt.figure(4)
        plt.semilogy(times, PJump[0], 'k.', times, PJump[1], 'r.')
        plt.xlabel(r"Time ($s$)")
        plt.ylabel("Jump Ratio")
        plt.title("Pressure Jump Ratio")
        plt.legend(["Predicted","Actual"], loc=1)
    
    return rhoJump, velJump, TJump, PJump

def multiplot(R, y, times, t1, t2, t3, t4, yaxis = '', title='', flag = 1):
    f,axes = plt.subplots(2,2)
    ((ax1,ax2),(ax3,ax4)) = axes
    if flag:
        ax1.plot(R[t1][:-1],y[t1], lw=2)
        ax1.set_title("Time: "+str(times[t1])+"s")
        ax2.plot(R[t2][:-1],y[t2], lw=2)
        ax2.set_title("Time: "+str(times[t2])+"s")
        ax3.plot(R[t3][:-1],y[t3], lw=2)
        ax3.set_title("Time: "+str(times[t3])+"s")
        ax4.plot(R[t4][:-1],y[t4], lw=2)
        ax4.set_title("Time: "+str(times[t4])+"s")
    else:
        ax1.plot(R[t1][:],y[t1], lw=2)
        ax1.set_title("Time: "+str(times[t1])+"s")
        ax2.plot(R[t2][:],y[t2], lw=2)
        ax2.set_title("Time: "+str(times[t2])+"s")
        ax3.plot(R[t3][:],y[t3], lw=2)
        ax3.set_title("Time: "+str(times[t3])+"s")
        ax4.plot(R[t4][:],y[t4], lw=2)
        ax4.set_title("Time: "+str(times[t4])+"s")
    
    #Consider generalizing this to any [square] number of plots in the multiplot
    #
    #It doesn't even have to be a square number of plots...
    
    for axis in axes:
        for element in axis:
            element.set_xlabel(r"Radius $(cm)$")
            element.set_ylabel(yaxis)
            element.xaxis.label.set_fontsize(20)
            element.yaxis.label.set_fontsize(20)

    f.tight_layout()