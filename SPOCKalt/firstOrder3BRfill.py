import numpy as np
import matplotlib.pyplot as plt
from celmech.disturbing_function import laplace_b as b
import cmath
from scipy.optimize import fsolve
from sympy import Symbol, solve
import rebound
sim = rebound.Simulation()



def f2(l,a):
    return 1/8 *(-4*l**2*b(1/2,l,0,a) + 2*a*b(1/2,l,1,a) + a**2*b(1/2,l,2,a))
def f10(l,a):
    return -1/4* ((-4*l**2-6*l-2)*b(1/2,l+1,0,a) + 2*a*b(1/2,l+1,1,a) + a**2*b(1/2,l+1,2,a))
def f45(l,a):
    return 1/8* ((4*l**2+11*l+6)*b(1/2,l+2,0,a) + (4*l+6)*a*b(1/2,l+2,1,a) + a**2*b(1/2,l+2,2,a))
def f49(l,a):
    return -1/4* ((4*l**2+10*l+6)*b(1/2,l+1,0,a) + (4*l+6)*a*b(1/2,l+1,1,a) + a**2*b(1/2,l+1,2,a))
def f53(l,a):
    return 1/8* ((4*l**2+6*l+4)*b(1/2,l,0,a) + (4*l+6)*a*b(1/2,l,1,a) + a**2*b(1/2,l,2,a))



def g1l(k1,k3,a12,a23):
    def b12(k,n=0):
        return b(1/2,k,n,a12)
    def b23(k,n=0):
        return b(1/2,k,n,a23)
    
    nu12, nu23 = a12**(3/2), a23**(3/2)
    
    line1 = 1/(1-nu23)*((1-k1)*b12(1-k1)*b23(k3) 
                      + (2-k1)*a12*(b12(1-k1)*b23(k3,1) + b12(1-k1,1)*b23(k3)) # 1st-order deriv
                      + a12**2/2*(b12(1-k1,2)*b23(k3) + 2*b12(1-k1,1)*b23(k3,1) + b12(1-k1)*b23(k3,2)) # 2nd-order deriv
                      + (k1-1)/k3*(3/(2-2*nu23)*b23(k3) + a23*b23(k3,1)) * ((1-k1)*b12(1-k1) + a12/2*b12(1-k1,1))) # end of square bracket
    line2 = + f10(k1,a12)/(k1*(1/nu12-1)) * (k3*b23(k3) + a23/2*b23(k3,1))
    line3 = - f49(-k1,a12)/(k3+1-k3*nu23) * (-k3*b23(-k3) + a23/2*b23(-k3,1))
    return line1+line2+line3


def g1g(k1,k3,a12,a23):
    def b12(k,n=0):
        return b(1/2,k,n,a12)
    def b23(k,n=0):
        return b(1/2,k,n,a23)
    
    nu12, nu23 = a12**(3/2), a23**(3/2)
    
    line1 = 1/(1-nu23) * ((1-10*k1)/8*b12(-k1)*b23(k3) 
                        + (13/8-k1)*a12*(b12(-k1)*b23(k3,1) + b12(-k1,1)*b23(k3))
                        + a12**2/2*(b12(-k1,2)*b23(k3) + 2*b12(-k1,1)*b23(k3,1) + b12(-k1)*b23(k3,2))
                        + (k1-1)/k3*(3/(2-2*nu23)*b23(k3) + a23*b23(k3,1)) * ((1/2-k1)*b12(-k1) + a12/2*b12(-k1,1)))
    line2 = - f2(k1,a12)/(k1*(1/nu12-1)) * (k3*b23(k3) + a23/2*b23(k3,1))
    line3 = + 2*f53(-k1,a12)/(k3+1-k3*nu23) * (-k3*b23(-k3) + a23/2*b23(-k3,1))
    return line1+line2+line3



def g3l(k1,k3,a12,a23):
    def b12(k,n=0):
        return b(1/2,k,n,a12)
    def b23(k,n=0):
        return b(1/2,k,n,a23)
    
    nu12, nu23 = a12**(3/2), a23**(3/2)
    
    line1 = 1/(1/nu12-1) * ((k3-3)/4*b23(k3)*b12(k1)
                            + (k3+3/8)*a23 * (b23(k3)*b12(k1,1) + b23(k3,1)*b12(k1))
                            + a23**2/2 * (b23(k3,2)*b12(k1) + 2*b23(k3,1)*b12(k1,1) + b23(k3)*b12(k1,2))
                            + (k3-1)/k1*(3/(2/nu12-1)*b12(k1) + a12*b12(k1,1)) * (k3*b23(k3) + a23/2*b23(k3,1)))
    line2 = + f2(-k3,a23)/(k3*(1-nu23)) * ((1/2-k1)*b12(k1) + a12/2*b12(a12,1))
    line3 = - 2*f45(k3-2,a23)/(k1/nu12-k1-1) * ((k1+1/2)*b12(k1) + a12/2*b12(k1,2))
    return line1+line2+line3



def g3g(k1,k3,a12,a23):
    def b12(k,n=0):
        return b(1/2,k,n,a12)
    def b23(k,n=0):
        return b(1/2,k,n,a23)
    
    nu12, nu23 = a12**(3/2), a23**(3/2)
    
    line1 = 1/(1/nu12-1) * (k3*a23*(b23(k3)*b12(k1,1) + b23(k3,1)*b12(k1))
                            + a23**2/2* (b23(k3,2)*b12(k1) + 2*b23(k3,1)*b12(k1,1) + b23(k3)*b12(k1,2))
                            + (k3-1)/k1 * (3/(2/nu12-2)*b12(k1) + a12*b12(k1,1)) * ((k3-1/2)*b23(k3) + a23/2*b23(k3,1)))
    line2 = - f2(-k3,a23)/(k3*(1-nu23)) * ((1/2-k1)*b12(k1) + a12/2*b12(k1,1))
    line3 = + f49(k3-2,a23)/(k1/nu12-k1-1) * ((k1+1/2)*b12(k1) + a12/2*b12(k1,1))
    return line1+line2+line3


def rho_sub(sim,k1,k3, trio):

    ps = sim.particles
    a1, a2, a3 = [ps[i].a for i in trio]
    m1, m2, m3 = [ps[i].m for i in trio]
    n1, n2, n3 = [2*np.pi/(ps[i].P) for i in trio]
    e1, e2, e3 = [ps[i].e*cmath.exp(1j*ps[i].pomega) for i in trio]
    eps = m1+m2+m3
    Lambda1, Lambda2, Lambda3 = m1*np.sqrt(sim.G*a1), m2*np.sqrt(sim.G*a2), m3*np.sqrt(sim.G*a3)

    a12, a23, P12, P23 = a1/a2, a2/a3, (a1/a2)**(3/2), (a2/a3)**(3/2)
    k2 = 1-k1-k3
    G1L,G1G,G3L,G3G = g1l(k1,k3,a12,a23), g1g(k1,k3,a12,a23), g3l(k1,k3,a12,a23), g3g(k1,k3,a12,a23)
    gtilde = np.sqrt(2/Lambda1*G1L**2 + 2/Lambda2*(G1G-G3L)**2 + 2/Lambda3*G3G**2)
    R = sim.G/(2*a3)*(m1*m2*m3)/eps**2 * gtilde
    K2 = 3*(n1*k1**2/Lambda1 + n2*k2**2/Lambda2 + n3*k3**2/Lambda3)
    kappa = np.sqrt(2)*R/K2

    C1, C2, C3 = Lambda1*(1-np.sqrt(1-np.abs(e1)**2)), Lambda2*(1-np.sqrt(1-np.abs(e2)**2)), Lambda3*(1-np.sqrt(1-np.abs(e3)**2))
    if ((k1+k3-1)/k3*1/((m3/Lambda3)**3+k1/k3*(m1/Lambda1)**3)) < 0:
        return 0
    Lambda20 = ((k1+k3-1)/k3*1/((m3/Lambda3)**3+k1/k3*(m1/Lambda1)**3))**(1/3)*m2
    DeltaG = C1 + C2 + C3 + (Lambda20 - Lambda2)

    r1 = -G1L/gtilde*np.sqrt(2/Lambda1)
    r2 = (G1G-G3L)/gtilde*np.sqrt(2/Lambda2)
    r3 = G3G/gtilde*np.sqrt(2/Lambda3)
    rotation = np.array([[r1, r2, r3],
                         [r3/np.sqrt(1-r2**2), 0, -r1/np.sqrt(1-r2**2)],
                         [r1*r2/np.sqrt(1-r2**2), (r2**2-1)/np.sqrt(1-r2**2), r2*r3/np.sqrt(1-r2**2)]])
    x_vec = np.array([[np.sqrt(C1)*e1], [np.sqrt(C2)*e2], [np.sqrt(C3)*e3]])
    y_vec = rotation @ x_vec
    I1, I2, I3 = np.abs(y_vec[0][0])**2, np.abs(y_vec[1][0])**2, np.abs(y_vec[2][0])**2

    I0 = (eps**2*kappa)**(-2/3)*(DeltaG-I2-I3)
    if I0 < 3/2:
        return 0

    X3 = 2*np.sqrt(2*I0/3)*np.cos(1/3*np.arccos((3/(2*I0))**(3/2))-4*np.pi/3)
    DeltaI = 4 * (eps**2*kappa)**(2/3) * np.sqrt(abs(X3))
    
    Delta12 = 3*P12*abs(k1/Lambda1-k2/Lambda2)*DeltaI
    Delta23 = 3*P23*abs(k2/Lambda2-k3/Lambda3)*DeltaI  # double check eq.(45)
#     return Delta12, Delta23
    
    rho = 2*K2*(eps**2*kappa)**(2/3)*np.sqrt(abs(X3))/(n2*(1/P12-P23))
    return rho





def getfirst3brfill(sim,trio):
    num = 0
    ind = [i for i in range(-4,4) if i!=0]
    for k1 in ind:
        for k3 in ind:
            if not(k1 < 0 and k3 >0) and (k1+k3 !=1):
                num = num + rho_sub(sim,k1,k3,trio)
    return num