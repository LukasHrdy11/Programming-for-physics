from math import pi,atan,sqrt
from numba import njit

# Machin
@njit
def piMachin():
  return (4*atan(1/5)-atan(1/239))*4

# Leibniz
@njit
def piLeibniz(nmax):
  piLeibniz=0.
  z=1.
  for n in range(1,nmax+1):
    piLeibniz+=z/(2*n-1)
    z=-z
  return piLeibniz*4

# Euler: kumulace od nejmenších členů
@njit
def piEuler(nmax):
  piEuler=0.
  for n in range(nmax,0,-1):
    piEuler+=1/(n*n)
  return sqrt(piEuler*6)

# Viete
@njit
def piViete(nmax):
  piViete=1.
  a=0.
  for n in range(1,nmax+1):
    a=sqrt(0.5+0.5*a)
    piViete*=a
  return 2/piViete

# Ramanujan
@njit
def piRamanujan(nmax):
  a=1.
  piRamanujan=a*1103
  for n in range(1,nmax+1):
    a=a*(4*n-3)*(4*n-2)*(4*n-1)*4*n/(396.*n)**4
    piRamanujan+=a*(1103+26390*n)
  return 1/(piRamanujan*sqrt(8)/9801)