#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sympy as sym
import numpy as np
import numba as nb

mixangle = .4910015 #Weinberg angle; I'll change the name in a bit but right now I want to finish all the Rs
x_values, w_values = np.polynomial.laguerre.laggauss(150)
x_valuese, w_valuese = np.polynomial.legendre.leggauss(150)
n = 150


# $$\displaystyle M_1' (\xi) = 2^5 G_F^2 (2 \sin^2 \theta_w + 1)^2 \left( \xi^2 - \frac{2 \sin^2 \theta_w}{2 \sin^2 \theta_w + 1} m_e^2\xi \right)$$

# In[2]:


a, b, x = sym.symbols('a,b,x')
GF, stw, me = sym.symbols('GF,stw,me') #we can see that stw is sin of theta_w SQUARED
M_1prime = 2**5 * GF**2 * (2 * stw + 1)**2 * ( x**2 - 2 * stw / (2 * stw + 1) *me**2*x )


# $$\displaystyle  M_1^{(1)} (p_1, E_2, E_3, q_3) = \int_{p_1+E_2-E_3-q_3}^{p_1+E_2-E_3+q_3} dy\, M_1' ( \xi = \frac{1}{2} \left[ (p_1 + E_2)^2 - m_e^2 - y^2 \right] )$$

# In[3]:


p1, E2, E3, q3 = sym.symbols('p1,E2,E3,q3')
y = sym.symbols('y')
M_1_1 = sym.integrate( M_1prime.subs(x, ((p1+E2)**2-me**2-y**2)/2), (y, p1+E2-E3-q3, p1+E2-E3+q3) )
M_11 = sym.lambdify((p1,E2,E3,q3,GF,stw,me),M_1_1)
M11 = nb.jit(M_11,nopython=True)


# $$\displaystyle  M_1^{(2)} (p_1, q_2) = \int_{p_1-q_2}^{p_1+q_2} dy\, M_1' ( \xi = \frac{1}{2} \left[ (p_1 + E_2)^2 - m_e^2 - y^2 \right] )$$

# In[4]:


q2 = sym.symbols('q2')
M_1_2 = sym.integrate( M_1prime.subs(x, ((p1+E2)**2-me**2-y**2)/2), (y, p1-q2, p1+q2) )
M_12 = sym.lambdify((p1,E2,q2,GF,stw,me),M_1_2)
M12 = nb.jit(M_12,nopython=True)


# $$\displaystyle  M_1^{(3)} (p_1, q_2) = \int_{E_3+q_3-p_1-E_2}^{p_1+q_2} dy\, M_1' ( \xi = \frac{1}{2} \left[ (p_1 + E_2)^2 - m_e^2 - y^2 \right] )$$

# In[5]:


M_1_3 = sym.integrate( M_1prime.subs(x, ((p1+E2)**2-me**2-y**2)/2), (y, E3+q3-p1-E2, p1+q2) )
M_13 = sym.lambdify((p1,E2,q2,E3,q3,GF,stw,me),M_1_3)
M13 = nb.jit(M_13,nopython=True)


# $$\displaystyle  M_1^{(4)} (p_1, q_2) = \int_{q_2-p_1}^{p_1+E_2-E_3+q_3} dy\, M_1' ( \xi = \frac{1}{2} \left[ (p_1 + E_2)^2 - m_e^2 - y^2 \right] )$$

# In[6]:


M_1_4 = sym.integrate( M_1prime.subs(x, ((p1+E2)**2-me**2-y**2)/2), (y, q2-p1, p1+E2-E3+q3) )
M_14 = sym.lambdify((p1,E2,q2,E3,q3,GF,stw,me),M_1_4)
M14 = nb.jit(M_14,nopython=True)


# $$\displaystyle M_2' (\xi) = 2^7 G_F^2 \sin^4 \theta_w \left( \xi^2 + \frac{2 \sin^2 \theta_w + 1}{2 \sin^2 \theta_w} m_e^2 \xi \right)$$

# In[7]:


M_2prime = 2**7 * GF**2 * (stw)**2 * ( x**2 + (2*stw + 1)/(2*stw) * me**2*x )


# $$\displaystyle  M_2^{(1)} (p_1, E_2, E_3, q_2) = \int_{p_1-E_3+E_2-q_2}^{p_1-E_3+E_2+q_2} dy\, M_2' ( \xi = \frac{1}{2} \left[ y^2 + m_e^2 - (p_1 - E_3)^2\right] )$$

# In[8]:


p1, E2, E3, q2 = sym.symbols('p1,E2,E3,q2')
y = sym.symbols('y')
M_2_1 = sym.integrate( M_2prime.subs(x, (y**2 + me**2 - (p1-E3)**2)/2), (y, p1-E3+E2-q2, p1-E3+E2+q2) )
M_21 = sym.lambdify((p1,E2,E3,q2,GF,stw,me),M_2_1)
M21 = nb.jit(M_21,nopython=True)


# $$\displaystyle  M_2^{(2)} (p_1, q_3) = \int_{p_1-q_3}^{p_1+q_3} dy\, M_2' ( \xi = \frac{1}{2} \left[ y^2 + m_e^2 - (p_1 - E_3)^2\right] )$$

# In[9]:


q3 = sym.symbols('q3')
M_2_2 = sym.integrate( M_2prime.subs(x, (y**2 + me**2 - (p1-E3)**2)/2), (y, p1-q3, p1+q3) )
M_22 = sym.lambdify((p1,E3,q3,GF,stw,me),M_2_2)
M22 = nb.jit(M_22,nopython=True)


# $$\displaystyle  M_2^{(3)} (p_1, E_2, q_2, E_3, q_3) = \int_{E_3-p_1-E_2+q_2}^{p_1+q_3} dy\, M_2' ( \xi = \frac{1}{2} \left[ y^2 + m_e^2 - (p_1 - E_3)^2\right] )$$

# In[10]:


M_2_3 = sym.integrate( M_2prime.subs(x, (y**2 + me**2 - (p1-E3)**2)/2), (y, E3-p1-E2+q2, p1+q3) )
M_23 = sym.lambdify((p1,E2,q2,E3,q3,GF,stw,me),M_2_3) #PAY ATTENTION TO THE ORDER of the variables
M23 = nb.jit(M_23,nopython=True)


# $$\displaystyle  M_2^{(4)} (p_1, E_2, q_2, E_3, q_3) = \int_{q_3-p_1}^{p_1-E_3+E_2+q_2} dy\, M_2' ( \xi = \frac{1}{2} \left[ y^2 + m_e^2 - (p_1 - E_3)^2\right] )$$

# In[11]:


M_2_4 = sym.integrate( M_2prime.subs(x, (y**2 + me**2 - (p1-E3)**2)/2), (y, q3-p1, p1-E3+E2+q2) )
M_24 = sym.lambdify((p1,E2,q2,E3,q3,GF,stw,me),M_2_4) #PAY ATTENTION TO THE ORDER of the variables
M24 = nb.jit(M_24,nopython=True)


# In[12]:


@nb.jit(nopython=True)
def trapezoid(array,dx):
    total = np.sum(dx*(array[1:]+array[:-1])/2)
    return total

@nb.jit(nopython=True)
def fv(p,T):
    return 1/(np.e**(p/T)+1)

@nb.jit(nopython=True)
def fe(E,T):
    return 1/(np.e**(E/T)+1)

@nb.jit(nopython=True)
def Fplus(p1,E2,E3,p4,T):
    return (1-fv(p1,T))*(1-fe(E2,T))*fe(E3,T)*fv(p4,T)

@nb.jit(nopython=True)
def Fminus(p1,E2,E3,p4,T):
    return fv(p1,T)*fe(E2,T)*(1-fe(E3,T))*(1-fv(p4,T))

@nb.jit(nopython=True)
def check0(plus,minus):
    return (plus-minus)/(plus+minus)


# In[13]:


@nb.jit(nopython=True)
def B1_11(p1,E2,T,f,boxsize): #p1 is momentum of incoming neutrino, E2 is energy of incoming electron, T is temp, f is array of f values
    a = .511
    B = E2
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M11_array = np.zeros(len(E3_array))
    M11_array[0] = M11(p1,E2,E3_array[0],q3_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M11_array[-1] = M11(p1,E2,E3_array[-1],q3_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M11_array[i+1] = M11(p1,E2,E3_array[i+1],q3_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M11_array
    integrandminus_array = Fminus_array*M11_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A1_11(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino,T is the temperature at the time of the collision,and f is the array of f values from UDC
    e3cut = (p1**2 + .511**2)**(1/2)
    E2_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E2_array)):
        Bplus, Bminus = B1_11(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus]) 


# In[14]:


@nb.jit(nopython=True)
def B2_11(p1,E2,T,f,boxsize): #p1 is momentum of incoming neutrino, E2 is energy of incoming electron, T is temp, f is array of f values
    q2 = (E2**2-.511**2)**(1/2)
    a = E2
    B = .5*(2*p1 + E2 - q2 + (.511**2)/(2*p1 + E2 - q2)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M12_value = M12(p1,E2,q2,1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)]         
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M12_value
    integrandminus_array = Fminus_array*M12_value
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A2_11(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E2_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0
    integralminus = 0
    for i in range (len(E2_array)):
        Bplus, Bminus = B2_11(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[15]:


@nb.jit(nopython=True)
def B3_11(p1,E2,T,f,boxsize):
    q2 = (E2**2-.511**2)**(1/2)
    a = .5*(2*p1 + E2 - q2 + (.511**2)/(2*p1 + E2 - q2)) #e2trans
    B = .5*(2*p1 + E2 + q2 + (.511**2)/(2*p1 + E2 + q2)) #e1lim
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M13_array = np.zeros(len(E3_array))
    M13_array[0] = M13(p1,E2,q2,E3_array[0],q3_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M13_array[-1] = M13(p1,E2,q2,E3_array[-1],q3_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M13_array[i+1] = M13(p1,E2,q2,E3_array[i+1],q3_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M13_array
    integrandminus_array = Fminus_array*M13_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A3_11(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E2_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0
    integralminus = 0
    for i in range (len(E2_array)):
        Bplus, Bminus = B3_11(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[16]:


@nb.jit(nopython=True)
def B4_11(p1,E2,T,f,boxsize): 
    q2 = (E2**2-.511**2)**(1/2)
    a = .511 #MeV
    B = .5*(2*p1 + E2 - q2 + (.511**2)/(2*p1 + E2 - q2)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M11_array = np.zeros(len(E3_array))
    M11_array[0] = M11(p1,E2,E3_array[0],q3_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M11_array[-1] = M11(p1,E2,E3_array[-1],q3_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M11_array[i+1] = M11(p1,E2,E3_array[i+1],q3_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0]-boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M11_array
    integrandminus_array = Fminus_array*M11_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A4_11(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = .511 + (2*p1**2)/(.511-2*p1)
    E2_array = ((e1cut-e3cut)/2)*x_valuese + (e1cut+e3cut)/2
    integralplus = 0
    integralminus = 0
    for i in range (len(E2_array)):
        Bplus, Bminus = B4_11(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[17]:


@nb.jit(nopython=True)
def B5_11(p1,E2,T,f,boxsize): 
    q2 = (E2**2-.511**2)**(1/2)
    a = .5*(2*p1 + E2 - q2 + (.511**2)/(2*p1 + E2 - q2)) #e2trans
    B = E2
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M14_array = np.zeros(len(E3_array))
    M14_array[0] = M14(p1,E2,q2,E3_array[0],q3_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M14_array[-1] = M14(p1,E2,q2,E3_array[-1],q3_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M14_array[i+1] = M14(p1,E2,q2,E3_array[i+1],q3_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M14_array
    integrandminus_array = Fminus_array*M14_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A5_11(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = .511 + (2*p1**2)/(.511-2*p1)
    E2_array = ((e1cut-e3cut)/2)*x_valuese + (e1cut+e3cut)/2
    integralplus = 0
    integralminus = 0
    for i in range (len(E2_array)):
        Bplus, Bminus = B5_11(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[18]:


@nb.jit(nopython=True)
def B6_11(p1,E2,T,f,boxsize): 
    q2 = (E2**2-.511**2)**(1/2)
    a = E2
    B = .5*(2*p1 + E2 + q2 + (.511**2)/(2*p1 + E2 + q2)) #e1lim
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M13_array = np.zeros(len(E3_array))
    M13_array[0] = M13(p1,E2,q2,E3_array[0],q3_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M13_array[-1] = M13(p1,E2,q2,E3_array[-1],q3_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M13_array[i+1] = M13(p1,E2,q2,E3_array[i+1],q3_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M13_array
    integrandminus_array = Fminus_array*M13_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A6_11(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = .511 + (2*p1**2)/(.511-2*p1)
    E2_array = ((e1cut-e3cut)/2)*x_valuese + (e1cut+e3cut)/2
    integralplus = 0
    integralminus = 0
    for i in range (len(E2_array)):
        Bplus, Bminus = B6_11(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[19]:


@nb.jit(nopython=True)
def B7_11(p1,E2,T,f,boxsize): 
    q2 = (E2**2-.511**2)**(1/2)
    a = .5*(2*p1 + E2 - q2 + (.511**2)/(2*p1 + E2 - q2)) #e2lim
    B = E2
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
        
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = (E3_array**2-.511**2)**(1/2)
    q3_array[-1] = 0
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M14_array = np.zeros(len(E3_array))
    M14_array[0] = M14(p1,E2,q2,E3_array[0],q3_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M14_array[-1] = M14(p1,E2,q2,E3_array[-1],q3_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M14_array[i+1] = M14(p1,E2,q2,E3_array[i+1],q3_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M14_array
    integrandminus_array = Fminus_array*M14_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A7_11(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = .511 + (2*p1**2)/(.511-2*p1)
    E2_array = x_values+e1cut
    Bplus_array = np.zeros(len(E2_array))
    Bminus_array = np.zeros(len(E2_array))
    integralplus = 0
    integralminus = 0
    for i in range (len(E2_array)):
        Bplus_array[i], Bminus_array[i] = B7_11(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus]) 


# In[20]:


@nb.jit(nopython=True)
def B8_11(p1,E2,T,f,boxsize): 
    q2 = (E2**2-.511**2)**(1/2)
    a = E2
    B = .5*(2*p1 + E2 + q2 + (.511**2)/(2*p1 + E2 + q2)) #e1lim
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M13_array = np.zeros(len(E3_array))
    M13_array[0] = M13(p1,E2,q2,E3_array[0],q3_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M13_array[-1] = M13(p1,E2,q2,E3_array[-1],q3_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M13_array[i+1] = M13(p1,E2,q2,E3_array[i+1],q3_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M13_array
    integrandminus_array = Fminus_array*M13_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A8_11(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = .511 + (2*p1**2)/(.511-2*p1)
    E2_array = x_values+e1cut
    Bplus_array = np.zeros(len(E2_array))
    Bminus_array = np.zeros(len(E2_array))
    integralplus = 0
    integralminus = 0
    for i in range (len(E2_array)):
        Bplus_array[i], Bminus_array[i] = B8_11(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus])


# $$\displaystyle  R_1^{(1)} = \frac{1}{2^4 (2\pi)^3 p_1^2}\left[\int_{m_e}^{E_{cut}^{(3)}} dE_2 \left[\int_{m_e}^{E_2}dE_3 F M_1^{(1)}\,  + \int_{E_2}^{E_{trans}^{(2)}}dE_3 F M_1^{(2)}\,  + \int_{E_{trans}^{(2)}}^{E_{lim}^{(1)}}dE_3 F M_1^{(3)}\, \right]\, \\ + \int_{E_{cut}^{(3)}}^{E_{cut}^{(1)}} dE_2 \left[\int_{m_e}^{E_{trans}^{(2)}}dE_3 F M_1^{(1)}\,  + \int_{E_{trans}^{(2)}}^{E_2}dE_3 F M_1^{(4)}\,  + \int_{E_2}^{E_{lim}^{(1)}}dE_3 F M_1^{(3)}\, \right] \\ + \int_{E_{cut}^{(1)}}^{\infty} dE_2 \left[\int_{E_{lim}^{(2)}}^{E_2}dE_3 F M_1^{(4)}\,  + \int_{E_2}^{E_{lim}^{(1)}}dE_3 F M_1^{(3)}\, \right]\, \right]$$

# In[21]:


@nb.jit(nopython=True)
def R11(p1,T,f,boxsize):
    coefficient = 1/(2**4*(2*np.pi)**3*p1**2)
    integral = (A1_11(p1,T,f,boxsize)+A2_11(p1,T,f,boxsize)+A3_11(p1,T,f,boxsize)+A4_11(p1,T,f,boxsize)
                +A5_11(p1,T,f,boxsize)+A6_11(p1,T,f,boxsize)+A7_11(p1,T,f,boxsize)+A8_11(p1,T,f,boxsize))
    if abs((integral[0]-integral[1])/(integral[0]+integral[1]))<10**-14:
        return 0
    else:
        return coefficient*(integral[0]-integral[1])


# In[22]:


@nb.jit(nopython=True)
def B1_12(p1,E2,T,f,boxsize): 
    a = .511
    B = E2
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M11_array = np.zeros(len(E3_array))
    M11_array[0] = M11(p1,E2,E3_array[0],q3_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M11_array[-1] = M11(p1,E2,E3_array[-1],q3_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M11_array[i+1] = M11(p1,E2,E3_array[i+1],q3_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
        
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
    
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M11_array
    integrandminus_array = Fminus_array*M11_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A1_12(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E2_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0
    integralminus = 0
    for i in range (len(E2_array)):
        Bplus, Bminus = B1_12(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus]) 


# In[23]:


@nb.jit(nopython=True)
def B2_12(p1,E2,T,f,boxsize):
    q2 = (E2**2-.511**2)**(1/2)
    a = E2
    B = .5*(2*p1 + E2 - q2 + (.511**2)/(2*p1 + E2 - q2)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M12_value = M12(p1,E2,q2,1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)]         
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M12_value
    integrandminus_array = Fminus_array*M12_value
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A2_12(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E2_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0
    integralminus = 0
    for i in range (len(E2_array)):
        Bplus, Bminus = B2_12(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus]) 


# In[24]:


@nb.jit(nopython=True)
def B3_12(p1,E2,T,f,boxsize):
    q2 = (E2**2-.511**2)**(1/2)
    a = .5*(2*p1 + E2 - q2 + (.511**2)/(2*p1 + E2 - q2)) #e2trans
    B = .5*(2*p1 + E2 + q2 + (.511**2)/(2*p1 + E2 + q2)) #e1lim
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M13_array = np.zeros(len(E3_array))
    M13_array[0] = M13(p1,E2,q2,E3_array[0],q3_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M13_array[-1] = M13(p1,E2,q2,E3_array[-1],q3_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M13_array[i+1] = M13(p1,E2,q2,E3_array[i+1],q3_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M13_array
    integrandminus_array = Fminus_array*M13_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A3_12(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E2_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0
    integralminus = 0
    for i in range (len(E2_array)):
        Bplus, Bminus = B3_12(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus]) 


# In[25]:


@nb.jit(nopython=True)
def B4_12(p1,E2,T,f,boxsize): 
    q2 = (E2**2-.511**2)**(1/2)
    a = .511 #MeV
    B = .5*(2*p1 + E2 - q2 + (.511**2)/(2*p1 + E2 - q2)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
        
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = (E3_array**2-.511**2)**(1/2)
    q3_array[-1] = 0
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M11_array = np.zeros(len(E3_array))
    M11_array[0] = M11(p1,E2,E3_array[0],q3_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M11_array[-1] = M11(p1,E2,E3_array[-1],q3_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M11_array[i+1] = M11(p1,E2,E3_array[i+1],q3_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M11_array
    integrandminus_array = Fminus_array*M11_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A4_12(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E2_array = x_values+e3cut #x_values and w_values defined above
    Bplus_array = np.zeros(len(E2_array))
    Bminus_array = np.zeros(len(E2_array))
    integralplus = 0
    integralminus = 0
    for i in range (len(E2_array)):
        Bplus_array[i], Bminus_array[i] = B4_12(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus])


# In[26]:


@nb.jit(nopython=True)
def B5_12(p1,E2,T,f,boxsize): 
    q2 = (E2**2-.511**2)**(1/2)
    a = .5*(2*p1 + E2 - q2 + (.511**2)/(2*p1 + E2 - q2)) #e2trans
    B = E2
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
        
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M14_array = np.zeros(len(E3_array))
    M14_array[0] = M14(p1,E2,q2,E3_array[0],q3_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M14_array[-1] = M14(p1,E2,q2,E3_array[-1],q3_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M14_array[i+1] = M14(p1,E2,q2,E3_array[i+1],q3_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
        
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M14_array
    integrandminus_array = Fminus_array*M14_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A5_12(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E2_array = x_values+e3cut #x_values and w_values defined above
    Bplus_array = np.zeros(len(E2_array))
    Bminus_array = np.zeros(len(E2_array))
    integralplus = 0
    integralminus = 0
    for i in range (len(E2_array)):
        Bplus_array[i], Bminus_array[i] = B5_12(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus])


# In[27]:


@nb.jit(nopython=True)
def B6_12(p1,E2,T,f,boxsize): 
    q2 = (E2**2-.511**2)**(1/2)
    a = E2
    B = .5*(2*p1 + E2 + q2 + (.511**2)/(2*p1 + E2 + q2)) #e1lim
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M13_array = np.zeros(len(E3_array))
    M13_array[0] = M13(p1,E2,q2,E3_array[0],q3_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M13_array[-1] = M13(p1,E2,q2,E3_array[-1],q3_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M13_array[i+1] = M13(p1,E2,q2,E3_array[i+1],q3_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
        
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
    
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M13_array
    integrandminus_array = Fminus_array*M13_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A6_12(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E2_array = x_values+e3cut #x_values and w_values defined above
    Bplus_array = np.zeros(len(E2_array))
    Bminus_array = np.zeros(len(E2_array))
    integralplus = 0
    integralminus = 0
    for i in range (len(E2_array)):
        Bplus_array[i], Bminus_array[i] = B6_12(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus])


# $$\displaystyle  R_1^{(2)} = \frac{1}{2^4 (2\pi)^3 p_1^2}\left[\int_{m_e}^{E_{cut}^{(3)}} dE_2 \left[\int_{m_e}^{E_2}dE_3 F M_1^{(1)}\,  + \int_{E_2}^{E_{trans}^{(2)}}dE_3 F M_1^{(2)}\,  + \int_{E_{trans}^{(2)}}^{E_{lim}^{(1)}}dE_3 F M_1^{(3)}\, \right]\, \\ + \int_{E_{cut}^{(3)}}^{\infty} dE_2 \left[\int_{m_e}^{E_{trans}^{(2}}dE_3 F M_1^{(1)}\,  + \int_{E_{trans}^{(2)}}^{E_2}dE_3 F M_1^{(4)}\,  + \int_{E_2}^{E_{lim}^{(1)}}dE_3 F M_1^{(3)}\, \right]\,\right]$$

# In[28]:


@nb.jit(nopython=True)
def R12(p1,T,f,boxsize):
    coefficient = 1/(2**4*(2*np.pi)**3*p1**2)
    integral = (A1_12(p1,T,f,boxsize)+A2_12(p1,T,f,boxsize)+A3_12(p1,T,f,boxsize)+A4_12(p1,T,f,boxsize)
                +A5_12(p1,T,f,boxsize)+A6_12(p1,T,f,boxsize))
    if abs((integral[0]-integral[1])/(integral[0]+integral[1]))<10**-14:
        return 0
    else:
        return coefficient*(integral[0]-integral[1])


# In[29]:


@nb.jit(nopython=True)
def B1_21(p1,E3,T,f,boxsize): 
    a = .511
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)]         
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
    
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A1_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E3_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B1_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[30]:


@nb.jit(nopython=True)
def B2_21(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M22_value = M22(p1,E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
    
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M22_value
    integrandminus_array = Fminus_array*M22_value
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A2_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E3_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B2_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[31]:


@nb.jit(nopython=True)
def B3_21(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    B = .5*(E3 - q3 - 2*p1 + (.511**2)/(E3 - q3 -2*p1)) #e1lim
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
        
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
    
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A3_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E3_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B3_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[32]:


@nb.jit(nopython=True)
def B4_21(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .511
    B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
        
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
    
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A4_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e3cut)/2)*x_valuese + (e3cut+e2cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B4_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[33]:


@nb.jit(nopython=True)
def B5_21(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
    
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A5_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e3cut)/2)*x_valuese + (e3cut+e2cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B5_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[34]:


@nb.jit(nopython=True)
def B6_21(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = .5*(E3 - q3 - 2*p1 + (.511**2)/(E3 - q3 -2*p1)) #e1lim
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
        
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
    
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A6_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e3cut)/2)*x_valuese + (e3cut+e2cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B6_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[35]:


@nb.jit(nopython=True)
def B7_21(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2lim
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
            
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A7_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-e2cut)/2)*x_valuese + (e1cut+e2cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B7_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-e2cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-e2cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[36]:


@nb.jit(nopython=True)
def B8_21(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    if (E3 - q3 - 2*p1)==0:
        B = T*100 #e1lim
    else:
        B = .5*(E3 - q3 - 2*p1 + (.511**2)/(E3 - q3 - 2*p1)) #e1lim
        
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
        
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A8_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    n = 100
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-e2cut)/2)*x_valuese + (e1cut+e2cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B8_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-e2cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-e2cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[37]:


@nb.jit(nopython=True)
def B9_21(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2lim
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
            
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A9_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = x_values+e1cut
    Bplus_array = np.zeros(len(E3_array))
    Bminus_array = np.zeros(len(E3_array))
    integralplus = 0
    integralminus = 0
    for i in range (len(E3_array)):
        Bplus_array[i], Bminus_array[i] = B9_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus]) 


# In[38]:


@nb.jit(nopython=True)
def B10_21(p1,E3,T,f,boxsize):  #in this case boxsize shouldn't really apply right?
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (len(E2_array)-1):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A10_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = x_values+e1cut
    Bplus_array = np.zeros(len(E3_array))
    Bminus_array = np.zeros(len(E3_array))
    integralplus = 0
    integralminus = 0
    for i in range (len(E3_array)):
        Bplus_array[i], Bminus_array[i] = B10_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus]) 


# In[39]:


@nb.jit(nopython=True)
def R21(p1,T,f,boxsize):
    coefficient = 1/(2**4*(2*np.pi)**3*p1**2)
    integral = (A1_21(p1,T,f,boxsize)+A2_21(p1,T,f,boxsize)+A3_21(p1,T,f,boxsize)+A4_21(p1,T,f,boxsize)
                +A5_21(p1,T,f,boxsize)+A6_21(p1,T,f,boxsize)+A7_21(p1,T,f,boxsize)+A8_21(p1,T,f,boxsize)
                +A9_21(p1,T,f,boxsize)+A10_21(p1,T,f,boxsize))
    if abs((integral[0]-integral[1])/(integral[0]+integral[1]))<10**-14:
        return 0
    else:
        return coefficient*(integral[0]-integral[1])


# In[40]:


@nb.jit(nopython=True)
def B1_22(p1,E3,T,f,boxsize): 
    a = .511
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
        
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A1_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E3_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B1_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[41]:


@nb.jit(nopython=True)
def B2_22(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M22_value = M22(p1,E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)]     
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
     
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M22_value
    integrandminus_array = Fminus_array*M22_value
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A2_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E3_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B2_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[42]:


@nb.jit(nopython=True)
def B3_22(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    B = .5*(E3 - q3 - 2*p1 + (.511**2)/(E3 - q3 -2*p1)) #e1lim
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)]     
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A3_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E3_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B3_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[43]:


@nb.jit(nopython=True)
def B4_22(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .511
    B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f[int(p4_array[i+1]/boxsize)]
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f[int(p4_array[i+1]/boxsize)])
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
        
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A4_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-e3cut)/2)*x_valuese + (e3cut+e1cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B4_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[44]:


@nb.jit(nopython=True)
def B5_22(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A5_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-e3cut)/2)*x_valuese + (e3cut+e1cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B5_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[45]:


@nb.jit(nopython=True)
def B6_22(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    if (E3 - q3 - 2*p1)<10**-10:
        B = T*100 #e1lim
    else:
        B = .5*(E3 - q3 - 2*p1 + (.511**2)/(E3 - q3 - 2*p1)) #e1lim
        
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
        
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)   
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A6_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-e3cut)/2)*x_valuese + (e3cut+e1cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B6_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[46]:


@nb.jit(nopython=True)
def B7_22(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .511
    B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A7_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e1cut)/2)*x_valuese + (e2cut+e1cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B7_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e1cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e1cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[47]:


@nb.jit(nopython=True)
def B8_22(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A8_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e1cut)/2)*x_valuese + (e2cut+e1cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B8_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e1cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e1cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[48]:


@nb.jit(nopython=True)
def B9_22(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (len(E2_array)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A9_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e1cut)/2)*x_valuese + (e2cut+e1cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B9_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e1cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e1cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[49]:


@nb.jit(nopython=True)
def B10_22(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2lim
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A10_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = x_values+e2cut
    Bplus_array = np.zeros(len(E3_array))
    Bminus_array = np.zeros(len(E3_array))
    integralplus = 0
    integralminus = 0
    for i in range (len(E3_array)):
        Bplus_array[i], Bminus_array[i] = B10_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus]) 


# In[50]:


@nb.jit(nopython=True)
def B11_22(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (len(E2_array)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A11_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = x_values+e2cut #x_values established above
    Bplus_array = np.zeros(len(E3_array))
    Bminus_array = np.zeros(len(E3_array))
    integralplus = 0
    integralminus = 0
    for i in range (len(E3_array)):
        Bplus_array[i], Bminus_array[i] = B11_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus]) 


# In[51]:


@nb.jit(nopython=True)
def R22(p1,T,f,boxsize):
    coefficient = 1/(2**4*(2*np.pi)**3*p1**2)
    integral = (A1_22(p1,T,f,boxsize)+A2_22(p1,T,f,boxsize)+A3_22(p1,T,f,boxsize)+A4_22(p1,T,f,boxsize)
                +A5_22(p1,T,f,boxsize)+A6_22(p1,T,f,boxsize)+A7_22(p1,T,f,boxsize)+A8_22(p1,T,f,boxsize)
                +A9_22(p1,T,f,boxsize)+A10_22(p1,T,f,boxsize)+A11_22(p1,T,f,boxsize))
    if abs((integral[0]-integral[1])/(integral[0]+integral[1]))<10**-14:
        return 0
    else:
        return coefficient*(integral[0]-integral[1])


# In[52]:


@nb.jit(nopython=True)
def B1_23(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .511
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus


@nb.jit(nopython=True)
def A1_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-.511)/2)*x_valuese + (e1cut+.511)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B1_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[53]:


@nb.jit(nopython=True)
def B2_23(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M22_value = M22(p1,E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M22_value
    integrandminus_array = Fminus_array*M22_value
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A2_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-.511)/2)*x_valuese + (e1cut+.511)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B2_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[54]:


@nb.jit(nopython=True)
def B3_23(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    B = .5*(E3 - q3 - 2*p1 + (.511**2)/(E3 - q3 -2*p1)) #e1lim
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A3_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-.511)/2)*x_valuese + (e1cut+.511)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B3_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[55]:


@nb.jit(nopython=True)
def B4_23(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .511
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A4_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e3cut-e1cut)/2)*x_valuese + (e1cut+e3cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B4_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-e1cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-e1cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[56]:


@nb.jit(nopython=True)
def B5_23(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M22_value = M22(p1,E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M22_value
    integrandminus_array = Fminus_array*M22_value
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A5_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e3cut-e1cut)/2)*x_valuese + (e1cut+e3cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B5_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-e1cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-e1cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[57]:


@nb.jit(nopython=True)
def B6_23(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    e2trans = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1))
    a = e2trans
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (len(E2_array)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A6_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e3cut-e1cut)/2)*x_valuese + (e1cut+e3cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B6_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-e1cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-e1cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[58]:


@nb.jit(nopython=True)
def B7_23(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .511
    B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A7_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e3cut)/2)*x_valuese + (e2cut+e3cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B7_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[59]:


@nb.jit(nopython=True)
def B8_23(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A8_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e3cut)/2)*x_valuese + (e2cut+e3cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B8_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[60]:


@nb.jit(nopython=True)
def B9_23(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (len(E2_array)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A9_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e3cut)/2)*x_valuese + (e2cut+e3cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B9_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[61]:


@nb.jit(nopython=True)
def B10_23(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2lim
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A10_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = x_values+e2cut
    Bplus_array = np.zeros(len(E3_array))
    Bminus_array = np.zeros(len(E3_array))
    integralplus = 0
    integralminus = 0
    for i in range (len(E3_array)):
        Bplus_array[i], Bminus_array[i] = B10_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus])


# In[62]:


@nb.jit(nopython=True)
def B11_23(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (len(E2_array)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A11_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = x_values+e2cut #x_values established above
    Bplus_array = np.zeros(len(E3_array))
    Bminus_array = np.zeros(len(E3_array))
    integralplus = 0
    integralminus = 0
    for i in range (len(E3_array)):
        Bplus_array[i], Bminus_array[i] = B11_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus])


# In[63]:


@nb.jit(nopython=True)
def R23(p1,T,f,boxsize):
    coefficient = 1/(2**4*(2*np.pi)**3*p1**2)
    integral = (A1_23(p1,T,f,boxsize)+A2_23(p1,T,f,boxsize)+A3_23(p1,T,f,boxsize)
                +A4_23(p1,T,f,boxsize)+A5_23(p1,T,f,boxsize)+A6_23(p1,T,f,boxsize)
                +A7_23(p1,T,f,boxsize)+A8_23(p1,T,f,boxsize)+A9_23(p1,T,f,boxsize)
                +A10_23(p1,T,f,boxsize)+A11_23(p1,T,f,boxsize))
    if abs((integral[0]-integral[1])/(integral[0]+integral[1]))<10**-14:
        return 0
    else:
        return coefficient*(integral[0]-integral[1])


# In[64]:


@nb.jit(nopython=True)
def B1_24(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .511
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A1_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-.511)/2)*x_valuese + (e1cut+.511)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B1_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[65]:


@nb.jit(nopython=True)
def B2_24(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M22_value = M22(p1,E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (len(E2_array)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
 
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M22_value
    integrandminus_array = Fminus_array*M22_value
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A2_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-.511)/2)*x_valuese + (e1cut+.511)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B2_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[66]:


@nb.jit(nopython=True)
def B3_24(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .511
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
     
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A3_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e3cut-e1cut)/2)*x_valuese + (e1cut+e3cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B3_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-e1cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-e1cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[67]:


@nb.jit(nopython=True)
def B4_24(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    if (E3 + q3 - 2*p1)==0:
        B = T*100 #e2trans
    else:
        B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M22_value = M22(p1,E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
 
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M22_value
    integrandminus_array = Fminus_array*M22_value
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A4_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e3cut-e1cut)/2)*x_valuese + (e1cut+e3cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B4_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-e1cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-e1cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[68]:


@nb.jit(nopython=True)
def B5_24(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    if (E3 + q3 - 2*p1)==0:
        a = T*100 #e2trans
    else:
        a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (len(E2_array)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
 
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A5_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e3cut-e1cut)/2)*x_valuese + (e1cut+e3cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B5_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-e1cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-e1cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[69]:


@nb.jit(nopython=True)
def B6_24(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .511
    B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0],1.166*10**-11,(np.sin(mixangle))**2,.511)
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1],1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1],1.166*10**-11,(np.sin(mixangle))**2,.511)
     
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A6_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e3cut)/2)*x_valuese + (e2cut+e3cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B6_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[70]:


@nb.jit(nopython=True)
def B7_24(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
 
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A7_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e3cut)/2)*x_valuese + (e2cut+e3cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B7_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[71]:


@nb.jit(nopython=True)
def B8_24(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (len(E2_array)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
 
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A8_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e3cut)/2)*x_valuese + (e2cut+e3cut)/2
    integralplus = 0
    integralminus = 0
    for i in range(len(E3_array)):
        Bplus, Bminus = B8_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[72]:


@nb.jit(nopython=True)
def B9_24(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2lim
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    for i in range (int((B-a)/boxsize)):
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
     
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A9_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = x_values+e2cut
    Bplus_array = np.zeros(len(E3_array))
    Bminus_array = np.zeros(len(E3_array))
    integralplus = 0
    integralminus = 0
    for i in range (len(E3_array)):
        Bplus_array[i], Bminus_array[i] = B9_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus]) 


# In[73]:


@nb.jit(nopython=True)
def B10_24(p1,E3,T,f,boxsize): 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
    
    for i in range (len(E2_array)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = 0.0
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3,1.166*10**-11,(np.sin(mixangle))**2,.511)
 
    k = int(p4_array[-1]/boxsize)
    j = int(p4_array[0]/boxsize)+1
    f_first = 0
    f_last = 0
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = f[j-1] + ((f[j] - f[j-1])/boxsize)*(p4_array[0] - boxsize*(j-1))
    if k<len(f):
        f_last = f[k] + ((f[k+1] - f[k])/boxsize)*(p4_array[-1] - boxsize*k)
        
    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A10_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = x_values+e2cut #x_values established above
    Bplus_array = np.zeros(len(E3_array))
    Bminus_array = np.zeros(len(E3_array))
    integralplus = 0
    integralminus = 0
    for i in range (len(E3_array)):
        Bplus_array[i], Bminus_array[i] = B10_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus])


# In[74]:


@nb.jit(nopython=True)
def R24(p1,T,f,boxsize):
    coefficient = 1/(2**4*(2*np.pi)**3*p1**2)
    integral = (A1_24(p1,T,f,boxsize)+A2_24(p1,T,f,boxsize)+A3_24(p1,T,f,boxsize)+A4_24(p1,T,f,boxsize)
                +A5_24(p1,T,f,boxsize)+A6_24(p1,T,f,boxsize)+A7_24(p1,T,f,boxsize)+A8_24(p1,T,f,boxsize)
                +A9_24(p1,T,f,boxsize)+A10_24(p1,T,f,boxsize))
    print(integral)
    if abs((integral[0]-integral[1])/(integral[0]+integral[1]))<10**-14:
        return 0
    else:
        return coefficient*(integral[0]-integral[1])


# In[75]:


@nb.jit(nopython=True, parallel=True)
def driver(p_array,T,f_array,boxsize):
    boxsize = p_array[1]-p_array[0]
    output_array = np.zeros(p_array)
    for i in nb.prange (len(p_array)):
        if p_array[i]<.15791:
            output_array[i] = R11(p_array[i],T,f,boxsize)+R21(p_array[i],T,f,boxsize)
        elif p_array[i]<.18067:
            output_array[i] = R11(p_array[i],T,f,boxsize)+R22(p_array[i],T,f,boxsize)
        elif p_array[i]<.2555:
            output_array[i] = R11(p_array[i],T,f,boxsize)+R23(p_array[i],T,f,boxsize)
        else:
            output_array[i] = R12(p_array[i],T,f,boxsize)+R24(p_array[i],T,f,boxsize)
    return output_array

