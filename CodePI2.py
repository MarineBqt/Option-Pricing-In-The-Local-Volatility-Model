import numpy as np
from scipy.stats import norm
import yfinance as yf 
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.optimize import least_squares
import datetime
import random


#Importer prix marché call/option  & Spot   
def Market_Importation(stock,K,T):
    selected_option=[]
    spot=yf.Ticker(stock).history(period='1d')['Close'][-1]
    option = yf.Ticker(stock).option_chain(T).calls
    implied_vol=[]
    for strike in K: 
        selected_option.append(option[option['strike'] == strike]['lastPrice'].iloc[0])
        implied_vol.append(option[option['strike'] == strike]['impliedVolatility'])
    #plt.plot(K,implied_vol)
    #plt.title("Market Implied Volatility")
    #plt.show()
    return selected_option,spot




#Implémentation de Black-Scholes
def BC_pricer(S, K, r, q ,sigma,T ,t , option):
    d1 = (np.log(S/K) + (r-q+(sigma ** 2)/2) * (T-t)) / (sigma * np.sqrt(T-t))
    d2 = d1 - sigma * np.sqrt(T-t)
    if option == "call":
      call = S*np.exp(-q*(T-t))*norm.cdf(d1)-K*np.exp(-r*(T-t))*norm.cdf(d2)
      return call 
    elif option == "put":
      put = K*np.exp(-r*(T-t))*norm.cdf(-d2) - S*np.exp(-q*(T-t))*norm.cdf(-d1) 
      return put 


####### Méthode des différences finies
#Diagonal Matrix Construction
def diagonal_matrix(sigma,Dupire): 
    B=0
    if Dupire :   #D de Cn+1  dans B.Cn=D.Cn+1
        b = (2./dt) + q + sigma**2 / (dx**2)
        a = ((r-q+0.5*sigma**2)/(2.*dx))-sigma**2/(2.*dx**2) 
        c = -((r-q+0.5*sigma**2)/(2.*dx))-sigma**2/(2.*dx**2) 
        b_B=(2./dt)-q-sigma**2/(dx**2)
        if(len(sigma)==1):
            D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Mspace, Mspace)).toarray() #Diagonal Matrix +condition aux limites
            B = sparse.diags([-a, b_B, -c], [-1, 0, 1], shape=(Mspace, Mspace)).toarray() #Diagonal Matrix +condition aux limites
        else :
            D = sparse.diags([a[1:], b, c], [-1, 0, 1], shape=(Mspace, Mspace)).toarray() #Diagonal Matrix +condition aux limites 
            B = sparse.diags([-a[1:], b_B, -c], [-1, 0, 1], shape=(Mspace, Mspace)).toarray() #Diagonal Matrix +condition aux limites
        D[0][0]=(2./dt) + q #bord gauche
        D[-1][-1]=(2./dt) + q #bord droit
        B[0][0]=(2./dt) - q #bord gauche
        B[-1][-1]=(2./dt) - q #bord droit
    else :
        a = (r-q-0.5*sigma**2)*(dt/(2*dx))-0.5*sigma**2*(dt/(dx*dx)) #Coef a,b,c  
        b = 1 + r*dt + sigma**2*(dt/(dx*dx))
        c = -(r-q-0.5*sigma**2)*(dt/(2*dx))-0.5*sigma**2*(dt/(dx*dx))
        D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Mspace, Mspace)).toarray() #Diagonal Matrix +condition aux limites
        D[0][0]=a+b #Condition au bord gauche
        D[-1][-1]=b+c+(r-q-sigma**2/2)*(dt/2*dx)
        D[-1][-2]=a-(r-q-sigma**2/2)*(dt/(2*dx))
    return D,B

  
#Solve the banded system  ay=b : on cherche x avec a la banded matrix
#Step 1 : Calculer la banded matrix a tel que a[u + i - j, j] == D[i,j]
def banded_matrix(D):
    m, n = D.shape
    a = np.zeros((1 + m, n))
    for i in range(m):
        for j in range(max(0, i - 1), min(n, i + 2)): #pour prendre que les indices valides dans D
            a[1 + i - j, j] = D[i, j]
    nonzero_rows= np.any(a != 0, axis=1) #lignes qui ne sont pas toutes nulles
    a = a[nonzero_rows]
    return a
   

#Plot : Tracer les prix de BS et vol locale à t=0 en fonction de S
def plot_surface(x,y,col,lab,xlab):
    plt.plot(x,y,color=col,label=lab)
    plt.xlabel(xlab)
    plt.ylabel("$C_0$")
    plt.legend()
    plt.title("Comparison of option pricing models")
    

def Price_Vol_locale(sigma,Dupire): #    aX=b
    if Dupire:
        b=np.maximum(1-np.exp(-x),0) #Point de départ (b)
    else:
        b=np.maximum(np.exp(x)-1,0) #prix du call à maturité =payoff
    for i in range(len(t)-1,-1,-1): #on parcourt l'interval de temps t: T.....0 (boucle en temps)
        D=diagonal_matrix(sigma,Dupire)[0] #calcul de la diagonal matrix en donnant vecteur de sigmas
        a=banded_matrix(D) #calcul banded matrix    
        if Dupire:
            B = diagonal_matrix(sigma,Dupire)[1]
            b=np.dot(B,b)
        else:
            c = np.zeros(Mspace)
            c[-1]= -(r-q-sigma**2/2*(1-1/dx))*dt*np.exp(max(x))
            b=b-c
        y=solve_banded((1, 1), a, b)#solver renvoie prix C au temps t 
        b=y
    if Dupire and len(sigma)>1:    
        y=np.interp(K, S*np.exp(-x)[::-1], S*y[::-1]) #interpolation: renvoie les prix de marché pour les K données
    return y #retourne prix du call à t=0 
    

def err_function(sigma,Price_Market,Dupire):
    C_vol_local=Price_Vol_locale(sigma,Dupire) 
    C_market=Price_Market 
    sigma_deriv = np.diff(sigma)*epsilon  # compute the derivative of sigma
    output=np.array(list(C_vol_local-C_market)+list(sigma_deriv))
    return output  #diff entre prix du call à t=0 pricer vol locale C0 - Prix marché BS à t=0
 
    
#Pricer : Trouver les bons sigmas qui minimisent l'erreur
def pricer(sigma_initial,Price_Market,Dupire):
    sigma=least_squares(err_function,sigma_initial, args=(Price_Market,Dupire),bounds=(0.01,1)) #Algo d'optimisation qui minimise l'erreur de la fonction err_function (appelé n fois) (renvoie le/les bon sigma) avec comme point de départ un (vecteur) de sigma
    return sigma #10^-6


########## #Modèle 1 : Pricer BS Volatilité Constante : on doit retrouver les prix de BS à t=0 ###########
# K=100
# Texp=1
# r=0.02
# q=0
# sigma_BS=0.2
# Mspace=100
# Ntime=100
# epsilon=0

# #Time discritisation
# t, dt = np.linspace(0, Texp, Ntime, retstep=True)       # time discretization

# #Space discritisation
# S_max = 3*float(K)               
# S_min = float(K)/3
# x_max = np.log(S_max/K)  # A2
# x_min = np.log(S_min/K)  # A1
# x, dx = np.linspace(x_min, x_max, Mspace, retstep=True)   # space discretization

# #Changement de variables
# S=K*np.exp(x)

# #BS Price
# C_BS=BC_pricer(S, K, r, q ,sigma_BS,Texp ,0 , "call")
# Price_Market=C_BS/K

# #Pricer
# sigma_initial=0.5 #sigma constant 
# sigma_optimal=pricer(sigma_initial,Price_Market,False) 
# print("Sigma Optimal (constant): ",sigma_optimal.x[0])

# #plot
# plot_surface(S,Price_Market,"darkblue","Market Price","S")
# plot_surface(S,Price_Vol_locale(sigma_optimal.x,False),"purple","Local Volatility Model","S")
# plt.show()

# erreur=abs(Price_Vol_locale(sigma_optimal.x,False)-Price_Market)
# plot_surface(S, erreur, "darkgreen", "erreur", "S")
# plt.show()


########### Modèle 2 : Pricer Dupire Volatilité Constante ###########
# Texp=1
# S=100
# r=0.05
# q=0.02
# sigma_BS=0.2
# Mspace=100
# Ntime=100
# epsilon=0

# #Time discritisation
# t, dt = np.linspace(0, Texp, Ntime, retstep=True)       # time discretization

# #Space discritisation
# x_max = 3
# x_min = -3
# x, dx = np.linspace(x_min, x_max, Mspace, retstep=True)   # space discretization


# #Changement de variables
# K=S*np.exp(-x)

# #BS Price
# C_BS=BC_pricer(S, K, r, q ,sigma_BS,Texp ,0 , "call")
# Price_Market=C_BS/S

# #Pricer
# sigma_initial=0.5 #sigma constant 
# sigma_optimal=pricer(sigma_initial,Price_Market,True) 
# print("Sigma Optimal (constant): ",sigma_optimal.x[0])

# #plot
# plot_surface(K,Price_Market,"darkblue","Market Price","K")
# plot_surface(K,Price_Vol_locale(sigma_optimal.x,True),"purple","Local Volatility Model","K")
# plt.show()

# #erreur
# erreur=abs(Price_Vol_locale(sigma_optimal.x,True)-Price_Market)
# plot_surface(K, erreur, "darkgreen", "erreur", "K")
# plt.show()




########## Modèle 2 : Pricer Dupire Volatilité Non Constante ###########

expiration_date="2023-07-21"
K=[100,105,110,115,120,125,130,135,140,145,150]


Ntime=50 
Mspace=100
r=0.02  #taux sans risque obligation sur yahoo
q=0
epsilon=10 #à jouer avec 

# #Maturité
today=datetime.date.today()
Texp = (datetime.datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.datetime.combine(today, datetime.datetime.min.time())).days /365

#Spot
S=Market_Importation("AMZN", K, expiration_date)[1] #SPY


#Time discritisation
t, dt = np.linspace(0, Texp, Ntime, retstep=True)       # time discretization

#Space discritisation
K_max=max(K)*3 
K_min=min(K)/3
x_max=round(np.log(S/K_min),2)
x_min=round(np.log(S/K_max),2)

x, dx = np.linspace(x_min, x_max, Mspace, retstep=True)   # space discretization

#Market Price
C_Market=Market_Importation("AMZN", K, expiration_date)[0]
Price_Market=C_Market#/S

#Pricer
sigma_initial=[0.2 for i in range(Mspace)]
sigma_optimal=pricer(sigma_initial,Price_Market,True)
sigma_at_K = np.interp(K, S*np.exp(-x)[::-1], sigma_optimal.x[::-1]) # sigmas at K
print("Sigma Optimal (pas constant) : ",sigma_at_K)


#plot
plot_surface(K,Price_Market,"darkblue","Market Price","K")
plot_surface(K,Price_Vol_locale(sigma_optimal.x,True),"purple","Local Volatility Model",'K')
plt.show()

#erreur
erreur=Price_Vol_locale(sigma_optimal.x,True)-Price_Market
plot_surface(K, erreur, "darkgreen", "erreur", "K")
plt.show()

#vol implicite
plt.plot(K,sigma_at_K)
plt.title("Volatility Smile")
plt.show()

#Ameliorer la fonction least square




