# Import PuLP modeler functions
from pulp import *
import pandas as pd
import numpy as np
from itertools import combinations
import math as mt
from random import *
import copy
from itertools import permutations

#InputData = "InputDataTelecomSmallInstance.xlsx"
InputData = "InputDataTelecomLargeInstance.xlsx"

if __name__ == "__main__":



  # Input Data Preparation #
   # Input Data Preparation #
   def read_excel_data(filename, sheet_name):
      data = pd.read_excel(filename, sheet_name=sheet_name, header=None)
      values = data.values
      if min(values.shape) == 1:  # This If is to make the code insensitive to column-wise or row-wise expression #
          if values.shape[0] == 1:
              values = values.tolist()
          else:
              values = values.transpose()
              values = values.tolist()
          return values[0]
      else:
          data_dict = {}
          if min(values.shape) == 2:  # For single-dimension parameters in Excel
              if values.shape[0] == 2:
                  for i in range(values.shape[1]):
                      data_dict[i+1] = values[1][i]
              else:
                  for i in range(values.shape[0]):
                      data_dict[i+1] = values[i][1]

          else:  # For two-dimension (matrix) parameters in Excel
              for i in range(values.shape[0]):
                  for j in range(values.shape[1]):
                      data_dict[(i+1, j+1)] = values[i][j]
          return data_dict



#Sets
customerNum = read_excel_data(InputData, "C")[0]
Set_C = [i for i in range(1,customerNum+1)]

endOfficeNum = read_excel_data(InputData, "M")[0]
Set_M = [i for i in range(1,endOfficeNum+1)]

digitalHubNum = read_excel_data(InputData, "N")[0]
Set_N = [i for i in range(1,digitalHubNum+1)]

alpha= read_excel_data(InputData, "alpha")
Hij= read_excel_data(InputData, "CustToTargetAllocCost(hij)")
Cjk= read_excel_data(InputData, "TargetToSteinerAllocCost(cjk)")
Gkm= read_excel_data(InputData, "SteinerToSteinerConnctCost(gkm)")
fk= read_excel_data(InputData, "SteinerFixedCost(fk)")
Ujmax= read_excel_data(InputData, "TargetCapicity(Uj)")
Vkmax= read_excel_data(InputData, "SteinerCapacity(Vk)")


#Xij=1 si Ai=j
#Yjk=1 si Bj=k
#Zkm=1 si Zk=m

###Variables####
def Xij(i,j,S):

    Ai=S["A"]
    if Ai[i-1]==j :
        return 1
    return 0

def Yjk(j,k,S):
    Bj=S["B"]
    if Bj[j-1]==k :
        return 1
    return 0

def Zkmf(k,m,S):
    Dk=S["D"]
    i=Dk.index(k)
    if i >0 :
        if  Dk[i-1] == m : 
            return 1
    elif Dk[-1] == m:
        return 1

    if i<len(Dk)-1:
        if Dk[i+1] == m :
            return 1
    elif Dk[0]==m :
        return 1

    return 0

def Zkm(k,m,S):
    Dk=S["D"]
    i=Dk.index(k)
    if Lk(k,S)==0 or Lk(m,S)==0 or m==k:
        return 0
    notEndCondition=True
    while notEndCondition :
        i=(i+1)%digitalHubNum
        if Lk(Dk[i],S) ==1 :
           notEndCondition=False
    if Dk[i] == m :
        return 1
    return 0



def Lk(k,S):
    Bj=S["B"]
    if k not in Bj :
        return 0
    return 1

def Wijk(i,j,k,S):
    return Xij(i,j,S)*Yjk(j,k,S)





def calcul_Z(S):


    return sum([sum([Xij(i,j,S)*Hij[i,j] for i in Set_C]) for j in Set_M ])+ sum([sum([Yjk(j,k,S)*Cjk[j,k] for j in Set_M]) for k in Set_N ])+  sum(fk[k-1]*Lk(k,S) for k in Set_N) + sum([sum([Gkm[k,m]*Zkm(k,m,S) for k in Set_N]) for m in Set_N ])





def respecte_contraintes(S):
    if contrainte1(S) :
        if contrainte2(S) :
            if contrainte3(S) :
                if contrainte4(S) :
                    if contrainte5(S) :
                        if contrainte6(S) :
                            if contrainte7(S) :
                                if contrainte8(S) :
                                    if contrainte9(S) :
                                        return True
    return False


def contrainte1(S) :
    for i in Set_C :
        if not sum([Xij(i,j,S) for j in Set_M]) <= 1 :
            return False
    return True


def contrainte2(S):
    for j in Set_M:
        if not sum([Yjk(j,k,S) for k in Set_N]) == 1 :
            return False
    return True

def contrainte3(S):
    for k in Set_N :
        for j in Set_M :
            if not Lk(k,S) >= Yjk(j,k,S) :
                return False
    return True


def contrainte4(S):
    for k in Set_N :
        if not sum([Zkm(k,m,S) for m in Set_N])+sum([Zkm(m,k,S) for m in Set_N])==2*Lk(k,S):
            return False
    return True


def contrainte5(S):
    #PN=list(permutations(Set_N)) cette méthode est trop longue bien que plus concise à écrire
    PN=[]

    for index in range(0,2**digitalHubNum) :#On crée les toutes les parties de N
        H=[]
        for element in Set_N :#Pour chaque élément du set
            if (index//(2**(element-1) ))%2==1 :
                H+=[element]
        length=len(H)
        if length>=3 :
            if length <= sum(Lk(k,S) for k in H)-1 :
                Tout_les_H+=[H]
    for H in PN :
        Set_t=[]
        for i in Set_N :
            if i not in H :
                Set_t+=[i]
        for t in Set_t :
            for l in H :
                Set_M_H_l=H[:]
                Set_M_H_l.remove(l)
                if not lpSum(sum([Zkm(k,m,S) for k in H]) for m in H)<= sum([Lk(j,S)+1 -Lk(t,S) for j in Set_j_H_l]) :
                    return False
    return True

def contrainte6(S):
    for j in Set_M :
        if not Ujmax[j-1] >= sum([Xij(i,j,S) for i in Set_C]):
            return False
    return True


def contrainte7(S):
    for k in Set_N:
        if not sum([sum([Wijk(i,j,k,S) for j in Set_M])for i in Set_C])<=Vkmax[k-1]:
            return False
    return True

def contrainte8(S):
    if not sum([Lk(k,S) for k in Set_N])>=3 :
        return False
    return True

def contrainte9(S):
    if not customerNum*alpha[0] <= sum([sum([Xij(i,j,S) for i in Set_C]) for j in Set_M ]):
        return False
    return True

def GenereVoisin_A(Svoisin,i):
    S=copy.deepcopy(Svoisin)
    A=S["A"]
    A[i//(endOfficeNum+1)]=i%(endOfficeNum+1)
    S["A"]=A
    return S





def GenereVoisin_B(Svoisin,i):
    S=copy.deepcopy(Svoisin)
    B=S["B"]
    B[i//(digitalHubNum+1)]=i%(digitalHubNum+1)
    S["B"]=B
    return S

def GenereVoisin_D(Svoisin,i,j):
    S=copy.deepcopy(Svoisin)
    D=S["D"]
    D[i],D[j]=D[j],D[i]
    S["D"]=D
    return S



def GenerateInitialSolution(Greedy) : #En fonction de la valeur Greedy, qui est un booléen on génère une solution aléatoire ou on utilise greedy
    Ai0=[]
    Bj0=[]
    Dk0=[]
    for i in range(customerNum):
        Ai0+=[randint(0,endOfficeNum)]
    for i in range(endOfficeNum):
        Bj0+=[randint(0,digitalHubNum)]
    if not Greedy :
        Dk0=range(1,digitalHubNum+1)[:]
        Dk0=sample(Dk0,len(Dk0))
    if Greedy :
        Dk0+=[1]
        current_vertex=Dk0[0]
        for i in range(1,digitalHubNum):
            l=[Gkm[current_vertex,m] for m in Set_N]
            l2=l[:]
            for i in Dk0 :
                l.remove(l2[i-1])
            current_vertex=l2.index(min(l))+1
            Dk0+=[current_vertex]
    S0={"A": Ai0, "B": Bj0, "D":Dk0}
    if not respecte_contraintes(S0):
        S0=GenerateInitialSolution(Greedy)
    return S0

def searchNeighbor(Sactuel,var):
    visible=True
    Z_min=calcul_Z(Sactuel)
    notBest=True
    i=0
    if var=="A":
        while i in range(customerNum*(endOfficeNum+1)) and notBest :
            Voisin_S=GenereVoisin_A(Sactuel,i)
            Z_Voisin_S=calcul_Z(Voisin_S)
            if Z_Voisin_S<Z_min and respecte_contraintes(Voisin_S):
                if visible :
                    print(Z_Voisin_S)
                notBest=False
                Sactuel=copy.deepcopy(Voisin_S)
                return Sactuel,notBest
            i+=1



    if var=="B":
        while i in range(endOfficeNum*(digitalHubNum+1)) and notBest :
            Voisin_S=GenereVoisin_B(Sactuel,i)
            Z_Voisin_S=calcul_Z(Voisin_S)
            if Z_Voisin_S<Z_min and respecte_contraintes(Voisin_S):
                if visible :
                    print(Z_Voisin_S)
                notBest=False
                Sactuel=copy.deepcopy(Voisin_S)
                return Sactuel,notBest
            i+=1
    if var =="D":
        while i in range(digitalHubNum) and notBest :
            j=i+1
            while j in range(digitalHubNum) and notBest :
                Voisin_S=GenereVoisin_D(Sactuel,i,j)
                Z_Voisin_S=calcul_Z(Voisin_S)
                if Z_Voisin_S<Z_min and respecte_contraintes(Voisin_S):
                    if visible :
                        print(Z_Voisin_S)
                    notBest=False
                    Sactuel=copy.deepcopy(Voisin_S)
                    return Sactuel,notBest
                j+=1
            i+=1
    return Sactuel,notBest

def LocalSearch(S,ordre):
    stillNeighbors=True
    Sactuel=copy.deepcopy(S)
    notBest=True
    while stillNeighbors :
        Sactuel,notBest=searchNeighbor(Sactuel,ordre[0])
        if notBest :
            Sactuel,notBest=searchNeighbor(Sactuel,ordre[1])
            if notBest :
                Sactuel,notBest=searchNeighbor(Sactuel,ordre[2])
                if notBest :
                    stillNeighbors=False
    print("Résultat de la localSearch :" ,Sactuel)
    return Sactuel
    
def Perturbation(S_etoile,history,methode): #en fonction de la méthode la perturbation sera faite différemment,
    global globalrecursif
    S={}
    if methode==1:
        i=randint(0,customerNum*(endOfficeNum+1)-1)
        A=GenereVoisin_A(S_etoile,i)["A"]
        i=randint(0,endOfficeNum*(digitalHubNum+1)-1)
        B=GenereVoisin_B(S_etoile,i)["B"]
        i=randint(0,digitalHubNum-1)
        j=randint(i,digitalHubNum-1)
        D=GenereVoisin_D(S_etoile,i,j)["D"]
        S= {"A":A,"B":B,"D":D}
    else :
        S=GenerateInitialSolution(False)
    if not respecte_contraintes(S) and globalrecursif<100 :
        globalrecursif+=1
        S=Perturbation(S,history,methode)
    if globalrecursif >=100 :
        globalrecursif =0
        return S_etoile
    globalrecursif =0
    return S

def AcceptanceCriterion(S_etoile,S_i_etoile,methode):
    Z_S_etoile=calcul_Z(S_etoile)
    Z_S_i_etoile=calcul_Z(S_i_etoile)
    if methode=="instable":#On prend le plus minimimum local le plus grand
        if Z_S_etoile>Z_S_i_etoile:
            return Z_S_etoile,S_etoile
        else :
            return Z_S_i_etoile,S_i_etoile
    if methode=="stable":#On prend le plus minimimum local le plus petit
        if not Z_S_etoile>Z_S_i_etoile:
            return Z_S_etoile,S_etoile
        else :
            return Z_S_i_etoile,S_i_etoile
    return Z_mini,S_i_etoile

###Main###
A=[5,8,4,2,8,5,4,4,1,3,2,2,5,0,1]
B=[1,3,3,3,4,4,3,1]
D=[6,5,1,2,3,4]
S0={"A": A, "B": B,"D":D}
globalrecursif=1
n=50
history={}
ordre=["B","D","A"]
S0=GenerateInitialSolution(True)
S_etoile=LocalSearch(S0,ordre)
history[1]=[S_etoile,calcul_Z(S_etoile)]
for i in range(n) :
    S_i = Perturbation(S_etoile,history,1)
    S_i_etoile=LocalSearch(S_i,ordre)
    history[i+2]=[S_i_etoile,calcul_Z(S_i_etoile)]
    Z_mini,S_etoile=AcceptanceCriterion(S_etoile,S_i_etoile,"stable")
print(history)
objective=[history[i][1] for i in range(1,len(history))]
print("Z_mini = ",min(objective))
print("S_min : ",history[objective.index(min(objective))+1][0])



