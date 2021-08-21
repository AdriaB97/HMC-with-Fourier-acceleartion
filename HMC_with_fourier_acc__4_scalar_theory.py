# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:27:49 2021

@author: usuari
"""

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy.ma as ma
import numpy as np
import argparse
import time
import os

#To use latex notation in matplotlib labels.
#Use symbols $ to use latex.

plt.rcParams.update({"text.usetex": True})

######################################################################
################### Functions for the Observables ####################
######################################################################

#The functions of the observables that are measured have to be defined at the beginning.
#The reason is that their name is used in the .json file and without its definition the program will not be able to evaluate the dictionary of the .json file.

#Temporal average of phi.
    
def phi_1(phi):
    
    return np.average(phi)

#Temporal average of square of phi.
    
def phi_2(phi):
    
    return np.average(phi**2)

#Temporal average of phi to the third.
    
def phi_3(phi):
    
    return np.average(phi**3)

#temporal average of phi to the fourth
    
def phi_4(phi):

    return np.average(phi**4)

#Terms of dimensionless kinetic energy.
    
def kin(phi,phi_tau_next,phi_x_next):
    
    return 1/2*(phi_tau_next-phi)**2 + 1/2*(phi_x_next-phi)**2

#Terms of dimensionless potential energy.
    
def pot(phi,m,lambd):
    
    return -1/2*m**2*phi**2 - lambd/np.math.factorial(4)*phi**4

#Temporal average of kinetic energy.

def kin_temp_av(x):
    
    #x is a list which contains two elements.
    #The first is the values of the points of the path.
    
    phi = x[0]
    
    #The second one is the number of points of the lattice
    
    N_lat = x[1]
    
    #Create two arrays with the values of phi displaced one row up or one column up.
    
    phi_tau_next = np.append(phi[1%N_lat:N_lat,:],phi[0:1%N_lat,:],axis=0)
    
    phi_x_next = np.append(phi[:,1%N_lat:N_lat],phi[:,0:1%N_lat],axis=1)
    
    #Compute and return the temporal average of kinetic energy.
    
    return np.average(kin(phi,phi_tau_next,phi_x_next))

#Temporal average of potential energy.

def pot_temp_av(x):
    
    #x is a list which contains three elements.
    #The first is the values of the points of the path.
    
    phi = x[0]
    
    #The second one is the mass of the scalar theory.
    
    m = x[1]
    
    #The third one is the coupling constant.
    
    lambd = x[2]
    
    #Compute and return the temporal average of potential energy.
    
    return np.average(pot(phi,m,lambd))

#Temporal average of total energy.

def total_eng_temp_av(x):
    
    #x is a list which contains four elements.
    #The first is the values of the points of the path.
    
    phi = x[0]
    
    #The second one is the mass of the scalar theory.
    
    m = x[1]
    
    #The third one is the coupling constant.
    
    lambd = x[2]
    
    #The second one is the number of points of the lattice
    
    N_lat = x[3]
    
    #Create two arrays with the values of phi displaced one row up or one column up.
    
    phi_tau_next = np.append(phi[1%N_lat:N_lat,:],phi[0:1%N_lat,:],axis=0)
    
    phi_x_next = np.append(phi[:,1%N_lat:N_lat],phi[:,0:1%N_lat],axis=1)
    
    return np.average(kin(phi,phi_tau_next,phi_x_next) + pot(phi,m,lambd))

#This function computes the bimodality of the distribution of fields.
    
def bimodality(data_phi):
    
    #Create a histogram with data_phi that has 100 bins.
    
    hist,hist_edges = np.histogram(data_phi,bins=100)
    
    #Create a boolean that is used to identify if the histogram reaches <phi>=0.
    
    boolean = True
    
    #Check if there are negative and positive values of <phi> in the histogram.
    
    if((hist_edges < 0).any() and (hist_edges > 0).any()):
    
        #In this case, there are negative and positive values of <phi> in the histogram.
        #This means that <phi>=0 is in the histogram
        #Use a for to go through the values <phi> and look for the place where the values flip sign.
        #This point will correspond to <phi>=0.
        
        for n_edge in range(len(hist_edges)):
        
            if((hist_edges[n_edge] > 0) and boolean):
        
                #In this case, the point where the values of <phi> flip sign has been found. 
                #n_edge-1 is the index for the bin corresponding to <phi>=0. 
                #Compute the bimodality with its definition.
                
                bimodality = 1 - hist[n_edge - 1]/max(hist)
        
                #Indicate that <phi>=0 is in the histogram setting boolean to False.
        
                boolean = False
            
    #Check if <phi>=0 is in the histogram with boolean.
            
    if(boolean):
        
        #In this case, boolean is True which means that <phi>=0 is no in the histogram.
        #This means that at <phi>=0 the distribution is 0 and bimodality is 1.
        
        bimodality = 1 
    
    #Return bimodality
    
    return bimodality


######################################################################
#################### Parameters in Command Line ######################
######################################################################

#Description of the program

parser = argparse.ArgumentParser(description='The program perfoms simulation of the 4-scalar theory using Hybrid Monte Carlo with Fourier acceleration.')

#The program requires a .json file with a dictionary will all the parameters are need to run the program.
#The use of files as input let the user save the parameters used in each run.

parser.add_argument('conf_file', metavar='conf_file', type=str, nargs=1, help='Directory of the file with the input values.')

args = parser.parse_args()

######################################################################
################## Loading File with Parameters ######################
######################################################################

#Open the file with the paramters.

with open(args.conf_file[0],'r') as file:
    
    print('-'*110 + '\n' + '-'*40 + ' Loading File with Paramters ' + '-'*41 + '\n' + '-'*110 +'\n')

    #The paramters are saved in a dictionary form.
    #The function eval convert the text in the file in an actual dictionary.
    #The dictionary is saved in parameters_dict.
    
    parameters_dict = eval(file.read())
    
print('Loaded parameters: ' + str(parameters_dict))
    
######################################################################
###################### Metropolis Parameters #########################
######################################################################

#Number of sweeps that will be performed.

N_sweep = parameters_dict['N_sweep']

######################################################################
####################### Leapfrog Parameters ##########################
######################################################################

#This is the number of leapfrog steps will apply in each Metropolis sweep.

N_max_leap = parameters_dict['N_max_leap']

#Discretitzation of the leapfrog method.

epsilon_leap = parameters_dict['epsilon_leap'] 

#Dimensionless mass parameter added to the fictious kinetic energy.

M = parameters_dict['M'] 

######################################################################
###################### Observables Parameters ########################
######################################################################

#Number of initial sweeps discarded before taking any measurement.

N_int_dis = parameters_dict['N_int_dis']

#Number of elements in one bins used in the Jacknife method to compute the variance of the observables.
#The number has to be greater than the correlation time but smaller than N_O.
#Idially n_bins should divide N_O. 
#If it is not the case, the first measurements are removed to ensures that n_binds divides N_0 minus the discarded measurements.

N_bins = parameters_dict['N_bins'] 

#Numebr of paths between measurements.

N_sep = parameters_dict['N_sep']

#Create a variable which is used to decide when to measure according to N_sep.

n_sep = 0

#Compute the number of measurements that will be taken considering N_sweep, N_int_dis and N_sep.
#Check that the number of measurements is a integer, otherwise unnecessary sweeps will be performed.

if((N_sweep - N_int_dis)%N_sep == 0):

    #In this case the number of measurements is an integer.
    
    N_O = (N_sweep - N_int_dis)//N_sep

else:
    
    #In this case the number of measurements is not an integer.
    
    print('N_sep does not divide N_sweep - N_int_dis and unncessary sweeps will be performed.\n')
    print('Please change the inputs')
    
    exit()
    
######################################################################
######################## Lattice Parameters ##########################
######################################################################

#Number of points in the time and position lattice.

N_lat = parameters_dict['N_lat'] 

#Step-size for time and position lattice (units h=c=1)

delta_lat = parameters_dict['delta_lat']

#Points in the time or position lattice.

lat_t_x = np.arange(start=0, stop=N_lat*delta_lat, step=delta_lat)

#Check if the user wants a hot or a cold start.

if(parameters_dict['hot_start']):
    
    #Use the value of phi_max to set a bound for the random selection of the initial hot configuration.
    #The selection use a uniform distribution in range [phi_max,phi_max)
    
    phi_max = parameters_dict['phi_max']
    
    #In this case, a hot start for the initial configuration of the fields of the lattice is used.

    lat_phi = phi_max*(np.random.uniform(0,1,(N_lat,N_lat))-0.5)
    
else:
    
    #In this case, a cold start for the initial configuration of the fields of the lattice is used.

    lat_phi = np.zeros((N_lat,N_lat))

#Create an array where the paths used for measurements will be saved.

path_list = np.zeros((N_lat,N_lat,N_O))

######################################################################
####################### Sacalar Field Parameters #####################
######################################################################

#Dimensionless square bare mass of the scalar field.

m_sq = parameters_dict['m_sq']

#Dimensionless coupling constant of the scalar field.

lambd = parameters_dict['lambd']

######################################################################
############################ Observables #############################
######################################################################

#_____________________________________________________________________
#                             Warning                                                  
# The code for the observables is such that only changing the parameters in this chapter is sufficient to specify which observables have to be measured. However, it is possible that in some complicated cases, the performance is affected so that the code should be changed in the lines where the measurements take place. What can be more problematic regarding efficiency is the use of the function eval to convert the the string with the name of the variables into useful code, but it should not affect too much the performance.                             
#_____________________________________________________________________

#Create a dictionary where each item is an observable that is wanted to measure.
#The key of each item is the one used in the files to identify the observable.
#To make the loading of the fields easier, the names should not have spaces.
#The value of the item is the name of the function used to compute the observable.
#The functions are specified below.

dict_O_func = parameters_dict['dict_O_func']

#Create a dictionary where the lists of the variables used in the functions in dict_O_func are saved.
#The key of each item is the same as in the previous dictinary.
#The value of the item is a string with the name of the variables like they appear in the code.
#Each variable has to be separated in the string by a comma from the rest.
#The function eval is used to convert the string into code.
#If more than one variable is used, then the function transform the different variables separated by a comma into a tuple.
#Each element of the tuple is one of the variables.
#This imples that the functions have to be defined as functions of a tuple where each element is one of the variables that is wanted to be used.
#If only one variable is present, no comma is needed and the eval function transform the variable in just code without tuple.
#Hence, for functions with one variable no tuple notation is needed.

dict_O_var = parameters_dict['dict_O_var']

#Create a dictionary where booleans that indicate if plots has to be made are saved.
#The key of each item is the same as in the previous dictinary.
#The value of the item is a boolean that if it is true, the plot for the observable must be done. Otherwise, no.

dict_O_plot_boolean = parameters_dict['dict_O_plot_boolean']

#Create a dictionary where strings that indicate if the observable depends on time, position or just the measurement.
#The key of each item is the same as in the previous dictinary.
#The value of the item is a string that can be 'Time', 'Position' or 'None'.
#If it is 'Time', then the observable only depends on time, because it is probably averaged over the position or only one is used.
#If it is 'Position', it is the same as before but interchanging time and position.
#If it is 'None', then the observable does not depend on time or position and as the previous cases the most probable situation is that and average over time and position is used.
#Other string will be invalid.

dict_O_dependance = parameters_dict['dict_O_dependance']

#Create a dictionary with the name used in the plots to identify the observables.
#The key of each item is the same as in the previous dictinary.
#The value of the item is string with the name that will be used in y-axis and the title of the plots of the observables.
#If a text is put between two $, one can use latex notation.

dict_O_plot_names = parameters_dict['dict_O_plot_names']

#Create a dictionary where the lists of measurements are saved.
#The key of each item is the same as in the previous dictinary.
#The value of the item is an empty list where the measurements will be stored.

dict_O_list = {n_O_items: [] for n_O_items in dict_O_var.keys()}

######################################################################
######################## Saving Parameters ###########################
######################################################################

#Path where the files with data will be saved (there has to be a "/" at the end of the path)

file_path = parameters_dict['file_path']

#Check if file_path exists.

if(not os.path.isdir(file_path)):
    
    print('The path "' + file_path + '" does not exist.')
    
    #In this case, the path does not exist.
    #The path is created.
    
    os.makedirs(file_path)
    
    print('Path created')
            
else:
    
    print('The path "' + file_path + '" exists.')
    
#The name of the files where results are saved have the same structure.
#The name contains a base which indicates what the file contains, e.g. paths, observables measurements...
#The second part of the name is common in all the files, so the string file_name is created with this part to simplify the code.
    
file_name = ''

######################################################################
############################## Action ################################
######################################################################

#Dimensionless action.

def S(phi,N_lat,m_sq,lambd):
    
    #Dimensionless Lagrangian
    
    def L(phi,phi_tau_next,phi_x_i_next,m,lambd):
        
        return 1/2*(phi_tau_next-phi)**2 + 1/2*(phi_x_i_next-phi)**2 + 1/2*m_sq*phi**2 + lambd/4*phi**4
    
    #Create two arrays with the values of phi displaced one row up or one column up.
    
    phi_tau_next = np.append(phi[1%N_lat:N_lat,:],phi[0:1%N_lat,:],axis=0)
    
    phi_x_i_next = np.append(phi[:,1%N_lat:N_lat],phi[:,0:1%N_lat],axis=1)
    
    #Compute the action
    
    action = np.sum(L(phi,phi_tau_next,phi_x_i_next,m_sq,lambd))
    
    #Return the action.
    
    return action

#Derivative of dimensionless action.

def S_der(phi,N_lat,m_sq,lambd):
    
    #Derivative of dimensionless Lagrangian.
    
    def L_der(phi,phi_tau_pre,phi_tau_next,phi_x_i_pre,phi_x_i_next,m_sq,lambd):
        
        return 4*phi - phi_tau_next - phi_tau_pre - phi_x_i_next - phi_x_i_pre + m_sq*phi + lambd*phi**3
           
    #Create four arrays with the values of phi displaced one row up or down or with one column up or down.
    
    phi_tau_next = np.append(phi[1%N_lat:N_lat,:],phi[0:1%N_lat,:],axis=0)
    phi_tau_pre = np.append(phi[-1%N_lat:N_lat,:],phi[0:-1%N_lat,:],axis=0)
    
    phi_x_i_next = np.append(phi[:,1%N_lat:N_lat],phi[:,0:1%N_lat],axis=1)
    phi_x_i_pre = np.append(phi[:,-1%N_lat:N_lat],phi[:,0:-1%N_lat],axis=1)

    #Compute and return the derivative of action.
    
    return L_der(phi,phi_tau_pre,phi_tau_next,phi_x_i_pre,phi_x_i_next,m_sq,lambd)

#The function computes all the terms of the Lagrangian.
    
def Lang(phi,N_lat,m_sq,lambd):

    #Dimensionless Lagrangian
    
    def L(phi,phi_tau_next,phi_x_i_next,m,lambd):
        
        return 1/2*(phi_tau_next-phi)**2 + 1/2*(phi_x_i_next-phi)**2 + 1/2*m_sq*phi**2 + lambd/4*phi**4
    
    #Create two arrays with the values of phi displaced one row up or one column up.
    
    phi_tau_next = np.append(phi[1%N_lat:N_lat,:],phi[0:1%N_lat,:],axis=0)
    
    phi_x_i_next = np.append(phi[:,1%N_lat:N_lat],phi[:,0:1%N_lat],axis=1)
    
    #Compute the action
    
    return L(phi,phi_tau_next,phi_x_i_next,m_sq,lambd)
    
######################################################################
########################## Kinetic energy ############################
######################################################################

#This function defines the kinetic energy for the momenta added in HMC.
    
def kin_pi(mom):
    
    return np.sum(mom**2/2)

#This function defines the kinetic energy un momentum space for the momenta added in HMC.
    
def kin_pi_mom(mom,kern_inv,N_lat):
    
    #The kinetic energy is just sum of the element-wise product between the |mom|**2 and kern_inv.
    #Use multiply function to perform the product and sum to perform the sum.
    
    return 1/(2*N_lat**2)*np.sum(np.multiply(np.abs(mom)**2,kern_inv))

#This function defiens the derivative in momentum space.

def kin_pi_der_mom(mom,kern_inv):
    
    #The derivative is just the element-wise product between the mom and kern_inv.
    #Use multiply function to perform the product.
    
    return np.multiply(mom,kern_inv)

#This function defines the inverse of the kernel of the kinetic energy in momentum space.
    
def kern_inv(N_lat,M):

    #Create a lattice with the dimensionless values of the momenta for each component.
    #Due to periodic conditions the momenta is descritized too.
    
    k = np.linspace(0,2*np.pi,N_lat,endpoint=False)

    #Create an empty N_latxN_lat array where the program saves the values of the inverse of the kernel of the fictious kinetic energy.
    
    A = np.zeros((N_lat,N_lat))
    
    #Use two for to go through the values of k.
    
    for n_A in range(N_lat):
        
        #For the second for it is used that A is symmetric so only one triangle of the matric has to be computed.
        
        for m_A in range(n_A,N_lat):

            #Compute and save the (n_A,m_A)-th value of the inverse of kernel taking into accound the symmetry of A.
            
            k_hat_sq = 4*(np.sin(k[n_A]/2)**2 + np.sin(k[m_A]/2)**2)
            
            A[n_A,m_A] = 1/(k_hat_sq + M**2)
            
            A[m_A,n_A] = 1/(k_hat_sq + M**2)
      
    #Return the values of the inverse of the kernel.
    
    return A
        
######################################################################
#################### Random Generation of Momenta ####################
######################################################################
    
#This function generates the fictious random momenta used in the leapfrog method in the momentum space.
#Their distribution is given by the negative exponential of the kinetic energy.
#In the generation it is imposed that the momenta are real in position space.

def pi_gen(N_lat,PI_std):
    
    #Compute half N_lat.
    
    N_lat_2 = int(N_lat/2)
    
    #First, a random N_latxN_lat matrix is generated.
    #This matrix, called PI, contains the real and imaginary components of the values of pi in momentum space.
    #The distribution of each component of PI follows a normal distribution with mean 0 and standard desviation PI_std.
    #Use the normal function of numpy to generate the random values of PI.
    
    PI = np.random.normal(0,PI_std)
    
    #Create an empty N_latxN_lat matrix where the values of pi in momentum space are saved.
    
    pi = np.zeros((N_lat,N_lat),dtype=complex)
    
    #From the previous matrix, the values of pi in momentum space are obtained, since PI contains the real and imaginary parts.
    #The values of pi are constructed so that in position space they give real values.
    #It is assumed that N_lat is even.
    
    #First, the values of pi that are real are saved.
    
    pi[0,0] = PI[0,0]
    pi[0,N_lat_2] = PI[0,N_lat_2]
    pi[N_lat_2,0] = PI[N_lat_2,0]
    pi[N_lat_2,N_lat_2] = PI[N_lat_2,N_lat_2]
    
    #Secondly, the values of pi that has only one index equal to 0 or N_lat_2.
    
    pi[0,1:N_lat_2] = (PI[0,1:N_lat_2] + 1j*PI[0,N_lat_2+1:N_lat][::-1])/np.sqrt(2)
    pi[0,N_lat_2+1:N_lat] = np.conj(pi[0,1:N_lat_2][::-1])
    
    pi[N_lat_2,1:N_lat_2] = (PI[N_lat_2,1:N_lat_2] + 1j*PI[N_lat_2,N_lat_2+1:N_lat][::-1])/np.sqrt(2)
    pi[N_lat_2,N_lat_2+1:N_lat] = np.conj(pi[N_lat_2,1:N_lat_2][::-1])
    
    pi[1:N_lat_2,0] = (PI[1:N_lat_2,0] + 1j*PI[N_lat_2+1:N_lat,0][::-1])/np.sqrt(2)
    pi[N_lat_2+1:N_lat,0] = np.conj(pi[1:N_lat_2,0][::-1])
    
    pi[1:N_lat_2,N_lat_2] = (PI[1:N_lat_2,N_lat_2] + 1j*PI[N_lat_2+1:N_lat,N_lat_2][::-1])/np.sqrt(2)
    pi[N_lat_2+1:N_lat,N_lat_2] = np.conj(pi[1:N_lat_2,N_lat_2][::-1])
    
    #Finally, the values of pi whose indices are different from 0 and N_lat_2.
    
    
    pi[1:N_lat_2,1:N_lat_2] = (PI[1:N_lat_2,1:N_lat_2] + 1j*(PI[N_lat_2+1:N_lat,N_lat_2+1:N_lat][::-1,::-1].T).T)/np.sqrt(2)
    pi[N_lat_2+1:N_lat,N_lat_2+1:N_lat] = np.conj((pi[1:N_lat_2,1:N_lat_2][::-1,::-1].T).T)
   
    pi[1:N_lat_2,N_lat_2+1:N_lat] = (PI[1:N_lat_2,N_lat_2+1:N_lat] + 1j*(PI[N_lat_2+1:N_lat,1:N_lat_2][::-1,::-1].T).T)/np.sqrt(2)
    pi[N_lat_2+1:N_lat,1:N_lat_2] = np.conj((pi[1:N_lat_2,N_lat_2+1:N_lat][::-1,::-1].T).T)
    
    #Return the values of pi.
    
    return pi

######################################################################
############################## Leapfrog ##############################
######################################################################

#This function defines one leapfrog step with Fourier acceleration.

def leapfrog_fourier(phi,pi,epsilon,N_lat,m_sq,lambd,kern_inv_val):
    
    #This function is like the leapfrog one, the only difference is the calculation of phi_step in Fourier space.
    
    #Whole half-step for pi and Fourier transform.
    
    pi_half = pi - epsilon*S_der(phi,N_lat,m_sq,lambd)
    pi_half_mom = np.fft.fft2(pi_half)

    #Whole step for phi.
    #The derivative is computed in momentum space and transformed using the inverse Fourier transformation.
    
    phi_step = phi + epsilon*np.real(np.fft.ifft2(kin_pi_der_mom(pi_half_mom,kern_inv_val)))

    #Return phi_step and pi_half which are in position space.

    return phi_step,pi_half

######################################################################
############################# Metropolis #############################
######################################################################

#The function computes one Metropolis sweep with N_lat_t updates of points of the path.

def Metropolis(phi,N_lat,m_sq,lambd,N_max_leap,epsilon_leap,kern_inv_val,PI_std,acc_rate):
    
    #Generate a random number.
    #This will be employed to decide which updates will be accepted.
    #The number is generated using a uniform distribution from 0 to 1.
    
    rnd_acc = np.random.uniform(0,1)
    
    #Generate a set of random numbers that will correspond to the momentum used in leapfrog method.
    #They are generated in momentum space with the function pi_gen.
    #Apply a inverse Fourier transform to get the values in position space.
    #pi is expected to be real so only the real part of the inverse Fourier transform is saved.
    
    pi_mom = pi_gen(N_lat,PI_std)
    pi = np.real(np.fft.ifft2(pi_mom))
    
    #Apply a the first leapfrog step.
    #Only one half-step of the momentum is performed.
    #Then in the following steps a whole step of momentum is done.
    #In this way, the first and final steps of the leapfrog method are combined and the derivative of the action necessary is computed only ones.
    #The step for the field is performed in momentum space (Fourier acceleration), so the values of pi and phi are Fourier transformed.
    #At the end, a inverse Fourier transform is applied to obtain the values of the field in position space.
    #As in the case of pi, only the real part is saved since the field is expected to be real
    
    pi_new = pi - epsilon_leap/2*S_der(phi,N_lat,m_sq,lambd)
    pi_new_mom = np.fft.fft2(pi_new)
    
    phi_new = phi + epsilon_leap*np.real(np.fft.ifft2(kin_pi_der_mom(pi_new_mom,kern_inv_val)))
    
    #pi_no_four = pi - epsilon_leap/2*S_der(phi,N_lat,m,lambd)
    #phi_no_four = phi + epsilon_leap*pi_no_four
    
    #Use a for to perform the N_max_leap iteration of the leapfrog method.
    #The for starts at 2 because the 0 case is the initial configurations and the case 1 is the step we have already done.

    for i in range(2,N_max_leap):
        
        #Perform one iteration of the leapfrog method
        
        phi_new,pi_new = leapfrog_fourier(phi_new,pi_new,epsilon_leap,N_lat,m_sq,lambd,kern_inv_val)
        
        
    #Perfom a final half-step, so that the final momentum is at the same time as the field.
    
    pi_new = pi_new - epsilon_leap/2*S_der(phi_new,N_lat,m_sq,lambd)
    pi_new_mom = np.fft.fft2(pi_new)
    
    #Compute the Hamiltonian used in the Leapfrog method for the new and old configuration.
    
    H_old = kin_pi_mom(pi_mom,kern_inv_val,N_lat) + S(phi,N_lat,m_sq,lambd)
    
    H_new = kin_pi_mom(pi_new_mom,kern_inv_val,N_lat) + S(phi_new,N_lat,m_sq,lambd)

    #Accept the update if the negative exponential of the variation of hamiltonian is greater than the random number rnd_acc[i].
    
    if(rnd_acc < np.exp(-H_new + H_old)):
        
        #In this case, the update is accepted.
        
        phi = phi_new

        #For each accepted update 1 is added to acc_rate.
        #When the simulation finishes, this will give a rough estimation of the acceptance rate for this algorithm with the given input parameters
        
        acc_rate = acc_rate + 1
    
    #Return the path after one sweep and acc_rate.
    
    return phi,acc_rate

######################################################################
######################## Jackknife method ############################
######################################################################
    
#This function computes the average and its statistical error for an observable using the jackknife methods.
    
def jacknife(O,N_bins):

    print('-'*110 + '\n' + '-'*46 + ' Jacknife  Method ' + '-'*46 + '\n' + '-'*110 +'\n')
    
    #Number of measurements.
    
    N_meas = len(O)
    
    #Check if N_bins divides the number of measurements.
    
    if(N_meas%N_bins != 0):
        
        #In this case N_bins does not divide the number of measurements.
        #Remove the first (N_meas modulo N_bins) terms from O.
        
        O = O[N_meas%N_bins:]
        
        #Re-compute the number of measurements after removing the first ones.
        
        N_meas = len(O)
    
    #The mean is just the arithmetic one which is computed with the average function in numpy.
    
    mean_O = np.average(O)
    
    #Create a variable where the Jacknife variance will be saved.
    
    variance = 0
    
    #To compute the variance using the Jacknife method the measurements are divided into N_meas//N_bins blocks.
    #Each block has N_bins measurments.
    #In the i-th iteration of the for the arithmetic mean of the observable is computed without the measurments of the i-th block.
    #In addition, on each iteration the previous mean is used to compute the Jacknife variance.
    
    for i in range(N_meas//N_bins):
        
        #Remove the i-th block from the measurments.
        
        O_jack = np.delete(O,np.arange(i*N_bins,(i+1)*N_bins))

        #Compute the mean with the measurments in O_jack.
        
        O_jack_mean = np.average(O_jack)
        
        #Add the O_jack_mean into the Jacknife variance with the proper factor accordind the the Jacknife method.

        variance = variance +  (N_meas//N_bins -  1)/(N_meas - N_bins)*(O_jack_mean - mean_O)**2

    #The statistical error of the mean is the square root of the variance.
    
    error_O = variance**(1/2)

    #Return the mean and the error.
    
    return mean_O, error_O

######################################################################
########### Running Metroplis and Taking Measurements ################
######################################################################

#Compute the values of the inverse of the kernel.
    
kern_inv_val = kern_inv(N_lat,M)

#From the values of the inverse of the kernel, compute the standard desviations for the fictious momenta in momentum space.

PI_std = N_lat*np.sqrt(1/kern_inv_val)

#Create variables where the average time elapsed for one sweep and to measure observables are saved.
    
time_sweep_av = 0
time_meas_av = 0

#Measure the time in seconds at the begining of the simulation.

time_simulation = time.time()

#Create a variable where the acceptance rate of the metropolis algorithm is computed.
    
acc_rate = 0

#Each iteration of the for is a Metropolis sweep.
    
for n_sweep in tqdm(range(N_sweep)):
    
    #The time at the begining of the sweep in seconds.
    
    time_sweep_initial = time.time()
    
    #print('-'*110 + '\n' + '-'*45 + ' Metropolis Sweep ' + str(n_sweep) + ' ' + '-'*45 + '\n' + '-'*110 +'\n')
     
    #Perform a sweep with Metropolis.
    #The function returns the path after the sweep and the acc_rate where a 1 has been added if the update is accepted.
    
    lat_phi,acc_rate = Metropolis(lat_phi,N_lat,m_sq,lambd,N_max_leap,epsilon_leap,kern_inv_val,PI_std,acc_rate) 

    #The time elapsed in the sweep.
    
    time_sweep_el = time.time() - time_sweep_initial
    
    #print('Time elapsed in sweep: ' + str(time_sweep_el))
    
    #The time is added to time_sweep_av.
    
    time_sweep_av += time_sweep_el
 
    #Only take measurement when the n_sweep is greater than N_int_dis.
    #The measurements are taken after reaching the equilibrium only.
    
    if(n_sweep == N_int_dis + n_sep*N_sep):
        
        #################################################################
        ####################### Measurements ############################
        #################################################################
            
        #The time at the begining of the measurements in seconds.
    
        time_meas_initial = time.time()
        
        #Compute the observables that are in dict_O_func.
        #The variables for the functions are in dict_O_var saved as a string which is translated into code with the eval function.
        #The results are saved in the corresponding list of dict_O_list.
        #Use a for to go through all the items of the dictionaries of the observables.
        
        for n_O_items in dict_O_func:
        
            dict_O_list[n_O_items].append(dict_O_func[n_O_items](eval(dict_O_var[n_O_items])))
        
        #Save the path in n_sep-th column of path_list.
        
        path_list[:,:,n_sep] = lat_phi
  
        #Add 1 to n_sep.
        
        n_sep += 1
    
        #The time elapsed in the measurement.
    
        time_meas_el = time.time() - time_meas_initial
        
        #print('Time elapsed in measurement: ' + str(time_meas_el))
        
        #The time is added to time_sweep_av.
        
        time_meas_av += time_meas_el
  
#Average the acceptance rate dividing by the number of sweeps that is also the number of proposed updates.
    
acc_rate = acc_rate/N_sweep
    
#Average times.
        
time_sweep_av = time_sweep_av/N_sweep
time_meas_av = time_meas_av/(N_O)
    
#Measure the time in seconds before plotting the observables.
        
time_plot_O = time.time()
    
######################################################################
############## Plots of measurements for each path ###################
######################################################################

print('-'*110 + '\n' + '-'*44 + ' Plotting Observables ' + '-'*44 + '\n' + '-'*110 +'\n')

#For each observable, the boolean in dict_O_plot_boolean is checked to decide which observables has to be ploted.
#The string in dict_O_dependance is also checked since only the cases 'None' are plotted, the other requires averages over the different paths and gamma method or jacknife should be taken into account.
#Create a integer variable used to ennumarate the figure.

n_fig = 1

for n_O_items in dict_O_list:
    
    if(dict_O_plot_boolean[n_O_items] & (dict_O_dependance[n_O_items]=='None')):
        
        #In this case, the boolean is True and the string is 'None' so the data are ploted.
        #Function global is used to create a variable with the name fig_ plus n_fig.
        #A figure is created and saved in the previous variable.
        
        globals()['fig_' + str(n_fig)] = plt.figure()
        
        #Add a subplot to the figure using add_subplot.
        #The name of the variable for the subplot is also created using globals and contains ax instead of fig.
        
        globals()['ax_' + str(n_fig)] =  eval('fig_' + str(n_fig)).add_subplot(1, 1, 1)
    
        #Plot the observable.
        
        eval('ax_' + str(n_fig)).scatter(np.arange(0,N_O,1),dict_O_list[n_O_items])

        #Display parameters.

        eval('ax_' + str(n_fig)).set_title('Measurments of observable ' + dict_O_plot_names[n_O_items] ,fontsize=25)        
        eval('ax_' + str(n_fig)).set_ylabel(dict_O_plot_names[n_O_items],fontsize=25)
        eval('ax_' + str(n_fig)).set_xlabel('Measurement',fontsize=25)
        eval('ax_' + str(n_fig)).tick_params(axis='both', labelsize=25)
        eval('ax_' + str(n_fig)).grid(True,which='major',axis='both',alpha=0.5) 
        eval('ax_' + str(n_fig)).grid(True,which='minor',axis='both',alpha=0.25)    
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.pause(5)
        eval('fig_' + str(n_fig)).tight_layout()
        
        #Save the figure.
        
        eval('fig_' + str(n_fig)).savefig(file_path + 'Observable_measurements_paths_' + n_O_items + file_name + '.png',dpi=500)

        #Add one to n_fig.
        
        n_fig += 1

#Measure the time elapsed after plotting the observables.
        
time_plot_O = time.time() - time_plot_O

#Measure the time in seconds before saving data
        
time_save_data = time.time()

######################################################################
################ Averaging and plotting measurements #################
######################################################################
    
#Compute the mean and statistical error of the mean using the Jacknife method for each observable.
#Use a for to go through the items of dict_O_list where the data is saved.
#The mean and error are saved in txt file.

print('-'*110 + '\n' + '-'*29 + ' Saving Averages and Stadistical Errors of Observables ' + '-'*29 + '\n' + '-'*110 +'\n')

for n_O_items in dict_O_list:
    
    #Check if dict_O_dependance[n_O_items] is 'Time', 'Position' or 'None'.
    
    if(dict_O_dependance[n_O_items] == 'Time'):
        
        #In this case, dict_O_dependance[n_O_items] is 'Time'.
        #For each point of the time lattice, an average is computed over all the paths.
        #The averages and errors are saved in a array of size 2xN_lat_t.
        
        mean_err_n_O_items = np.zeros((2,N_lat))
        
        #Create an array with the data of dict_O_list[n_O_items].
        
        dict_O_list_n_O_items_arr = np.array(dict_O_list[n_O_items])
        
        #Use a for to go through the points of the lattice
        
        for n_O_jacknife_time in range(N_lat):
            
            #Compute the mean and erro using the Jacknife method for the n_O_jacknife_time-th time point of the lattice.
            
            mean_err_n_O_items[0,n_O_jacknife_time],mean_err_n_O_items[1,n_O_jacknife_time] = jacknife(dict_O_list_n_O_items_arr[:,n_O_jacknife_time],N_bins)
        
        print('Observable ' + n_O_items)
        print('Mean: ' + str(mean_err_n_O_items[0]))
        print('Statistical error: ' + str(mean_err_n_O_items[1]))
    
        #Save data in a file.
        
        np.savetxt(file_path + 'Observables_jacknife_mean_and_error_' + n_O_items + file_name + '.txt',mean_err_n_O_items)

        #For each observable, the boolean in dict_O_plot_boolean is checked to decide which observables has to be ploted.
            
        if(dict_O_plot_boolean[n_O_items]):
            
            #In this case, the boolean is True so the data are ploted.
            #Function global is used to create a variable with the name fig_ plus n_fig.
            #A figure is created and saved in the previous variable.
            
            globals()['fig_' + str(n_fig)] = plt.figure()
            
            #Add a subplot to the figure using add_subplot.
            #The name of the variable for the subplot is also created using globals and contains ax instead of fig.
            
            globals()['ax_' + str(n_fig)] =  eval('fig_' + str(n_fig)).add_subplot(1, 1, 1)
        
            #Plot the observable.
            #In this case the dict_O_dependance is 'Time', so the average over each time is plotted vs the time lattice.
            
            eval('ax_' + str(n_fig)).errorbar(lat_t_x,mean_err_n_O_items[0],yerr=mean_err_n_O_items[1],linestyle='none',marker='o',c='b')
    
            #Display parameters.
    
            eval('ax_' + str(n_fig)).set_title('Measurments of observable ' + dict_O_plot_names[n_O_items] ,fontsize=25)        
            eval('ax_' + str(n_fig)).set_ylabel(dict_O_plot_names[n_O_items],fontsize=25)
            eval('ax_' + str(n_fig)).set_xlabel('Measurement',fontsize=25)
            eval('ax_' + str(n_fig)).tick_params(axis='both', labelsize=25)
            eval('ax_' + str(n_fig)).grid(True,which='major',axis='both',alpha=0.5) 
            eval('ax_' + str(n_fig)).grid(True,which='minor',axis='both',alpha=0.25)    
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            plt.pause(1)
            eval('fig_' + str(n_fig)).tight_layout()
            
            #Save the figure.
            
            eval('fig_' + str(n_fig)).savefig(file_path + 'Observable_measurements_paths_' + n_O_items + file_name + '.png',dpi=500)
    
            #Add one to n_fig.
            
            n_fig += 1

    elif(dict_O_dependance[n_O_items] == 'Position'):
        
        #In this case, dict_O_dependance[n_O_items] is 'Position'.
        #For each point of the position lattice, an average is computed over all the paths.
        #The averages and errors are saved in a array of size 2xN_lat_x.
        
        mean_err_n_O_items = np.zeros((2,N_lat))
        
        #Create an array with the data of dict_O_list[n_O_items].
        
        dict_O_list_n_O_items_arr = np.array(dict_O_list[n_O_items])
        
        #Use a for to go through the points of the lattice
        
        for n_O_jacknife_time in range(N_lat):
            
            #Compute the mean and erro using the Jacknife method for the n_O_jacknife_time-th time point of the lattice.
            
            mean_err_n_O_items[0,n_O_jacknife_time],mean_err_n_O_items[1,n_O_jacknife_time] = jacknife(dict_O_list_n_O_items_arr[:,n_O_jacknife_time],N_bins)
        
        print('Observable ' + n_O_items)
        print('Mean: ' + str(mean_err_n_O_items[0]))
        print('Statistical error: ' + str(mean_err_n_O_items[1]))
    
        #Save data in a file.
        
        np.savetxt(file_path + 'Observables_jacknife_mean_and_error_' + n_O_items + file_name + '.txt',mean_err_n_O_items)

        #For each observable, the boolean in dict_O_plot_boolean is checked to decide which observables has to be ploted.
        
        if(dict_O_plot_boolean[n_O_items] ):
            
            #In this case, the boolean is True so the data are ploted.
            #Function global is used to create a variable with the name fig_ plus n_fig.
            #A figure is created and saved in the previous variable.
            
            globals()['fig_' + str(n_fig)] = plt.figure()
            
            #Add a subplot to the figure using add_subplot.
            #The name of the variable for the subplot is also created using globals and contains ax instead of fig.
            
            globals()['ax_' + str(n_fig)] =  eval('fig_' + str(n_fig)).add_subplot(1, 1, 1)
        
            #Plot the observable.
            #In this case, the dict_O_dependance is 'Position', so the average over each position is plotted vs the position lattice.
            
            eval('ax_' + str(n_fig)).errorbar(lat_t_x,mean_err_n_O_items[0],yerr=mean_err_n_O_items[1],linestyle='none',marker='o',c='b')
    
            #Display parameters.
    
            eval('ax_' + str(n_fig)).set_title('Measurments of observable ' + dict_O_plot_names[n_O_items] ,fontsize=25)        
            eval('ax_' + str(n_fig)).set_ylabel(dict_O_plot_names[n_O_items],fontsize=25)
            eval('ax_' + str(n_fig)).set_xlabel('Measurement',fontsize=25)
            eval('ax_' + str(n_fig)).tick_params(axis='both', labelsize=25)
            eval('ax_' + str(n_fig)).grid(True,which='major',axis='both',alpha=0.5) 
            eval('ax_' + str(n_fig)).grid(True,which='minor',axis='both',alpha=0.25)    
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            plt.pause(1)
            eval('fig_' + str(n_fig)).tight_layout()
            
            #Save the figure.
            
            eval('fig_' + str(n_fig)).savefig(file_path + 'Observable_measurements_paths_' + n_O_items + file_name + '.png',dpi=500)
    
            #Add one to n_fig.
            
            n_fig += 1

    elif(dict_O_dependance[n_O_items] == 'None'):
        
        #In this case, dict_O_dependance[n_O_items] is 'None'.
        #For each path there is only one number which is averaged over all of them.
        #The average and error are saved in a array of size 2x1.
        
        mean_err_n_O_items = np.zeros((2,1))
        
        #Compute the mean and error using the Jacknife method.
            
        mean_err_n_O_items[0,0],mean_err_n_O_items[1,0] = jacknife(dict_O_list[n_O_items],N_bins)
        
        print('Observable ' + n_O_items)
        print('Mean: ' + str(mean_err_n_O_items[0]))
        print('Statistical error: ' + str(mean_err_n_O_items[1]))
    
        #Save data in a file.
        
        np.savetxt(file_path + 'Observables_jacknife_mean_and_error_' + n_O_items + file_name + '.txt',mean_err_n_O_items)

    else:
         
        #In this case the string is not 'Time, 'Position' or 'None' and the entry is invalid.
        #Inform the user of the invalid input.
        
        print('Invalid input in dict_O_dependance for the observable ' + n_O_items + ': ' + dict_O_dependance[n_O_items])
         
######################################################################
################# Saving measurements for each path ##################
######################################################################
         
#Save the measurements of the observables for each path into different files, one for each observable.
#Use two for to go through each observable and each path.

print('-'*110 + '\n' + '-'*32 + ' Saving Measurements of Observables for each Path ' + '-'*31 + '\n' + '-'*110 +'\n')

for n_O_items in dict_O_list:

    #Check if dict_O_dependance[n_O_items] is 'Time', 'Position' or 'None' to save it properly.
    
    if(dict_O_dependance[n_O_items] == 'Time'):
        
        #In this case, it is 'Time', so the results of the observable is a list of points one for each time of the lattice.
    
        with open(file_path + 'Observable_measurements_paths_' + n_O_items + file_name + '.txt','w+') as file:     
           
            #In the first row we add the points of the time lattice.
            
            np.savetxt(file,np.array([lat_t_x]))
            
            for n_path in range(N_O):
                
                #Write the data, each path is a row.
                
                np.savetxt(file,np.array([dict_O_list[n_O_items][n_path]]))
        
    elif(dict_O_dependance[n_O_items] == 'Position'):
        
        #In this case, it is 'Position' which is similar to the previous case.
    
        with open(file_path + 'Observable_measurements_paths_' + n_O_items + file_name + '.txt','w+') as file:     
           
            #In the first row we add the points of the position lattice.
            
            np.savetxt(file,np.array([lat_t_x]))
            
            for n_path in range(N_O):
                
                #Write the data, each path is a row.
                
                np.savetxt(file,np.array([dict_O_list[n_O_items][n_path]]))
                
    elif(dict_O_dependance[n_O_items] == 'None'):
        
        #In this case, it is 'None', so for each path there is only one number.
    
        with open(file_path + 'Observable_measurements_paths_' + n_O_items + file_name + '.txt','w+') as file:     
           
            
            for n_path in range(N_O):
                
                #Write the data, each path is a row.
                
                file.write(str(dict_O_list[n_O_items][n_path]) + '\n')
       
    else:
        
        #In this case the string is not 'Time, 'Position' or 'None' and the entry is invalid.
        #Inform the user of the invalid input.
        
        print('Invalid input in dict_O_dependance for the observable ' + n_O_items + ': ' + dict_O_dependance[n_O_items])
         
######################################################################
########################## Saving Paths ##############################
######################################################################
        
#Save the paths to a file.

print('-'*110 + '\n' + '-'*48 + ' Saving Paths ' + '-'*48 + '\n' + '-'*110 +'\n')

with open(file_path + 'Paths' + file_name + '.txt','w+') as file:

    for n_path_save in tqdm(range(path_list.shape[2])):
    
        np.savetxt(file,path_list[:,:,n_path_save],header='#'*41 + ' Measurement '+ str(n_path_save + 1) + '#'*42,comments='')

######################################################################
################ Computing and Saving Bimodality #####################
######################################################################

#Check if <phi> is measured otherwise bimodality cannot be computed.

if 'phi' in dict_O_func:
    
    #In this case, <phi> is mesaured and bimodality can be computed.
    
    print('-'*110 + '\n' + '-'*38 + ' Computing and Saving Bimodality ' + '-'*39 + '\n' + '-'*110 +'\n')
    
    #Compute the bimodality.
    
    bimodality_val = bimodality(dict_O_list['phi'])
    
    print('Bimodality: ' + str(bimodality_val))
    
    #Save bimodality
    
    np.savetxt(file_path + 'Biomodality' + file_name + '.txt',[bimodality_val])

######################################################################
##################### Saving Acceptance Rate #########################
######################################################################

#Save acceptance rate.

print('-'*110 + '\n' + '-'*43 + ' Saving Acceptance Rate ' + '-'*43 + '\n' + '-'*110 +'\n')

print('Acceptance rate average: ' + str(acc_rate))

np.savetxt(file_path + 'Acceptance_rate' + file_name + '.txt',[acc_rate])

######################################################################
############# Computing and Saving Computational Time ################
######################################################################

#Measure the time elapsed after saving data.
        
time_save_data = time.time() - time_save_data

#Measure the time elapsed after completing the whole simulation.

time_simulation = time.time() - time_simulation

print('Data saved')

#Display time elapsed for the whole program and the different steps.

print('-'*110 + '\n' + '-'*48 + ' Elapsed Time ' + '-'*48 + '\n' + '-'*110 +'\n')

print('Whole simulation: ' + str(time_simulation))
print('Average Metropolis sweep: ' + str(time_sweep_av))
print('Average Measurments: ' + str(time_meas_av))
print('Plotting Observables: ' + str(time_plot_O))
print('Saving data: ' + str(time_save_data))

#Save the time elapsed information.

with open(file_path + 'time_elapsed' + file_name + '.txt','w+') as file:
    
    file.write('Whole simulation: ' + str(time_simulation) + '\n')
    file.write('Average Metropolis sweep: ' + str(time_sweep_av) + '\n')
    file.write('Average Measurments: ' + str(time_meas_av) + '\n')
    file.write('Plotting Observables: ' + str(time_plot_O) + '\n')
    file.write('Saving data: ' + str(time_save_data) + '\n')
