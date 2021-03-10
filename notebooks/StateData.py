import os
import numpy as np
import pandas as pd
from scipy.optimize import nnls

class StateData():
    
    def __init__(self,state_name = "California",filename = ""):
    
        self.population = None
        self.orig_beta_data = None
        self.CA_test_df = None
        self.S_0 = None
        self.column_names = None
        self.undetect_factor = None
        
        input_data_dir = os.getcwd()+"\..\datasets\\"
        output_data_dir = os.getcwd()+"\..\Generated Output Pickle Files\\"

        pop_Wyoming = 640386
        pop_Vermont = 637455
        pop_California = 39512223
        pop_California_simulated = 41159374
        
        if(state_name=='Wyoming'):
            population = pop_Wyoming
        elif(state_name=='Vermont'):
            population = pop_Vermont
        elif(state_name =='California'):
            population = pop_California
        elif(state_name =='California Simulated'):
            population = pop_California_simulated

        if(state_name =='California'):
            age_distribution = np.array([0.273,0.4844,0.1366,0.106])
        elif(state_name in ('Wyoming','Vermont','California Simulated')):
            age_distribution = np.array([0.059444636,0.060558887,0.063713953,0.064177457,0.067246426,\
                                         0.072009485,0.069644395,0.065303982,0.061312498,0.060585008,\
                                         0.062168104,0.065081866,0.062444041,0.053833488,0.043367819,\
                                         0.029388689,0.039719265]) 

        if(state_name =='California'):
            cal_df = pd.read_csv(input_data_dir+"California Agewise processed data.csv")

            orig_beta_data = cal_df.rename(columns={'Infected_0-17':'0-17','Infected_18-49':'18-49',\
                                            'Infected_50-64':'50-64','Infected_65+':'65+'})
            orig_beta_data = orig_beta_data.drop(['date'],axis=1)

            self.CA_test_df = pd.read_csv(input_data_dir + "California_statewide_testing.csv")

            CA_infected = orig_beta_data.sum(axis=1)
            susceptible = population - CA_infected

            CA_test_df = self.CA_test_df.rolling(window = 7).mean()
            CA_test_series = CA_test_df[-len(susceptible):].reset_index().drop(columns = ['index'])['tested']

            undetect_factor = susceptible/CA_test_series
            self.undetect_factor = pd.DataFrame(undetect_factor)

            ## The value for undetected factor is very high for initial days so we are replacing it with low value.
            initial_date = 65
            self.undetect_factor[0][:initial_date] = 13

        elif (state_name =='California Simulated'):
            orig_beta_data = pd.read_csv(input_data_dir+"Simulated California Data/"+"Cal-"+filename+".csv")
            orig_beta_data = orig_beta_data.drop(721,0)
            orig_beta_data = orig_beta_data.drop('t',1)
            orig_beta_data = orig_beta_data.rename(columns= lambda x: x[1:])
        else:
            orig_beta_data = pd.read_csv(input_data_dir+"Infection_no_int_"+state_name+".csv")
            orig_beta_data = orig_beta_data.drop(721,0)
            orig_beta_data = orig_beta_data.drop('t',1)
            orig_beta_data = orig_beta_data.rename(columns= lambda x: x[1:])
            
        self.population = population
        self.orig_beta_data = orig_beta_data
        self.column_names = orig_beta_data.columns.values
        self.S_0 = np.multiply(age_distribution,population).astype(int)
        

    def SIR_Data_with_T_R(self,orig_beta_data,T_R, S_0, population, smoothing = False, undetected_fact = False, window_length = 1):
        """
            Inputs: 
                1. orig_beta_data: daywise cummulative number of infected cases for different age groups
                2. T_R : Recovery time
                3. S_0 : Initial susceptible number of people for each age group
                4. population : Total population
                5. smoothing : boolean stating if smmothing is required
                6. window_length : window length for moving average smoothing
            outputs:
                1. S_data : daily agewise susceptible number of people 
                2. beta_data : daily agewise infected number of people 
                3. R_data : daily agewise recovered or deceased number of people 
        """
        if smoothing and not undetected_fact:
            first_diff = orig_beta_data.diff().fillna(0)
            smoothened_first_diff = first_diff.rolling(window_length,min_periods=1).mean()
            smoothened_first_diff.iloc[0] = orig_beta_data.iloc[0]
            orig_beta_data = smoothened_first_diff.cumsum()
            
        if smoothing and undetected_fact:
            first_diff = orig_beta_data.diff().fillna(0)
            first_diff.values[:,:] = first_diff.values*self.undetect_factor.values
            smoothened_first_diff = first_diff.rolling(window_length,min_periods=1).mean()
            smoothened_first_diff.iloc[0] = orig_beta_data.iloc[0]
            orig_beta_data = smoothened_first_diff.cumsum()

        R_data = pd.DataFrame().reindex_like(orig_beta_data).fillna(0)
        R_data = R_data.rename(columns = lambda x: 'R'+ x)

        R_data.values[T_R:,:] = orig_beta_data.values[:-T_R,:]

        R_data_tilde = R_data/S_0


        beta_data_tilde = pd.DataFrame().reindex_like(orig_beta_data).fillna(0)
        beta_data_tilde = beta_data_tilde.rename(columns = lambda x: 'I'+ x)

        old_beta_data_tilde = orig_beta_data/S_0

        beta_data_tilde.values[:,:] = old_beta_data_tilde.values - R_data_tilde.values


        S_data_tilde = pd.DataFrame().reindex_like(orig_beta_data).fillna(0)
        S_data_tilde = S_data_tilde.rename(columns = lambda x: 'S'+ x)

        S_data_tilde.values[:,:] = 1- beta_data_tilde.values - R_data_tilde.values 
        
        ###### Generating Fractional Data of S, beta and R

        S_data = S_data_tilde*S_0/population
        beta_data = beta_data_tilde*S_0/population
        R_data = R_data_tilde*S_0/population
        
        return S_data, beta_data, R_data

    def get_params_and_generated_data(self,S_data, beta_data, R_data, phases, T_R):
        """
            Inputs: 
                1. S_data : daily agewise susceptible number of people 
                2. beta_data : daily agewise infected number of people 
                3. R_data : daily agewise recovered or deceased number of people 
                4. Phases : list of individual phases
                5. T_R : Recovery time
            outputs:
                1. A_No_int_nn_list: list of A_matrix parameters for each phase
                2. Gamma_nn_list : list of Gamma parameters for each phase
                3. S_data_gen : daily agewise susceptible number of people generated by age-structured SIR model
                4. beta_data_gen : daily agewise infected number of people generated by age-structured SIR model
                5. R_data_gen : daily agewise recovered or deceased number of people generated by age-structured SIR model
        """
        A_No_int_nn_list = []
        Gamma_nn_list = []


        for phase in phases:

            S_data_phase = S_data.iloc[phase]
            beta_data_phase = beta_data.iloc[phase]
            R_data_phase = R_data.iloc[phase]

            no_age_groups = R_data.shape[1]
            if (phase[0]==0):
                start_date = T_R+1
            else:
                start_date = phase[0]

            end_date = phase[-1]
            
            print("start_date, end_date for a phase")
            print(start_date,end_date)

            ###### Generating C_master and D_master matrices ########

            no_params = no_age_groups*no_age_groups + no_age_groups
            no_diff_eqns = 3*no_age_groups

            q = no_age_groups

            for t in range(start_date,end_date):

                C_t = np.zeros((no_diff_eqns,no_params))
                D_t = np.zeros((no_diff_eqns,1))

                for i in range(no_diff_eqns):
                    if(i<q):
                        C_t[i][q*i:q*(i+1)] = -1*S_data.iloc[t][i]*np.array(beta_data.iloc[t])
                    elif(q<=i<2*q):
                        C_t[i][q*(i-q):q*(i-q+1)] = S_data.iloc[t][i-q]*np.array(beta_data.iloc[t])
                        C_t[i][q*q+i-q] = -1 * beta_data.iloc[t][i-q]
                    else:
                        C_t[i][q*q+i-2*q] =  beta_data.iloc[t][i-2*q]

                if(t==0):
                    D_t[0:q] = np.array(S_data.iloc[t]).reshape((q,1))

                    D_t[q:2*q] = np.array(beta_data.iloc[t]).reshape((q,1))

                    D_t[2*q:3*q] =np.array(R_data.iloc[t]).reshape((q,1))
                else:
                    D_t[0:q] = np.array(S_data.iloc[t]-S_data.iloc[t-1]).reshape((q,1))

                    D_t[q:2*q] = np.array(beta_data.iloc[t]-beta_data.iloc[t-1]).reshape((q,1))

                    D_t[2*q:3*q] = np.array(R_data.iloc[t]-R_data.iloc[t-1]).reshape((q,1))

                if(t==start_date):
                    C_master = C_t
                    D_master = D_t
                else:
                    C_master = np.vstack((C_master,C_t))
                    D_master = np.vstack((D_master,D_t))


            D_master = D_master.reshape((end_date-start_date)*no_diff_eqns)

            ######## Generating parameter values using non-negative least square solution

            solution = nnls(C_master, D_master)

            A_No_int_nn = solution[0][:q*q].reshape((q,q))
            Gamma_nn = solution[0][q*q:]
            
            print("Max, Min in A matrix")
            print(A_No_int_nn.max(),A_No_int_nn.min())
            #print("Gamma parameter values")
            #print(Gamma_nn)
            
            A_No_int_nn_list.append(A_No_int_nn)
            Gamma_nn_list.append(Gamma_nn)

            ## Data generation with the parameters and difference method

            S_data_gen = np.zeros(S_data_phase.shape)
            beta_data_gen = np.zeros(beta_data_phase.shape)
            R_data_gen = np.zeros(R_data_phase.shape)

            for k in range(0,end_date-phase[0]+1):
                if(phase[0]==0 and k<start_date):
                    continue
                elif(phase[0]==0 and k==start_date):
                    S_data_gen[k] = S_data.values[k]
                    beta_data_gen[k] = beta_data.values[k]
                    R_data_gen[k] = R_data.values[k]
                elif(phase[0]!=0 and k==0):
                    S_data_gen[k] = S_data.values[k+start_date]
                    beta_data_gen[k] = beta_data.values[k+start_date]
                    R_data_gen[k] = R_data.values[k+start_date]
                else:
                    for i in range(S_data_gen.shape[1]):

                        S_data_gen[k][i] = S_data_gen[k-1][i]*(1-np.dot(beta_data_gen[k-1],A_No_int_nn[i]))

                        beta_data_gen[k][i] = beta_data_gen[k-1][i]*(1-Gamma_nn[i]) + \
                                              S_data_gen[k-1][i]*np.dot(beta_data_gen[k-1],A_No_int_nn[i])

                        R_data_gen[k][i] = R_data_gen[k-1][i] + Gamma_nn[i]*beta_data_gen[k-1][i]

            S_data_gen =  pd.DataFrame(S_data_gen,columns = ['S'+name for name in self.column_names])
            beta_data_gen =  pd.DataFrame(beta_data_gen,columns = ['I'+name for name in self.column_names])
            R_data_gen = pd.DataFrame(R_data_gen,columns = ['R'+name for name in self.column_names])

            if(phase[0] == 0):
                S_data_gen_total = S_data_gen
                beta_data_gen_total = beta_data_gen
                R_data_gen_total = R_data_gen
            else:
                S_data_gen_total = pd.concat([S_data_gen_total,S_data_gen]).reset_index(drop=True)
                beta_data_gen_total = pd.concat([beta_data_gen_total,beta_data_gen]).reset_index(drop=True)
                R_data_gen_total = pd.concat([R_data_gen_total,R_data_gen]).reset_index(drop=True)


        ##################################################
        S_data_gen = S_data_gen_total
        beta_data_gen = beta_data_gen_total
        R_data_gen = R_data_gen_total
        ##################################################
        
        return A_No_int_nn_list, Gamma_nn_list, S_data_gen, beta_data_gen, R_data_gen
