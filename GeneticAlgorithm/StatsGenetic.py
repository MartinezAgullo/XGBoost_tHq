import numpy as np
import pandas as pd
import math
from scipy.stats import norm

class CommonStats():

    def AsymptoticPoissonPoissonModel(self, s, b, sigma):

        n = s+b
        sigma *= b

        f1 = n * np.log((n * (b+sigma**2)) / (b**2 + (n*sigma**2)))
        f2 = -(b**2/sigma**2) * np.log(1 + ((sigma**2 * (n-b)) / (b*(b+sigma**2))))
        Zn = np.sqrt(2*(f1 + f2))

        #dist = norm(s, sigma)
        #if s<1:
        #    Zn = 1/(2-s)
        #else:
        #    Zn = np.exp(s**2)/b
        #n= 0
        #while n<s:
        #    n= n+1
        #sig = s/np.sqrt(b + s)
        #Zn = 1/(1-sig)
        #print('Dist: average %f deviation %f'%(s,sigma))
        if math.isnan(Zn):
            print('Warning Zn is NaN')
            Zn = -99
            return Zn
        #if n < b:
        #    return -1*Zn
        #else:
        #    return Zn
        return Zn
class cross_mutate:

    def __init__(self):

        self.crossover_rate = 0.9
        self.mutation_rate = 0.9
        self.mutation_stddev = 0.8
        self.ignore = 'Zn'
        self.return_df = ''

    def run(self, df):
        debug = True
        if debug: print("[CM run] cross_mutate.run()")
        temp_df = pd.DataFrame(columns=df.columns)
        npairs = int(df.shape[0]/2.) # df.shape[0] returns the number of rows. 
        index = 0
        if debug: print("[CM run] Hyperparms of the dataframe to cross-mutate:")
        if debug: print(df.iloc[: , :3])
        if debug: print("[CM run] Organaises the dataframe into "+str(npairs)+" pairs for cross pairing")
        for i in range(npairs): # Runs over the first half of the df
            if debug: print(" ")
            if debug: print("[CM run] Execute cross_pair for pair "+str(i))
            if debug: print("[CM run] Pair consisting in the rows "+str(index)+" and "+str(index+1))
            pair = df.iloc[index:index+2] # Takes the two rows between index and index+2
            temp_df = temp_df.append(self.cross_pair(pair),ignore_index=True)
            temp_df.reset_index(drop=True,inplace=True)
            index += 2 
        if debug: print("[CM run] cross_pairing complete for all pairs:")
        if debug: print(df.iloc[: , :3])
        #print self.return_df
        #print 'DF_TO_mutate'
        #print temp_df
        if debug: print("[CM run] After cross_pairing, we do cross_pairing")
        self.mutate(temp_df)
        #print 'TEST_after_mutate'
        #print self.return_df
        return self.return_df    

    def mutate(self, df):
        temp_df = pd.DataFrame(columns=df.columns)
        debug = True
        if debug: print("[CM mutate] Mutating the population of hyperparameters. For each hypeparameter, "+str(self.mutation_rate*100)+"% probability of mutating.")
        for row in range(df.shape[0]):
            if debug: print("[CM mutate]  - Mutating row "+str(row))
            for column in df.columns:
                if column != self.ignore:
                    #if debug: print("[CM mutate]    80% probability of mutating "+str(column)+"for row "+str(row))
                    if np.random.uniform(0.0, 1.0) < self.mutation_rate: # Each parameter has 80% of chances of being modified
                        if debug: print("[CM mutate]   Mutating column "+str(column))
                        #if df.ix[row, column] < 10:
                        if df.ix[row, column]==0:
                            precission= 0
                        else:
                            precission = round(math.log10(abs(df.ix[row, column])),0) 
                            #df.ix retuns a cell. math.log10() method returns the base-10 logarithm of a number.
                            if debug: print("[CM mutate]    precission param for ["+str(row)+", "+str(column)+"] = "+str(precission))
                        if math.isnan(precission):
                            print('[CM mutate]    Warining: precission is NaN')
                            print(df.at[row, column])
                            precission = -1 # y listos xd
                        else:
                            if debug: print("[CM mutate]    Mutate multiplying by a random normally distributed between ["+str(1-self.mutation_stddev)+", "+str(1+self.mutation_stddev)+"]")
                        if precission > 0:
                            if debug: print("[CM mutate]    precission > 0 :: Round the hyperparam to 1 decimal")
                            temp_df.at[row, column] = round(df.ix[row, column] * np.random.normal(1,self.mutation_stddev), 1)
                            # pablo: I would change how the mutation is done 
                            if debug: print("[CM mutate]    The hyperparam "+str(column)+" mutated from "+str(df.ix[row, column])+" to "+str(temp_df.at[row, column]))
                        else:
                            if debug: print("[CM mutate]    precission > 0 :: Round the hyperparam to "+str(int(abs(precission))+2)+" decimals")
                            temp_df.at[row, column] = round(df.ix[row, column] * np.random.normal(1,self.mutation_stddev), int(abs(precission))+2)
                            if debug: print("[CM mutate]    The hyperparam "+str(column)+" mutated from "+str(df.ix[row, column])+" to "+str(temp_df.at[row, column]))
                        #else:
                        #    temp_df.at[row, column] = self.myround(df.ix[row, column] * np.random.normal(1,self.mutation_stddev))
                    else:
                        # Not do mutation
                        if debug: print("[CM mutate]   Not mutating"+str(column))
                        temp_df.at[row, column] = df.ix[row, column]
                else:
                    # Fill Zn cell with -99
                    if debug: print("[CM mutate]   Not mutating for "+str(column)+". Filling with -99")
                    temp_df.at[row, column] = -99     
        self.return_df = temp_df 

    def myround(self, value, base = 5000):
        return base * round(value / base)        

    def cross_pair(self, df):
        debug = True
        if debug: print("[CM cross_pair] A dataframe with two rows is added for cross-pair:")
        if debug: print(df)
        df.reset_index(drop=True,inplace=True)
        self.column_count = df.shape[1] - 1 # Number of columns minus 1.
        #if debug: print("[CM cross_pair] Reset df index for cross pairing")
        #if debug: print(df)
        do_crossover = False
        if debug: print("[CM cross_pair] A probability of "+str((self.crossover_rate)*100)+"% to perform crossover for this pair of rows")
        if np.random.uniform(0.0,1.0) < self.crossover_rate:
            do_crossover = True
        cross_point = -99
        if do_crossover:
            if debug: print("[CM cross_pair] Doing cross-over :: Mixing hyperparams between two rows")
            temp_df = pd.DataFrame(columns=df.columns)
            cross_point = int(round(np.random.uniform(0.0,self.column_count),0)) 
            # Produces a random integer between 0 and 
            # the number of columns minus one.

            index = 0
            for column in df.columns: #Loop over columns
                if column != self.ignore:
                    #if debug: print("[CM cross_pair]   Probability to cross-over for "+str(column)+".")
                    if cross_point > index:
                        if debug: print("[CM cross_pair]  - "+str(column)+" :: Conserve the values")
                        temp_df.at[0,column] = float(df.ix[0,column])
                        temp_df.at[1,column] = float(df.ix[1,column])
                    else:
                        if debug: print("[CM cross_pair]  - "+str(column)+" :: Substite values for the pair")
                        temp_df.at[0,column] = float(df.ix[1,column])
                        temp_df.at[1,column] = float(df.ix[0,column])
                else:
                    if debug: print("[CM cross_pair]  - "+str(column)+" ::  -99 is assigned for both rows")
                    temp_df.at[0,column] = -99.
                    temp_df.at[1,column] = -99.       
                index += 1
            if debug: print("[CM cross_pair] The rows have been changed to:")
            if debug: print(temp_df)
            return temp_df
        else:
            if debug: print("[CM cross_pair] Not doing cross-over")
            return df        
      
