import pandas as pd
import numpy as np

class Linear_Model:
    def __init__(self, Features:list [str], Target: str):
        self.Weights = [0] * len(Features) 
        self.Bias = 0
        self.Features = Features
        self.Target = Target
        pass

    def Train_Model(self, Data_Frame, Epochs: int, Learning_Rate:float):
        for iteration in range(Epochs):
            adjustment_Weights = [0] * len(self.Weights)
            adjustment_Bias = 0
            numb_of_data_point = len(Data_Frame)
            for n in range(numb_of_data_point):

                input_Lists = [None] * len(self.Features)
                for x in range(len(input_Lists)):
                    input_Lists[x] = Data_Frame[self.Features[x]].iloc[n]
                Real_Output = Data_Frame[self.Target].iloc[n]

                Predict_Output = 0
                for x in range(len(input_Lists)):
                    Predict_Output += self.Weights[x] * input_Lists[x]
                Predict_Output += self.Bias

                Error = Real_Output - Predict_Output
                for x in range(len(input_Lists)):
                    adjustment_Weights[x] += (-2)/numb_of_data_point * (Error) * input_Lists[x] 
                adjustment_Bias = (-2)/numb_of_data_point * Error
            
            for i in range (len(self.Features)):
                self.Weights[i] = self.Weights[i] - Learning_Rate * adjustment_Weights[i] 
            self.Bias = self.Bias - Learning_Rate * adjustment_Bias 
        return 
    
    def Test_Model(self, Data_Frame):
        summ = 0
        for i in range(len(Data_Frame)):
            slope_collection = 0
            y = Data_Frame[self.Target].iloc[i]
            for x in range(len(self.Weights)):
                slope_collection += (self.Weights[x]  * Data_Frame[self.Features[x]].iloc[i])
            summ += (y - (slope_collection + self.Bias))**2 

        total_of_square_sums = 0 
        y_mean = Data_Frame[self.Target].mean() 
        for i in range(len(Data_Frame[self.Target])):
            y = Data_Frame[self.Target].iloc[i]
            total_of_square_sums += (y-y_mean)**2      
        return 1 - (summ / total_of_square_sums) 

        return summ
     
    def Predict(self, Input: list):
        slope_Total = 0
        for i in range(len(self.Weights)):
            slope_Total += self.Weights[i] * Input[i]
        return slope_Total + self.Bias
    
    def Get_Weights(self):
        return self.Weights
    
    def Get_Bias(self):
        return self.Bias



class Polynomial_Model:
    def __init__(self, Degrees_Vector: list, Features:list[str], Target: str):
        if len(Features) != len(Degrees_Vector):
            raise ValueError("Number of Degrees and Features Must Match. i.e. every polynomial line representing a feature needs a given degree.")
        self.Features = Features
        self.Weight_Vector_List = [None] * len(self.Features)
        self.Degrees_Vector = Degrees_Vector 
        self.Target = Target
        pass

    def Train_Model(self, Data_Frame, Epochs: int, Learning_Rate: float):
        number_Of_Data_Points = len(Data_Frame)
        Input_Vector_List = [None] * len(self.Features)
        Y_Vector_Real = np.zeros((number_Of_Data_Points, 1))   
        #Load each accordin list size 
        for i in range(len(self.Features)):
            Weight_Vector = np.zeros((self.Degrees_Vector[i] + 1, 1))
            self.Weight_Vector_List[i] = Weight_Vector
            DR_Weight = np.zeros_like(self.Weight_Vector_List[i])        
        # Initialize the variables that need values
        for index in range(len(self.Features)):
            Input_Vector = np.zeros((number_Of_Data_Points, self.Degrees_Vector[index] + 1))
            for power in range(self.Degrees_Vector[index] + 1):
                Input_Vector[:, power] = Data_Frame[self.Features[index]].values ** power
            Input_Vector_List[index] = Input_Vector
        Y_Vector_Real[:,0] = Data_Frame[self.Target].values
        #Gradient Descent Down here!
        for iteration in range(Epochs):
            Y_Vector_Predict = np.zeros_like(Y_Vector_Real)
            for i in range(len(Input_Vector_List)):
                Y_Vector_Predict +=  np.dot(Input_Vector_List[i], self.Weight_Vector_List[i])
            
            for i in range(len(self.Weight_Vector_List)):
                Error_Vector = Y_Vector_Real - Y_Vector_Predict
                Input_Vector_Transposed = np.transpose(Input_Vector_List[i])
                DR_Weight =  (-2/number_Of_Data_Points) * np.dot(Input_Vector_Transposed, Error_Vector)
                self.Weight_Vector_List[i] = self.Weight_Vector_List[i] - Learning_Rate * DR_Weight
        return 
        
    def Test_Model(self, Data_Frame):
        # Sum of Square...
        residuals = 0
        # Total of Square...
        sums = 0        
            
        real = Data_Frame[self.Target].to_numpy()
        prediction = np.zeros_like(real)
        mean = real.mean()


        for numb_of_feat in range(len(self.Features)):
            feature = self.Features[numb_of_feat]
            weights = self.Weight_Vector_List[numb_of_feat].reshape(-1)
            degrees = len(weights) - 1
            Inputs = Data_Frame[feature].to_numpy()
            for power in range(degrees + 1):
                prediction += weights[power] * (Inputs ** power)

        residuals = np.sum((real - prediction) ** 2)
        sums = np.sum((real - mean) ** 2)

        return 1 - residuals / sums

    def Predict(self, Inputs):
        if len(Inputs) != len(self.Features):
            raise ValueError("len(Inputs) != len(self.Features)")
        prediction = 0
        for numb_of_feat in range(len(self.Features)):
            weights = self.Weight_Vector_List[numb_of_feat].reshape(-1)
            degrees = len(weights) - 1
            Input = Inputs[numb_of_feat]
            for power in range(degrees + 1):
                prediction += weights[power] * (Input ** power)
        return prediction

    def Get_Weights(self):
        return self.Weight_Vector_List

