'''
This file is used to verify how many amounts of valid data we have, and filtering invalid data.
And: plot p, u values.

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Data_Purify():
    '''
    ### Purify the outlier data.
    Arguments:\n
    
    filename, 

    purify_factor_p, maximum value of p, default=10, \n
    purify_factor_u, maximum value of u, default=10
    
    '''
    def __init__(self,filename, purify_factor_p=10, purify_factor_u=10) -> None:


        self.purify_factor_p = purify_factor_p
        self.purify_factor_u = purify_factor_u
        # Load the CSV dataset
        df = pd.read_csv(filename, header=None)

        # Assuming the first row is the header, adjust if needed
        data = df.values  # Transpose to have shape (2, n)


        self.data_x = []
        self.data_u = []
        self.data_p = []
        self.data_r = []

        for row in range(data.shape[0]//2):
            tmp_input_data = [data[2*row:2*row+2,i] for i in range(data.shape[1]) if i%4==0]
            tmp_data_u = [data[2*row,i+3] for i in range(data.shape[1]) if i%4==0]
            tmp_data_r = [data[2*row+1,i+3] for i in range(data.shape[1]) if i%4==0]
            tmp_data_p = [data[2*row:2*row+2,i+1:i+3] for i in range(data.shape[1]) if i%4==0]
            # tmp_data_u = np.array(data_u)
            self.data_x.extend(tmp_input_data)
            self.data_u.extend(tmp_data_u)
            self.data_p.extend(tmp_data_p)
            self.data_r.extend(tmp_data_r)



        print('----------DATA SUMMARY------------')
        print(f'There are {len(self.data_u)} raw data points')
        # print(f'Total number of references: {1+(max(self.data_r)-min(self.data_r))//0.09}')


    def purified_data(self):

        data_p1 = [x[0,0] for x in self.data_p]
        data_p2 = [x[0,1] for x in self.data_p]
        data_p3 = [x[1,1] for x in self.data_p]

        data_p1 = np.array(data_p1)
        data_p2 = np.array(data_p2)
        data_p3 = np.array(data_p3)

        self.data_u = np.array(self.data_u)
        mask_outlier1 = np.abs(data_p1) < (self.purify_factor_p)
        mask_outlier3 = np.abs(data_p2) < (self.purify_factor_p)
        mask_outlier4 = np.abs(data_p3) < (self.purify_factor_p)
        mask_outlier2 = np.abs(self.data_u) < (self.purify_factor_u)
        self.mask_outlier = np.logical_and(
            np.logical_and(mask_outlier1,mask_outlier2),
            np.logical_and(mask_outlier3,mask_outlier4)
        )

        # for outliers
        self.outlier_mask = np.logical_not(self.mask_outlier)
        self.outliers = np.array(self.data_x)[self.outlier_mask]
        # use mask get rid of outliers
        self.data_p = np.array(self.data_p)[self.mask_outlier]
        self.data_r = np.array(self.data_r)[self.mask_outlier]
        self.data_x = np.array(self.data_x)[self.mask_outlier]
        self.data_u = np.array(self.data_u)[self.mask_outlier]
        print(f'Dataset is purified! Now there are {len(self.data_r)} data points available.')
        print('--------------------------------')

        self.len_data = len(self.data_u)

        # Final Data for feeding NN.

        # combine [x1,x2] and r as a single input:
        self.input_data_combine = [np.append(self.data_x[i],self.data_r[i]) for i in range(len(self.data_x))]

        # now we only need 4: p1,p2,p3,u
        self.label_data = [[self.data_p[i][0,0],self.data_p[i][0,1],self.data_p[i][1,1],self.data_u[i]] for i in range(len(self.data_u))]

        return self.input_data_combine, self.label_data

    def outliters_plot(self):
        # show x1 and x2 of outliers
        plt.figure()
        plt.scatter(self.outliers[:,0], self.outliers[:,1])
        plt.title('distributions of outliers')
        plt.show()
        plt.close()

    def one_reference(self,ref):
        # choose only one reference as our dataset
        selected_mask = self.data_r==ref
        
        self.one_r = self.data_r[selected_mask]
        self.one_p = np.array(self.data_p)[selected_mask]
        self.one_x = np.array(self.data_x)[selected_mask]
        self.one_u = np.array(self.data_u)[selected_mask]
        # Final Data for feeding NN.

        # combine [x1,x2] and r as a single input:
        self.input_data_one = [np.append(self.one_x[i],self.one_r[i]) for i in range(len(self.one_x))]

        # now we only need 4: p1,p2,p3,u
        # !!! IMPORTANT: I decided to normalize p, because p values are much more higher than u
        # self.one_p_normalized = self.one_p/np.mean(self.data_p)

        self.label_data_one = [[self.one_p[i][0,0],self.one_p[i][0,1],self.one_p[i][1,1],self.one_u[i]] for i in range(len(self.one_u))]

        print(f'Dataset with reference {ref} is built! \n Now there are {self.one_u.shape[0]} data points available.')
        print('--------------------------------')

        return self.input_data_one, self.label_data_one

    def ROI_data(self):
        '''
        split dataset near the center
        '''
        mask_x = np.logical_and((self.data_x[:,0] <= 5*np.pi/180), (self.data_x[:,1] >= -5*np.pi/180))
        mask_r = np.abs(self.data_r) <= 1*np.pi/180
        mask_roi = np.logical_and(mask_x,mask_r)
        # combine mask

        self.roi_r = np.array(self.data_r)[mask_roi]
        self.roi_p = np.array(self.data_p)[mask_roi]
        self.roi_x = np.array(self.data_x)[mask_roi]
        self.roi_u = np.array(self.data_u)[mask_roi]
        # Final Data for feeding NN.

        # combine [x1,x2] and r as a single input:
        self.input_data_roi = [np.append(self.roi_x[i],self.roi_r[i]) for i in range(len(self.roi_x))]

        # now we only need 4: p1,p2,p3,u
        self.label_data_roi = [[self.roi_p[i][0,0],self.roi_p[i][0,1],self.roi_p[i][1,1],self.roi_u[i]] for i in range(len(self.roi_u))]

        print(f'ROI dadaset is built! \n Now there are {self.roi_u.shape[0]} data points available.')
        print('--------------------------------')

        return self.input_data_roi, self.label_data_roi

    def draw_data(self, data_u, data_p):

        plt.subplot(221)
        plt.plot(list(range(len(data_u))),data_u)
        # plt.show()
        plt.title('Parameter u(t)')
        # plt.close()

        plt.subplot(222)
        data_p1 = [x[0,0] for x in data_p]
        plt.plot(list(range(len(data_p1))),data_p1)
        # plt.show()
        plt.title('Parameter p(1)')

        plt.subplot(223)
        data_p2 = [x[0,1] for x in data_p]
        plt.plot(list(range(len(data_p2))),data_p2)
        # plt.show()
        plt.title('Parameter p2')

        plt.subplot(224)
        data_p3 = [x[1,1] for x in data_p]
        plt.plot(list(range(len(data_p3))),data_p3)
        plt.show()
        plt.title('Parameter p3')
        plt.close()



'''


# data =  Data_Purify('drone_1_z.csv',90,3)
data =  Data_Purify('drone_1_z.csv',250,3)
_,_ = data.purified_data()
# _,_ = data.ROI_data()
_,_ = data.one_reference(0)
# data.outliters_plot()
data.draw_data(data.data_u,data.data_p)
data.draw_data(data.one_u,data.one_p)


'''