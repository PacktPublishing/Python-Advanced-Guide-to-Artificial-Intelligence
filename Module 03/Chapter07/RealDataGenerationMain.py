'''
Created on 09-May-2018

@author: aii32199

'''
import numpy as np
import matplotlib.pyplot as plt
from Chapter_1_GANs.GAN_Simple import RealInput,RandomInput

# Create a main() function to call our defined method       
def main():        
    
    #We will generate a Gaussian distribution with 10000 points
    npoints = 10000
    nrange = 25
    #Let's call our real data generator
    #x = RealInput(npoints)
    x=np.concatenate((np.random.normal(0,1,5000),np.random.normal(5,1,5000))) 
    x_ = RandomInput(nrange, npoints)
    #Once we got the sample points we will calculate the histogram of the sample space;
    #Which is also known as probability distribution function. it will require number of
    #bins in which we want to distribute our sample points.
    
    #We will evaluate our histogram for 100 bins in the range of -10 to 10
    nbins = 100
    nrange = 10
    
    #Linspace from numpy will help us to create linear spacing between the points
    bins = np.linspace(-nrange, nrange, nbins)
    
    #We will use numpy's histogram function for creating PDF for desired bins
    pd, _ = np.histogram(x, bins=bins, density=True)
    p_,_ = np.histogram(x_, bins=bins, density=True)
    p_x = np.linspace(-nrange, nrange, len(pd))
    
    #Following lines will help us out in visualization of histogram        
    plt.plot(p_x, pd, label='real data')
    #plt.plot(p_x, p_, label='random data')
    plt.legend()
    plt.show()
    
main()