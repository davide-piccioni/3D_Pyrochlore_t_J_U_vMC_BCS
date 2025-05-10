import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


def read_complex_energy_file(filename):
    # Load the file and parse each line as a complex number
    with open(filename, 'r') as file:
        data = []
        for line in file:
            line = line.strip()
            if line and line.startswith('(') and line.endswith(')'):
                try:
                    real, imag = map(float, line.strip('()').split(','))
                    data.append(complex(real, imag))
                except ValueError:
                    print(f"Skipping invalid line: {line}")
            else:
                try: 
                    real = float(line)
                    data.append(complex(real, 0))
                except ValueError:
                    print(f"Skipping invalid line: {line}")
    return np.array(data)


@njit
def bin_processing(e_raw,N_tot,N_bins,bin_length):
    e_mean_real = np.zeros(N_bins)
    e_std_real = np.zeros(N_bins)
    e_mean_imag = np.zeros(N_bins)
    e_std_imag = np.zeros(N_bins)

    e_bins_real = np.zeros(N_bins)
    e_bins_imag = np.zeros(N_bins)
    N_start = N_tot - N_bins*bin_length
    for ind_bin in np.arange(N_bins):
        for index_N in range(N_start,N_start+bin_length):
            e_bins_real[ind_bin] += np.real(e_raw[index_N])
            e_bins_imag[ind_bin] += np.imag(e_raw[index_N])
        e_bins_real[ind_bin] = e_bins_real[ind_bin] / bin_length
        e_bins_imag[ind_bin] = e_bins_imag[ind_bin] / bin_length
        N_start = N_start + bin_length
    
    e_mean_real[0] = e_bins_real[0]
    e_mean_imag[0] = e_bins_imag[0]
    e_std_real[0]  = 0.
    e_std_imag[0]  = 0.
    for i in range(2,N_bins+1):
        e_mean_real[i-1] = np.mean(e_bins_real[:i])
        e_std_real[i-1]  = np.std(e_bins_real[:i])/np.sqrt((i))
        e_mean_imag[i-1] = np.mean(e_bins_imag[:i])
        e_std_imag[i-1]  = np.std(e_bins_imag[:i])/np.sqrt((i))
        
    return e_mean_real + 1j*e_mean_imag, e_std_real + 1j*e_std_imag


# Main function to handle the folder and integer input
def process_files(folder_path):
    # Check if folder exists and contains "*raw.dat" files
    if os.path.exists(folder_path):
        raw_files = glob.glob(os.path.join(folder_path, '*raw.dat'))
        if len(raw_files) == 0:
            print("Are you sure there are files containing observables?")
            return
    else:
        print("Folder does not exist.")
        return

    bin_length = 0
    n_bins_skipped = 0

    energy_file = os.path.join(folder_path, 'energy_raw.dat')
    if os.path.exists(energy_file):
        e_raw = read_complex_energy_file(energy_file) 

        N_tot = e_raw.size
        print("The file energy_raw.dat contains "+str(N_tot)+" energies.") 
        bin_length = max(1,int(input("Which bin lenght do you want to set? ")))

        n_bins_skipped = max(1,int(abs(int(input("How many bins do you want to skip? ")))))

        N_bins = int( (N_tot - n_bins_skipped*bin_length)/bin_length )

        if (N_bins < 10 ):
            print("You were setting a bin_length too big, resulting in less than 10 bins.")
            while N_bins < 10:
                bin_length = int(bin_length/2)
                N_bins = int( (N_tot - n_bins_skipped*bin_length)/bin_length )
        
        print("\n\nPerforming binning on the energy data <H> with:")
        print("\nbin_length: "+str(bin_length))
        print("N_bins = "+str(N_bins))

        e_mean, e_std = bin_processing(e_raw,N_tot,N_bins,bin_length)

        print("\n<H> = ",e_mean[-1]," ± ",e_std[-1],"\n\n")

        fig=plt.figure(figsize=(9, 8))
        plt.errorbar(np.arange(1,N_bins+1),np.real(e_mean),yerr=np.real(e_std))
        plt.title('Real part of Energy = <H>', fontsize=18)
        plt.ticklabel_format(useOffset=False)
        plt.xlabel('Bins', fontsize=14)
        plt.ylabel('Re(E)', fontsize=14)
        plt.grid(True)
        plt.savefig(folder_path+'Real_Energy_bins.pdf')
        plt.show()

        fig=plt.figure(figsize=(9, 8))
        plt.errorbar(np.arange(1,N_bins+1),np.imag(e_mean),yerr=np.imag(e_std))
        plt.title('Imaginary part of Energy = <H>', fontsize=18)
        plt.ticklabel_format(useOffset=False)
        plt.xlabel('Bins', fontsize=14)
        plt.ylabel('Im(E)', fontsize=14)
        plt.grid(True)
        plt.savefig(folder_path+'Imag_Energy_bins.pdf')
        plt.show()
        
    else:
        print("\nenergy_raw.dat not found.\n")
        return

    energy_squared_file = os.path.join(folder_path, 'energy_squared_raw.dat')
    if os.path.exists(energy_squared_file):
        e_raw = read_complex_energy_file(energy_squared_file) 

        N_tot = e_raw.size
        
        # bin_length and n_bins_skipped are already defined

        N_bins = int( (N_tot - n_bins_skipped*bin_length)/bin_length )

        if (N_bins < 10 ):
            print("You were setting a bin_length too big, resulting in less than 10 bins.")
            while N_bins < 10:
                bin_length = int(bin_length/2)
                N_bins = int( (N_tot - n_bins_skipped*bin_length)/bin_length )
        
        print("\n\nPerforming binning on the observable <H^2> with:")
        print("\nbin_length: "+str(bin_length))
        print("N_bins = "+str(N_bins))

        e_mean, e_std = bin_processing(e_raw,N_tot,N_bins,bin_length)

        print("\n<H^2> = ",np.real(e_mean[-1])," ± ",np.real(e_std[-1]),"\n\n")

        fig=plt.figure(figsize=(9, 8))
        plt.errorbar(np.arange(1,N_bins+1),np.real(e_mean),yerr=np.real(e_std))
        plt.title('<H^2>', fontsize=18)
        plt.ticklabel_format(useOffset=False)
        plt.xlabel('Bins', fontsize=14)
        plt.ylabel('<H^2>', fontsize=14)
        plt.grid(True)
        plt.savefig(folder_path+'Real_EnergySquared_bins.pdf')
        plt.show()

    else:
        print("energy_raw.dat not found.")
        return
    return


# Example of calling the function
if __name__ == "__main__":
    folder_path = input("Enter folder path: ")  # e.g., "/path/to/folder/"
    process_files(folder_path)
