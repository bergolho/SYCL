import sys
import numpy as np
import matplotlib.pyplot as plt

def plot_transmembrane_potential(t1, v1, t2, v2, t3, v3):
	#plt.grid()
    plt.plot(t1, v1, label="CUDA", c="darkgreen", linewidth=2.0, linestyle='dashed')
    plt.plot(t2, v2, label="SYCL-cpu", c="orange", linewidth=3.0, alpha=0.7)
    plt.plot(t3, v3, label="SYCL-gpu", c="red", linewidth=1.0, linestyle='--')
    plt.xlabel("t (ms)",fontsize=15)
    plt.ylabel("V (mV)",fontsize=15)
    plt.title("Action potential - TenTusscher",fontsize=14)
    plt.legend(loc=0,fontsize=14)
    #plt.show()
    plt.savefig("ten_tusscher_solution_device_comparison.png", dpi=150)

def main():
	
    if len(sys.argv) != 4:
        print("----------------------------------------------------------------------------------------")
        print("Usage:> python %s <input_file_1> <input_file_2> <input_file_3>" % sys.argv[0])
        print("----------------------------------------------------------------------------------------")
        print("<input_file_1> = Input file 1 with the state variables from each timestep (CUDA)")
        print("<input_file_2> = Input file 2 with the state variables from each timestep (SYCL-cpu)")
        print("<input_file_3> = Input file 3 with the state variables from each timestep (SYCL-gpu)")
        print("----------------------------------------------------------------------------------------")
        return 1

    input_file_1 = sys.argv[1]
    input_file_2 = sys.argv[2]
    input_file_3 = sys.argv[3]
    data_1 = np.genfromtxt(input_file_1)
    data_2 = np.genfromtxt(input_file_2)
    data_3 = np.genfromtxt(input_file_3)

    plot_transmembrane_potential(data_1[:,0], data_1[:,1], data_2[:,0], data_2[:,1], data_3[:,0], data_3[:,1])


if __name__ == "__main__":
	main()
