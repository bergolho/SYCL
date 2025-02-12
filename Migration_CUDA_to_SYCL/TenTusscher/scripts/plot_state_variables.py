import sys
import numpy as np
import matplotlib.pyplot as plt

def plot_transmembrane_potential(t, v):
	#plt.grid()
	plt.plot(t, v, label="Vm", c="black", linewidth=2.0)
	plt.xlabel("t (ms)",fontsize=15)
	plt.ylabel("V (mV)",fontsize=15)
	plt.title("Action potential",fontsize=14)
	plt.legend(loc=0,fontsize=14)
	plt.show()
	#plt.savefig("ap.pdf")

def main():
	
    if len(sys.argv) != 2:
        print("-------------------------------------------------------------------------")
        print("Usage:> python %s <input_file>" % sys.argv[0])
        print("-------------------------------------------------------------------------")
        print("<input_file> = Input file with the state variables from each timestep")
        print("-------------------------------------------------------------------------")
        return 1

    input_file = sys.argv[1]
    data = np.genfromtxt(input_file)

    plot_transmembrane_potential(data[:,0], data[:,1])


if __name__ == "__main__":
	main()
