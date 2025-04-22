import matplotlib.pyplot as plt

cores = [1, 2, 4, 8, 16]
times = [42.25, 14.44, 7.56, 4.25, 2.66]
speedup = [times[0] / t for t in times]

plt.plot(cores, speedup, marker='o', label='Measured')
plt.plot(cores, cores, 'k--', label='Ideal linear')
plt.xlabel('Number of Cores')
plt.ylabel('Speed-up')
plt.title('Speed-up vs. Number of Cores (Static Scheduling)')
plt.legend()
plt.grid(True)
plt.savefig("plots/task5a_speedup.png")
plt.show()
