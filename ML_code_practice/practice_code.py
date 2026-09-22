import matplotlib.pyplot as plt

labels=['python','java','c++','javascript']
sizes=[40,25,20,15]

plt.pie(sizes,labels=labels,autopct='%1.1f%%')

plt.title("programing language usage")

plt.show()
