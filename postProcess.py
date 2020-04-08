import matplotlib.pyplot as plt

# "1","3","4"
list = {
	"0": "VGG19",
	"1": "ResNet18",
	"3": "GoogLeNet",
	"4": "DenseNet121"
}
for file in ("0","1","3","4"):
	with open("./save/" + file + "/result.txt") as f:
		adam = []
		sgd = []
		lr = []
		for line in f.readlines():
			wordlist = line.strip().split(" ")
			if wordlist[5] == "adam":
				lr.append(float(wordlist[3]))
				adam.append(float(wordlist[7]))
			else:
				sgd.append(float(wordlist[7]))
		# print(adam,sgd,lr,sep="\n")
		lr.reverse()
		adam.reverse()
		sgd.reverse()
		x=range(len(lr))
		plt.figure()
		plt.xticks(x,lr)
		plt.xlabel("Learning Rate")
		plt.ylabel("Final Accuracy (%)")
		plt.title("Final Accuracy of " + list[file] + " Using Adam and SGD Optimizer")
		kwargs = {
			"marker": "o",
			"lw":2
		}
		plt.plot(x, sgd, label="SGD train accuracy", **kwargs)
		plt.plot(x, adam, label="Adam train accuracy", **kwargs)
		plt.legend()
		sv = plt.gcf()
		sv.savefig("./save/"+list[file]+".png", format="png", dpi=100)
		plt.show()
