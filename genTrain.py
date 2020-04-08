with open("train.sh", "w+") as f:
	for model in (0, 1, 3, 4):
		for lr in (0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001):
			for opt in ("sgd", "adam"):
				print("python ./main.py", "--model", model, "--lr", lr, "-o", opt,"-d","./save/"+str(model), file=f)
