import socket
import os
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("-f","--file",default="train.sh")
args=parser.parse_args()

s = socket.socket()
host = socket.gethostname()
port = 10386
s.bind((host, port))
s.listen(0)

f=open(args.file,"r")
for i in f.readlines():
	c, addr = s.accept()
	print("link address:", addr)
	try:
		c.send(i.strip().encode("utf-8"))
		print("message sent:",i)
	except:
		print("failed sending message.")
		continue
	c.close()
f.close()