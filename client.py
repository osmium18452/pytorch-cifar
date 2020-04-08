import socket
import argparse
import time
import os

parser=argparse.ArgumentParser()
parser.add_argument("-g","--gpu",default="0")
parser.add_argument("-s","--server",default="192.168.81.1")
args=parser.parse_args()

host = args.server
port = 10386
while True:
	s = socket.socket()
	s.connect((host, port))
	cmd=s.recv(1024).decode("utf-8")
	cmd=cmd+" -g "+args.gpu
	os.system(cmd)
	time.sleep(0.1)
	s.close()

