import numpy
from PIL import Image
import math



class InputLayer(object):
	"""The input layer of a neural network"""

	def generateFactors(n):
		start = int(math.sqrt(n))
		print("finding factors for " + str(n) + " starting at " + str(start))
		for i in range (0, n):
			if n % (start - i) == 0:
				return (start - i, int(n / (start - i)))
			if n % (start + i) == 0:
				return (start + i, int(n / (start + i)))
		print("[*] Warning: You have picked a prime for number of input nodes")
		return (n, 1)

	def __init__(self, n):
		super(InputLayer, self).__init__()
		self.numNeurons = n  # doesn't count bias neuron
		self.inbox = [None] * n
		self.activationVector = numpy.zeros(n + 1)
		self.activationVector[n] = 1  # set output of bias neuron
		self.aspectRatio = InputLayer.generateFactors(n)
		print("Aspect Ration " + str(self.aspectRatio[0]) + ":" + str(self.aspectRatio[1]))
	
	def receiveInput(self, image):
		# image must be 3:4 aspect ratio
		(rawX, rawY) = image.size
		chunkSizeX = int(rawX / self.aspectRatio[0])
		chunkSizeY = int(rawY / self.aspectRatio[1])
		if chunkSizeY <= 1 or chunkSizeX <=1:
			print("[*] Warning: Image too small. May lead to system degredation.")
		i = 0
		for y in range (0, self.aspectRatio[1]):
			for x in range(0, self.aspectRatio[0]):
				self.inbox[i] = image.crop((x * chunkSizeX, y * chunkSizeY, (x + 1) * chunkSizeX, (y + 1) * chunkSizeY))
				i += 1
		#print("[+] Input succesfully received. Awaiting activation...")

	def activate(self):
		for i in range(0, self.numNeurons):
			data = self.inbox[i].getdata()
			x = sum(data) / len(data)
			x -= 177  # Shift the data so it is more meaningful to the sigmoid function
			output = 1 / (1 + math.exp(-x))
			self.activationVector[i] = output
		#print("[+] Input layer succesfully activated. Ready for data transfer...")

class HiddenLayer(object):
	"""The hidden layer of a neural network"""
	def __init__(self, n):
		super(HiddenLayer, self).__init__()
		self.numNeurons = n  # Doesn't count bias neuron
		self.rawInput = numpy.zeros(n)
		self.weightedInput = numpy.zeros(n)
		self.activationVector = numpy.zeros(n + 1)
		self.activationVector[n] = 1  # Set activation for bias neuron

	def receiveInput(self, inputActivation):
		self.rawInput = inputActivation.copy()
		#print("[+] Hidden layer has received input. Awaiting activation...")

	def activate(self):
		self.weightedInput = self.weightMatrix.dot(self.rawInput)
		try:
			for i in range(0, self.numNeurons):
				self.activationVector[i] = 1 / (1 + math.exp(-self.weightedInput[i]))
		except OverflowError:
			print("Overflowed on " + str(self.weightedInput[i]))
			print("weightedInput:")
			print(self.weightedInput)
			print("HiddenLayer weight matrix:")
			print(self.weightMatrix)
		#print("[+] Hidden layer succesfully activated. Ready for data transfer...")

class OutputLayer(object):
	"""The output layer of a neural network"""
	def __init__(self, n):
		super(OutputLayer, self).__init__()
		self.numNeurons = n
		self.rawInput = numpy.zeros(n)
		self.weightedInput = numpy.zeros(n)
		self.outbox = numpy.zeros(n)

	def receiveInput(self, inputActivation):
		self.rawInput = inputActivation.copy()
		#print("[+] Output layer has received input. Awaiting activation...")

	def activate(self):
		#print("[*] Processing...")
		self.weightedInput = self.weightMatrix.dot(self.rawInput)
		for i in range(0, self.numNeurons):
			self.outbox[i] = 1/(1 + math.exp(-self.weightedInput[i]))
		#print("[+] Complete. Results in outbox.")

	def displayOutbox(self):
		for i in range(0, 26):
			print(chr(i + 65) + ": " + str(self.outbox[i]))