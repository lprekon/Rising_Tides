import numpy
import random
import re
# A generic layer of neurons for a neural network, with dense connections.
class NeuronLayer():
	"""A generic layer of neurons in a neural network"""

	def _init_(self, height, width, includeBiasNode=True, generateWeight=False, loadMatrix=False, matrix=None):
		self.height = height
		self.width = width
		self.inputVector = numpy.zeros(width)
		self.includeBiasNode = includeBiasNode
		height = (includeBiasNode) ? height + 1 : height
		self.outputVector = numpy.zeros(height)
		self.outputVector[height - 1] = 1  # If this isn't a bias node, it will just get overwritten later
		if generateWeight:
			generateWeightMatrix(self)
		if loadMatrix:
			self.weightMatrix = matrix

	def generateWeightMatrix(self):
		width = self.width
		height = self.height
		self.weightMatrix = numpy.zeros((height, width))
		for j in range(height):
			for i in range(width):
				self.weightMatrix[j][i] = random.uniform(-5, 5)

	def setWeightMatrix(self, matrix):
		self.weightMatrix = matrix

class NeuralNetwork():

	def _init_(self, inputSize=5, layerSizes={}, includeBiasNode=True, weightFileFile=None):  # Use layerSizes and weightFile to signal desire to generate or load the matrix, respectively.
		if weightFile != None:
			f = open(weightFile, 'rb')
			numLayers = int(f.readline().rstrip())
			for l in range(numLayers):
				w,h = [int(x) for x in f.readline().]
		else if layerSizes != {}:
			self.neuralLayers = {}
			width = inputSize
			for height in layerSizes:
				self.neuralLayers.append(NeuronLayer(height=height, width=width, includeBiasNode=includeBiasNode, generateWeight=True))
				width = height + (includeBiasNode) ? 1 : 0

	def serializeMatrix(matrix):
		rows, columns = matrix.size
		serialMatrix = "{(" + rows +"," columns + ")"
		for r in range(rows):
			for c in range(columns):
				serialMatrix += matrix[r,c]


	def deSerializeMatrix(matrixString):
		dimensions = re.match(r"\{\((\d*),(\d*)\)", matrixString)
		rows = int(dimensions.group(1))
		columns = int(dimensions.group(2))
		matrix = numpy.zeros((rows, columns))
		numList = re.match(r"\{\(\d*,\d*\)((?:\d*,?)*)\}")
		count = 0
		for num in numList.split(","):
			matrix[int(count / columns), count % columns] = int(num)
			count = count + 1
		return matrix