import numpy
import random
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

