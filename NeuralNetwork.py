import sys
import NeuronLayers
import numpy
from PIL import Image
import time
import random
import math

inputLayer = None
hiddenLayer = None
outputLayer = None
learningFactor = .05
def generateNeuralNetwork(neuronsPerLayer):
	print("[*] Generating network...")
	global inputLayer
	global hiddenLayer
	global outputLayer
	inputLayer = NeuronLayers.InputLayer(neuronsPerLayer[0])
	hiddenLayer = NeuronLayers.HiddenLayer(inputLayer.aspectRatio[0] + inputLayer.aspectRatio[1])
	outputLayer = NeuronLayers.OutputLayer(neuronsPerLayer[1])
	hiddenLayer.weightMatrix = numpy.zeros((hiddenLayer.numNeurons, inputLayer.numNeurons + 1))  # Plus 1 to account for bias neuron
	for j in range (0, hiddenLayer.weightMatrix.shape[0]):
		for k in range(0, hiddenLayer.weightMatrix.shape[1]):
			hiddenLayer.weightMatrix[j][k] = random.uniform(-1, 1)
	outputLayer.weightMatrix = numpy.zeros((outputLayer.numNeurons, hiddenLayer.numNeurons + 1))  # See above
	for j in range (0, outputLayer.weightMatrix.shape[0]):
		for k in range(0, outputLayer.weightMatrix.shape[1]):
			outputLayer.weightMatrix[j][k]  = random.uniform(-1, 1)
	print("[+] Success!")


def activateNetwork(inp):
	inputLayer.receiveInput(inp)
	inputLayer.activate()
	hiddenLayer.receiveInput(inputLayer.activationVector)
	hiddenLayer.activate()
	outputLayer.receiveInput(hiddenLayer.activationVector)
	outputLayer.activate()


def main():
	if len(sys.argv) < 3:
		sys.exit("Usage: " + sys.argv[0] + " <input number>, <image>")
	inNum = int(sys.argv[1])
	outNum = 26
	imageFile = sys.argv[2]
	generateNeuralNetwork((inNum, outNum))
	if imageFile != "train":
		im = Image.open(imageFile).convert('L')
		print("[*] Forward Propogating")
		activateNetwork(im)
		outputLayer.displayOutbox()
	else:
		print("let's begin.")
		#print("first row of starting output matrix:")
		#print(outputLayer.weightMatrix[0])
		#print("first row of starting hidden matrix:")
		#print(hiddenLayer.weightMatrix[0])
		for i in range(20):
			for routine in range(50):
				print("starting routine " + str(routine + 1))
				for trainingSet in range(0, 50):
					outputGradient = numpy.zeros(outputLayer.weightMatrix.shape)
					hiddenGradient = numpy.zeros(hiddenLayer.weightMatrix.shape)
					for lesson in range(0, 20):
						letter = int(random.uniform(0, 25.9))  # slightly reduced chance of picking 'Z' but I reckon that's ok
						movAvg = 1/(20 - lesson)
						image = Image.open(chr(65 + letter) + ".png").convert("L")
						activateNetwork(image)
						# Forward progation complete. Begin backpropogation
						outputError = numpy.zeros(outputLayer.numNeurons)
						for i in range(0, outputLayer.numNeurons):
							expected = 0
							if(i == letter):
								expected = 1
							wInput = outputLayer.weightedInput[i]
							outputError[i] = (outputLayer.outbox[i] - expected) * math.exp(wInput)/((math.exp(wInput) + 1) ** 2)  # sigma-prime(x) = e^x/(e^x + 1)^2
						for j in range(0, outputLayer.numNeurons):
							for k in range(0, hiddenLayer.numNeurons + 1):  # Plus 1 to account for bias neuron, which isn't included in numNeurons
								outputGradient[j][k] = (1 - movAvg) * outputGradient[j][k] + movAvg * outputError[j] * hiddenLayer.activationVector[k]
						hiddenError = numpy.transpose(outputLayer.weightMatrix).dot(outputError)
						for i in range(0, hiddenLayer.numNeurons):
							wInput = hiddenLayer.weightedInput[i]
							hiddenError[i] *= math.exp(wInput)/((math.exp(wInput) + 1) ** 2)
						for j in range(0, hiddenLayer.numNeurons):
							for k in range(0, inputLayer.numNeurons + 1):  # Plus 1 to account for bias neuron, which isn't included in numNeurons
								hiddenGradient[j][k] = (1 - movAvg) * hiddenGradient[j][k] + movAvg * hiddenError[j] * inputLayer.activationVector[k]
					hiddenGradient *= learningFactor
					outputGradient *= learningFactor
					outputLayer.weightMatrix = outputLayer.weightMatrix - outputGradient
					hiddenLayer.weightMatrix = hiddenLayer.weightMatrix - hiddenGradient
				#print("first row of final output matrix:")
				#print(outputLayer.weightMatrix[0])
				#print("first row of final hidden matrix:")
				#print(hiddenLayer.weightMatrix[0])
				print("testing A")
				image = Image.open("A.png").convert("L")
				activateNetwork(image)
				outputLayer.displayOutbox()
			print("All training sets complete. Running final test...")
			for i in range(0, 26):
				char = chr(65 + i)
				image = Image.open(char + ".png").convert("L")
				activateNetwork(image)
				selectedOutput = outputLayer.outbox[i]
				avgOtherOutput = (sum(outputLayer.outbox[:i]) + sum(outputLayer.outbox[i+1:])) / (outputLayer.numNeurons - 1)
				print("[" + char + "]: " + str(selectedOutput))
				print("[avg. other output]: " + str(avgOtherOutput) + "\n")


if __name__ == '__main__':
	main()
