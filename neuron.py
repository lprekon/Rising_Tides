import random
class Neuron(object):
	"""A Neuron for a Neural Network"""

	def defaultFire(self):
		value = self.eval(self)
		for (neuron, ident) in self.outputs:
			neuron.recvInput(value, ident)
		self.lastVal = value
		self.inputs = []	# wipe inputs clean to allow refiring (future proofing for non-linear networks)

	def __init__(self, evalRule, fireRule = defaultFire):
		super(Neuron, self).__init__()
		self.weights = []   # The list of weights to be applied to inputs. Order matters
		self.inputs=[]      # The input values after adjusting by weight. The order doesn't matter
		self.outputs=[]     # A list of duples in the form (<receiving neuron>, <ident>) where <ident> is the index of the
							#  appropriate weight in the receiver
		self.eval = evalRule
		self.fire = fireRule
		self.lastVal = 0

	def recvInput(self, value, ident):
		self.inputs.append((value + 0.0001) * self.weights[ident])
		if len(self.inputs) == len(self.weights):
			self.fire(self)

	def addInput(self):		# add new random weight and return its index
		self.weights.append(random.randrange(-1000, 1000, 1)/1000)  # Because randrange requires integer steps
		return len(self.weights) - 1

	def printStats(self):
		print("Neuron " + str(id(self)) + " receives from " + str(len(self.weights)) + " other neurons with weights:")
		print(self.weights)
		print("and sends to " + str(len(self.outputs)) + " other neurons")
		print("It last fired value " + str(self.lastVal) + "\n")
