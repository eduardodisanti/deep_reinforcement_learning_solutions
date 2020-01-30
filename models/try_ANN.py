import numpy as np
from models.ANN import ANN2
from models.ANN import ANN

print("==========[ ANN ]==========")
NN = ANN(4, 5, 2, 2)
NN.init()
x = np.array([0, 1, 2, 3])
o1 = NN.forward(x)
print("PREDICTION", o1)

p = NN.get_params()
print("PARAMETERS 1", p)
NN = ANN(4, 5, 2, 2)
NN.init()
NN.set_params(p)
print("PARAMETERS 2", NN.get_params())
o2 = NN.forward(x)
print("PREDICTION WITH SAME PARAMETERS", o2)
print("==========================================================")


print("==========[ ANN2 ]==========")
NN = ANN2(5, 4, 3, 2, 2)
NN.init()
x = np.array([0, 1, 2, 3, 5])
o1 = NN.forward(x)
print("PREDICTION", o1)
p = NN.get_params()
print("PARAMETERS 1", p)

NN = ANN2(5, 4, 3, 2, 2)
NN.init()
NN.set_params(p)
print("PARAMETERS 2", NN.get_params())
o2 = NN.forward(x)
print("PREDICTION WITH SAME PARAMETERS", o2)
print("==========================================================")
