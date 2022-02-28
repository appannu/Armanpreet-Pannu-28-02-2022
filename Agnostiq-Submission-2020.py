import pennylane as qml
from pennylane import numpy as np

#set n and k where n is latent sapce and n+k in input space
n=5
k=2



# I need to choose a general form of the unitary that scales polynomially with the number of input qubits n
# this will depend on the form of the states I am trying to compress.

# For this example, I will use matrix product states (MPS) which are ground states of local Hamiltonians
# as such, it should suffice to take a unitary which is local. The bond dimension of MPS states depends on
# how "local" the unitary/Hamiltonian is but for simplicity, I will assume 2-local and hence the unitary U
# is a composition of 2 qubit sub-unitaries which apply on adjacent qubits.
# Furthermore, arbitrary 2 qubit unitaries are 4x4 unitary matricies which are hard to paramaterize in general,
# therefore I will only consider only 2 qubit unitaries which are 1 qubit control unitaries. i.e. the (i)-th qubit controls
# if a unitary is applied to the (i+1)th qubit.


# a 2x2 unitary matrix can be completely paramaterized as: [[w,z],[-e^i\theta z*,e^i\theta w*]]
# where w,z are complex numbers and theta is a phase with value between 0 and 2pi 
def unitary(params): #params is a 5 element array consisting of the real and imag part of w,z and then \theta
    w=params[0]+1.j*params[1]
    z=params[2]+1.j*params[3]
    t=params[4]
    return np.array([[w,z] ,[-np.exp(1.j*t)*np.conjugate(z),np.exp(1.j*t)*np.conjugate(w)]])

# Given that there are n+k-1 adjacent qubits (0,1),...,(n+k-2,n+k-1) and each corresponding unitary has 3 degrees of freedom,
# there are exactly (n+k-1)*3 paramaters in the global unitary U, which is polynomially scaling with n

#initial guess
p=np.ones((n-1,5),dtype=np.double)



# the next step to to contruct the quantum circuit 
def quantum_function(input_state,p): #p represents the paramaters of the unitary which is a (n-1)x5 matrix
    # the first step is to generate a quantum state which represents out input state that we want to compress
    # implementing general n-qubit states is very difficult (not to mention exponetially scaling) so as per the second part of 
    # the challenge, I will assume the input data is a binary (n+k)-vector (v_1,...,n_{n+k}) which corresponds to the
    # quantum state |v_1> x ... x |n_{n+k}>.
    for i in np.arange(k,n+k+k):
        if input_state[i-k]==1:
            qml.PauliX(wires=i)
    
    
    #generate reference, we will use \ket{0}^k as reference so no further action is needed
    
    
    #apply unitary as encoder, by looping oper all adjacent qubits and applying the sub-unitary paramaterized by p[i]
    for i in np.arange(n+k-1):
        print(i)
        j=int(i) #a bug in pennylane where it can't take integers from arrays so we need to redefine i as an independant int
        qml.ControlledQubitUnitary(unitary(p[j]),control_wires=j,wires=j+1)
        
    # swap the reference and the k qubits
    for i in np.arange(k):
        j=int(i)
        qml.SWAP(wires=[j,j+k])
        
    #apply adjoint of unitary as decoder
    for i in np.flip(np.arange(n+k-1)):
        j=int(i)
        qml.ControlledQubitUnitary(unitary(p[j]),control_wires=j,wires=j+1)
        
    
    first_k=[] #this is again due to the bug in pennylane as it can't directly accept wire lables from arrays as is necessary in the next step
    for i in np.arange(k):
        first_k.append(int(i))
        
    # fidelity computation:
    # since our reference state is simply |0>^k which is an element of the computational basis, the fidelity between this and the trash state is simply the probaility |0>^k in the computational basis
    
    return qml.probs(wires=first_k) #the fidelity is just the [0]th element



# for samke of example, we randomly choose training states and assign them equal weighting
num_training_states=3
test_states=np.random.randint(2, size=(num_training_states,n+k))
weight=(1./num_training_states)*np.ones(num_training_states) #equal weighting

# now define the cost function
def cost(p):
    p=p.reshape((n+k-1),5) #the classical optimizers only work with 1d arrays
    cost=0.
    #iterate over the training states
    for i in np.arange(num_training_states):
        #define the device with n+k+k wires, assume the first k are the reference
        dev=qml.device('default.qubit',wires=n+k+k)
        circuit = qml.QNode(quantum_function, dev)
        fidelity=circuit(test_states[i],p)[0]
        cost+=weight[i]*fidelity
    return cost


# traing the quantum encodedr by using a classical optimizer to determine paramaters p
# We will use the basic scipy optimizer
import scipy.optimize as optimize

#initial guess
p=np.random.randint(2, size=((n+k-1)*5))
result = optimize.minimize(cost, p)

print(result)
        

    
    
    
    
    
    
    
    


