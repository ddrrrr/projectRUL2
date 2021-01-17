import numpy as np

def Bin_Encoder(x,array_len):
    output = np.zeros(array_len)
    binx = bin(int(x))
    for i in range(len(binx)-1):
        output[-i] = int(binx[-i])
    for i in range(1,array_len):
        output[-i] = int(output[-i-1])^int(output[-i])
    return output

def Bin_Decoder(x,array_len):
    output = 0
    for i in range(array_len):
        if i > 0:
            output += 2**(array_len-1-i) * (int(x[i-1])^int(x[i]))
        else:
            output += 2**(array_len-1) * x[0]
    return output