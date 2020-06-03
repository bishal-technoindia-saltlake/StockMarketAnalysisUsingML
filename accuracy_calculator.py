import numpy as np
def main(y_predict: np.ndarray, y_actual: np.ndarray, length: int):
        count=0
        total=0
        for i in range (length): 
            total = total+1
            res1 = "{0:.1f}".format(y_predict[i])
            res2 = "{0:.1f}".format(y_actual[i])
            if res1 == res2:
                count = count + 1
        accuracy = count*100//total
        return accuracy