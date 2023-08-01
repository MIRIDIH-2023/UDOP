import numpy as np


def huber_loss(pred,label, threshold):
    
    mae = np.abs(pred-label)
    mse = 0.5 * (pred-label)**2
    
    loss_map = mae < threshold
    loss = np.where( loss_map, 
                    mse, 
                    threshold * mae - 0.5 * threshold**2 )
    
    return loss



def custom_loss(pred,label, threshold, eps=1e-6):
    
    mae = np.abs(pred-label)
    ln_loss = np.log( mae + eps)   # ln(x), 미분하면 1/x
    mse = 0.5 * (pred-label)**2  # 0.5 x^2 , 미분하면 x
    
    loss_map = ln_loss < threshold  #equal as...   mae < e^threshold 7.8xx
    loss = np.where( loss_map, 
                    mse, 
                    threshold**2  * (ln_loss - np.log(threshold) + 0.5) )
    
    return loss

a = np.random.normal(loc=250, scale=10, size=10)
a = np.clip(np.round(a), 0, 500).astype(int)

b = np.random.normal(loc=250, scale=10, size=10)
b = np.clip(np.round(b), 0, 500).astype(int)

print(custom_loss(a,b,2)) # 0~26

#0 500  0 ~26   7픽셀까지는 mse