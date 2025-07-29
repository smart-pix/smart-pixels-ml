import tensorflow as tf
from keras.callbacks import Callback
import numpy as np

# Linear scheduler
class LinearScheduler(Callback):
    def __init__(
        self,
        reg_weight_var,
        start=0.5,            # Initial regularization weight λ
        end=5.0,              # Maximum λ to reach
        max_epochs=500,       # Number of epochs over which λ will increase linearly
        stop_threshold=-25000 # Validation loss threshold beyond which λ stops increasing
    ):
        super().__init__()
        self.reg_weight = reg_weight_var 
        self.start = start
        self.end = end
        self.max_epochs = max_epochs
        self.stop_thresh = stop_threshold

    def on_train_begin(self, logs=None):
        # Set the initial λ value
        self.reg_weight.assign(self.start)
        print(f"reg_weight initialized to {self.start:.4f}")

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return  # If validation loss is not available, do nothing

        if val_loss > self.stop_thresh:
            # If val loss is still above the threshold, increase λ linearly based on the current epoch
            α = min((epoch + 1) / self.max_epochs, 1.0)
            new_w = self.start + α * (self.end - self.start)
            self.reg_weight.assign(new_w)
            print(f"[Epoch {epoch+1}] val_loss={val_loss:.1f} > {self.stop_thresh} → reg_weight={new_w:.4f}")
        else:
            # If val loss is below or equal to the threshold, freeze λ at its current value
            print(f"[Epoch {epoch+1}] val_loss={val_loss:.1f} ≤ {self.stop_thresh} → reg_weight frozen at {self.reg_weight.numpy():.4f}")

# Adaptive scheduler:
# The reg weight adapts its value depending on the nll loss performance:
# If current nll is improving → reg weight doesn’t change
# f current nll is stalled → the reg weight increases
# If current nll is worsening → the reg weight decreases

class AdaptiveScheduler(tf.keras.callbacks.Callback):
    def __init__(self, reg_weight_var, start=0.5, max_reg_weight=5.0, step=0.05, patience=15):
        self.reg_weight = reg_weight_var 
        self.start = start
        self.max_reg_weight = max_reg_weight
        self.step_up = step
        self.step_down = step
        self.patience = patience
        self.best_nll = float('inf')
        self.wait = 0

    def on_train_begin(self, logs=None):
        # Set the initial λ value
        self.reg_weight.assign(self.start)
        print(f" Initial reg_weight set to: {self.start:.4f}")

    def on_epoch_end(self, epoch, logs=None):
        metrics = get_loss_metrics()
        nll = float(metrics['nll_mean'])
        reg_weight = float(self.reg_weight.numpy())

        # Improvement condition: significant NLL decrease
        if nll < self.best_nll - 1e-2:
            self.best_nll = nll
            self.wait = 0
            print(f"[Epoch {epoch+1}] NLL improved to {nll:.2f} → reg_weight stays at {reg_weight:.4f}")
        else:
            self.wait += 1
            # Static condition: no improvement for 'patience' epochs
            if self.wait >= self.patience and reg_weight < self.max_reg_weight:
                new_weight = min(reg_weight + self.step_up, self.max_reg_weight)
                self.reg_weight.assign(new_weight)
                print(f"[Epoch {epoch+1}] NLL plateaued  → reg_weight increased to {new_weight:.4f}")
                self.wait = 0  # reset counter

        # Worsening condition: NLL significantly worse
        if nll > self.best_nll + 0.5:
            new_weight = max(reg_weight - self.step_down, self.start)
            self.reg_weight.assign(new_weight)
            print(f"[Epoch {epoch+1}] NLL worsened  → reg_weight reduced to {new_weight:.4f}")


# Cosine scheduler: 
# Weights increase like a cosine, it starts slow, then faster in the middle and slow again at the end
class CosineScheduler(Callback):
    def __init__(self, reg_weight_var, start=0.5, end=10.0, max_epochs=500, stop_threshold=-25000):
        super().__init__()
        self.reg_weight = reg_weight_var 
        self.start = start
        self.end = end
        self.max_epochs = max_epochs
        self.stop_thresh = stop_threshold
        

    def on_train_begin(self, logs=None):
        # Set the initial λ value
        self.reg_weight.assign(self.start)
        print(f"reg_weight initialized to {self.start:.4f}")

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return  

        # If val loss is still above the threshold, update λ using a cosine schedule from start to end
        if val_loss > self.stop_thresh:

            # Normalize epoch index to [0, 1]
            α = min((epoch + 1) / self.max_epochs, 1.0)

            # Smooth increase from 0 to 1
            cos_progress = 0.5 * (1 - np.cos(np.pi * α))  

            # Scale cosine output α to the desired λ range
            new_w = self.start + cos_progress * (self.end - self.start)

            # Update the regularization weight
            self.reg_weight.assign(new_w)
            print(f"[Epoch {epoch+1}] val_loss={val_loss:.1f} > {self.stop_thresh} → reg_weight={new_w:.4f}")
        else:
            # If val loss is low enough, freeze λ at current value
            print(f"[Epoch {epoch+1}] val_loss={val_loss:.1f} ≤ {self.stop_thresh} → reg_weight frozen at {self.reg_weight.numpy():.4f}")
            
# Sigmoid scheduler:
# Weight starts slowly, becomes steeper around the middle of training, and then levels off as training progresses. 
class SigmoidScheduler(Callback):
    def __init__(self, reg_weight_var, start=0.5, end=10.0, max_epochs=500, stop_threshold=-25000, sharpness=10):
        super().__init__()
        self.reg_weight = reg_weight_var 
        self.start = start                
        self.end = end                   
        self.max_epochs = max_epochs      
        self.stop_thresh = stop_threshold 
        self.sharpness = sharpness # How steep the sigmoid transition is
        

    def on_train_begin(self, logs=None):
        # Set the initial λ value
        self.reg_weight.assign(self.start)
        print(f"🔧 reg_weight initialized to {self.start:.4f}")

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return

        if val_loss > self.stop_thresh:
            
            # Normalize epoch index to [0, 1]
            x = (epoch + 1) / self.max_epochs
            
            # Apply sigmoid function centered at x = 0.5
            α = 1 / (1 + np.exp(-self.sharpness * (x - 0.5)))
            
            # Scale sigmoid output α to the desired λ range
            new_w = self.start + α * (self.end - self.start)
            
            # Update the regularization weight
            self.reg_weight.assign(new_w)
            print(f"[Epoch {epoch+1}] val_loss={val_loss:.1f} > {self.stop_thresh} → reg_weight={new_w:.4f}")
        else:
            # If val loss is low enough, freeze λ at current value
            print(f"[Epoch {epoch+1}] val_loss={val_loss:.1f} ≤ {self.stop_thresh} → reg_weight frozen at {self.reg_weight.numpy():.4f}")

#=== loss trackers ===

nll_tracker = tf.keras.metrics.Mean(name="nll_loss")
reg_tracker = tf.keras.metrics.Mean(name="reg_loss")

def track_loss_values(nll, reg):
    nll_tracker.update_state(nll)
    reg_tracker.update_state(reg)

def reset_loss_trackers():
    nll_tracker.reset_state()
    reg_tracker.reset_state()

def get_loss_metrics():
    return {
        'nll_mean': float(nll_tracker.result().numpy()),
        'reg_mean': float(reg_tracker.result().numpy()),
    }


    