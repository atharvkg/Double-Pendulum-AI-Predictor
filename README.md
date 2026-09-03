# Double-Pendulum-AI-Predictor
This code uses RK4 and the ODEs for the double pendulum system to generate a dataset of thousands of short trajectories. Then it trains a Physics Informed Long Short Term Memory(PILSTM) model on the dataset using torch. I implement energy checks, Lyapunov calculations to determine chaotic behavior, a sensitivity demonstration etc.
