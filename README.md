# Physics-Informed-Deep-Learning
A Generic Data-Driven Framework via Physics-Informed Deep Learning
- Dependencies

  - [Matplotlib](https://matplotlib.org/)
  - [NumPy](http://www.numpy.org/)
  - [TensorFlow](https://www.tensorflow.org/)>=2.2.0
  - [DeepXDE](https://github.com/lululxvi/deepxde)

1. Bidirectional LSTM Model:

  - [Using LSTM network to predict time-series temperatures / clustering approach](https://github.com/softsys4ai/Physics-Informed-Deep-Learning/blob/main/LSTMNetwork/Analytical_Thermal_Clustering.ipynb)
   <img align="center" width="98" alt="Capture" src="https://user-images.githubusercontent.com/45353778/113974235-d6000d80-9852-11eb-8483-d698eb5d7488.PNG">
    

2. Physics-Informed Neural Network Model:
    ![Picture1](https://user-images.githubusercontent.com/45353778/113974667-8b32c580-9853-11eb-9cdb-e782f6c40f36.png)
    - [Solve a sample 2D heat equation](https://github.com/softsys4ai/Physics-Informed-Deep-Learning/blob/main/PINN/2D_heat_equation.ipynb)
    - [Solve the 2D heat equation on our data](https://github.com/softsys4ai/Physics-Informed-Deep-Learning/blob/main/PINN/PINN.ipynb)
    - [Solve a 2D heat equation for one time-step](https://github.com/softsys4ai/Physics-Informed-Deep-Learning/blob/main/PINN/One_Time_Step.ipynb)
    - [Solve the 2D heat equation for many time-steps by re-initializing the model](https://github.com/softsys4ai/Physics-Informed-Deep-Learning/blob/main/PINN/Many_time_steps.ipynb)
    - [Solve the 2D heat equation using re-training and clustering](https://github.com/softsys4ai/Physics-Informed-Deep-Learning/blob/main/PINN/Many_time_steps_Retraining.ipynb)
