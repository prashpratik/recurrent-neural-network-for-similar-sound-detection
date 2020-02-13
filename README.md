# Implementation of Recurrent Neural Network (RNN) for Similar Sound Detection
In this project, I have implemented Recurrent Neural Network (RNN) for similar sound detection.
As we know that sound can be represented as a time series data or a sequential data, we can use this property of sound to train a RNN model and then, let it predict the output sound.
For this particular project, I have used three sounds, hello1.wav, hello2.wav and bye.wav. In hello1.wav and hello2.wav, there is a sound of hello and in bye.wav, there is a sound of bye.
Our aim is to let model decide which sounds are similar. For this, I have trained the model with hello1.wav and then, let the model predict the output for other inputs hello2.wav and bye.wav separately.
The output which will give minimum total loss or minimum Mean Square Error (MSE) will be similar to the trained sound.
I have plotted different graphs and plots for them as well to compare the results.
