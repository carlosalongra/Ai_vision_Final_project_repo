# Real-Time Handwritten Digit Recognition to Speech

In this repository, you can find two models, CNN and AlexNet, which are trained to detect handwritten digit numbers using the MNIST dataset.

Steps to run the files:

1. Train the model using python "CNN/finalModel.py" or "python AlexNet/AlexNet.ipynb".

2. Depending on the model you want to test, go to process_image.py and change the path to the desired model: 
        model = load_model('CNN/Final_Model.h5')  or 'AlexNet/AlexNet.ipynb'

3. Run the Pygame app using python app.py and test the app.

How to use the app:

1. The left side of the screen is used to write the digit using the right click on your mouse. If you wish to clear the screen, click the left click on your mouse.

2. The right side of the screen is used to detect the number and show the prediction of the number introduced by the user. It has no function for the user to use other than displaying the output.