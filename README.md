# Matching Transform
This project uses backprogation of mean squared error to learn a hidden 4x4 transformation matrix. 
Gradient descent may be overkill here but I see it as good practice.
We observe (calcuate) input and output vectors to generate batches of for the optimization section.
The derivative of squared error is assigned to the "weights" our guess transformation.
An finally we take a step.
And seems to be working pretty well.
Perhaps it is trivial without a nonlinearity, but the problem is linear (transformation matrix).
Also, this cuda code may not be very efficient (some unneccessary gpu-cpu movement, etc.)

I've also included an excel spreadsheet where a hidden projection vector is learned in the same manner.
Input in A3-6 and output in CBC3-6 (160 steps).
