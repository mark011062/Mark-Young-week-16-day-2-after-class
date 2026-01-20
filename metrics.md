For this run I used the FashionMNIST setup with a custom LearnedAffine layer and a three epoch training run. The training ran on CPU. There was no tuning beyond the default. The main goal of this project was to verify that the pipeline works end to end and that the custom layer behaves as expected during training. 



Here are the results after the three training epochs

Train loss: 0.5124 | Train acc: 0.8201

Val   loss: 0.4374 | Val   acc: 0.8420

Current LR: 0.000700



Epoch 2/3

Train loss: 0.3702 | Train acc: 0.8666

Val   loss: 0.3826 | Val   acc: 0.8622

Current LR: 0.000490



Epoch 3/3

Train loss: 0.3290 | Train acc: 0.8804

Val   loss: 0.3639 | Val   acc: 0.8696

Current LR: 0.000343

We see that after three epochs, validation accuracy landed at 0.8696 with validation loss trending downward. After reading about what should be expected, this tracks with what we should expect from a small model along with a short training schedule. 

What worked - The model trains gradients flow, validation accuracy improves, and the custom layer doesn't break autograd. 

What I'd change - I would run more epochs to see where the curve flattens. 

