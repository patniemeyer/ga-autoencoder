
How can I improve the convergence of this genetic algorithm in training an auto encoder on MNIST?

*autoencoder-ga.py* - A classic GA implementation with selection based on fitness.

autoencoder-es.py - An evolutionary strategy that uses a sum of the entire population weighted by fitness (https://arxiv.org/pdf/1703.03864.pdf)

The GA code appears to be working but compared to back-propagation is incredibly slow.  
The ES code is fast but gets stuck and exhibits weird behavior.  It seems very senstive to the training parameters and amount of data.

Both algorithms exhibit the interesting behavior of starting off with a flat line on the fitness curve before eventually "discovering" something and then beginning the descent.  I would like to better understand what is happening at that inflection point.

I am aware that I can improve the runtime performance of both of these through parallelization and reduce the memory footprint by storing only seed values for mutations, etc.  Right now I’m just trying to hone an algorithm and see some results before spending more time tweaking performance.


