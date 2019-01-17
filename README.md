

How can I improve the convergence of this genetic algorithm in training an auto encoder on MNIST?

My example code below (only two files / a few hundred lines of python including the GA lib)  trains a three layer auto encoder on MNIST using the GA.

https://github.com/patniemeyer/ga-autoencoder

My code appears to be working but compared to back-propagation is incredibly slow.  I keep reading papers and anecdotes about training networks with millions of free parameters and seeing results in just a few generations, however my (comparatively small, 200k parameter) network converges extremely slowly.  With a training set limited to only a hundred images I have seen it continue to make (slow) progress after 50k generations.

Things I’ve tried / learned:
1. Crossover doesn’t seem to help so I’ve just maxed out mutation.
2. I have subtracted the mean MNIST image from the input batches, since the algorithm always seems to need to find the mean image (huge local minima) before differentiating between the digits.
3. I’ve insured that there are always small, single value, single layer mutations available so that it should never gets completely “stuck”.
4. I’ve tried to make sure that my weight initialization has a distribution similar to what I know to be decent final results.  (But even if I’m wildly off it should just take longer to converge, right?)
5. I’ve tried experimenting with “safe mutations” basing the mutation magnitude on the gradient of the output with respect to the weights, but I did not see any improvement in simple tests and gave up (perhaps prematurely).  Regardless, I would prefer not to be dependent on being able to calculate the gradients - that’s why I’m using a GA! :)
6. I have experimented with pytorch’s weight_norm() wrapper that separates the direction and magnitude of the weight tensor and makes them layer parameters: My thinking was that if I’m wildly off in the initialization perhaps this would give the GA more leverage on it.  I really had high hopes that this would help but again, either I’m doing it wrong or there was no improvement.

I am aware that I can improve the runtime performance of the GA through parallelization and reduce the memory footprint by storing only seed values for mutations, etc.  Right now I’m just trying to hone the algorithm and see some results before spending more time tweaking performance.



