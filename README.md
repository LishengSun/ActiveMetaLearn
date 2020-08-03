# ActivMetal
Code for our paper 'ActivMetaL'

# The repository is progressively updated to fix some technical issues... Stay tuned! 

## I. Install and test Cofirank
As described in the paper, our experiments have CofiRank as sub-routine. To reproduce these experiments, one needs to install [CofiRank](https://github.com/markusweimer/cofirank). The source code is included here, if you are in a Linux environment, you can compile it:  
```
cd /cofirank 
make -f CofiRank-Makefile.mk CONF=Deploy (compile)
```
and test the installation:
```
./dist/cofirank-deploy config/default.cfg (run test)
```
### Eventually use our Docker image
We have built a Docker image to ease the installation of Cofirank in a non-Linux environment: [activmetal](https://hub.docker.com/r/lisesun/activmetal/) .          
       
For example, suppose you have clone this repository at absolute_path/ActiveMetaLearn/cofirank:     
```
docker pull lisesun/activmetal (pull our image)
docker run -it -v absolute_path/ActiveMetaLearn/cofirank/:/ActiveMetaLearn activmetal (run image having volume mounted to ./ActiveMetaLearn)
```
Then inside the Docker container, we can compile and run CofiRank:    
```
cd /ActiveMetaLearn/cofirank 
make -f CofiRank-Makefile.mk CONF=Deploy (compile)
./dist/cofirank-deploy config/default.cfg (run test)
```

Note: This approach can suffer from the [Segmentation fault issue](https://github.com/LishengSun/ActiveMetaLearn/issues/2), which we are trying to solve. For the moment, we suggest using a virtual machine to create a Linux environment. A good reference can be found here: https://machinelearningmastery.com/linux-virtual-machine-machine-learning-development-python-3/. Any help for a better docker image is also welcome!

## II. Matrix visualization 
See our [ipython notebook](https://github.com/LishengSun/ActiveMetaLearn/blob/master/DEMONSTRATION/performance-matrix-visualization.ipynb)

## III. Run experiments
      
Assume CofiRank is installed and compiled, the experiments can be customized and run by
```
python run_experiments.py -d artificial -s active_meta_learning_with_cofirank -n True -rd /RESULTS/
```
which will apply the strategy of active meta learning with cofirank (-s active_meta_learning_with_cofirank) on the artifical dataset (-d artificial) after a global normalisation (-n True) of the dataset, and save the results (plots, scores and logging files) to the indicated result directory /RESULTS/ (-rd /RESULTS/, please replace /RESULTS/ with an absolute path to the directory where you want to save the results.)


