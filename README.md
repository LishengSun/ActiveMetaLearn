# ActivMetal
Code for our paper 'ActivMetaL'

## I. Docker image
We have built a Docker image to ease using CofiRank: [activmetal](https://hub.docker.com/r/lisesun/activmetal/) .          
To use it, you need to first download CofiRank from [their website](https://github.com/markusweimer/cofirank), then run it inside a Docker container of our image.        
For example, suppose you have downloaded CofiRank at absolute_path/cofirank:     
```
docker pull lisesun/activmetal (pull our image)
docker run -it -v absolute_path/cofirank/:/cofirank activmetal (run image having volume mounted to ./cofirank)
```
Then inside the Docker container, we can compile and run CofiRank:    
```
cd /cofirank 
make -f CofiRank-Makefile.mk CONF=Deploy (compile)
./dist/cofirank-deploy config/default.cfg (run test)
```

## II. Matrix visualization 
See our [ipython notebook](https://github.com/LishengSun/ActiveMetaLearn/blob/master/DEMONSTRATION/performance-matrix-visualization.ipynb)

## III. Run experiments
As described in the paper, our experiments have CofiRank as sub-routine. To reproduce these experiments, one needs to install CofiRank (as explained in sec. I).          
Assume CofiRank is installed and compiled (as shown in cofirank/), then the experiments can be customized and run by
```
python -i run_experiments.py -d 'artificial' -s 'active_meta_learning_with_cofirank' 'random' 'simple_rank_with_median' -n True -rd '/RESULTS/'
```
Codes for experiments are pregressively uploaded ...
