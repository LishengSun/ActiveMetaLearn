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
Coming soon

## III. Run experiments
Coming soon
