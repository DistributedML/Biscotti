# Setting up env

In the DistSys Directory, run the install script to download all the required packages.
Next, set your $GOPATH to include the DistSys directory:

1. `cd DistSys`  
2. `bash install.sh`  
3. `export GOPATH=$PWD`  

# simpleBlockChain

In the DistSys Directory run the following:

`sudo $GOPATH/bin/DistSys -i node_id -t total_nodes -d dataset`  

For example,  
`sudo $GOPATH/bin/DistSys -i 0 -t 4 -d creditcard`  

Runs a node with Id 0 in a network of 4 nodes each with a part of the creditcard dataset  
Node Ids start from 0 upto (numberOfNodes - 1)
  

