# Setting up env

In the DistSys Directory, run bash install.sh to download all the required packages.
Next, set your $GOPATH to include the DistSys directory:

1. DistSys/install.sh
2. export GOPATH=$PWD

# simpleBlockChain

In the DistSys Directory run the following:

1. sudo $GOPATH/bin/DistSys node_id total_nodes dataset 

  E.g sudo $GOPATH/bin/DistSys 0 4 creditcard
  
  Runs a node with Id 0 in a network of 4 nodes each with a part of the creditcard dataset
  
  Node Ids start from 0 upto (numberOfNodes - 1)
  

