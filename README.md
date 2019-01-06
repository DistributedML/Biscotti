# Welcome!

# Dependencies

We use the go-python library for interfacing between the the distributed system code in Go and the ML logic in python. Unfortunately, Go-python doesn't support python versions > 2.7.12. Please ensure that your default OS python version is 2.7.12.

# Setting up env

Inside azure/azure-setup, there is an install script called azure-install.sh. Run this script to install go, all the related dependencies. The script also clones this repo for you too.

# Running Biscotti

## Local deployment

Go to the DistSys folder. Run the script localTest.sh using the following format:

```
bash localTest.sh <numNodes> <dataset>

```
For example

bash localTest.sh 10 creditcard

```

## Non-local deployment

1. You must create a file in azure/azure-conf containing the list of all ip's where the nodes are going to be deployed.

2. For deploying Biscotti on different machines, you need to have set up ssh-access to all other machines from your local machine using your public key.

3. On each machine, install all dependencies using the azure-install.sh script above.

4. Deploy biscotti on your machines by running the runBiscotti script in azure/azure-run.

```
bash runBiscotti.sh <nodesInEachVM> <totalNodes> <hostFileName> <dataset>

```

For example if you want to deploy 100 nodes across 20 machines using the mnist dataset, then run the script using the following command.

```

bash runBiscotti.sh 5 100 hostFile mnist

```