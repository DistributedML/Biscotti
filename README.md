# Biscotti: machine learning on the blockchain

Biscotti is a fully decentralized peer-to-peer system for multi-party machine learning (ML). Peers participate in the learning process by contributing (possibly private) datasets and coordinating in training a global model of the union of their datasets. Biscotti uses blockchain primitives for coordination between peers and relies on differential privacy and cryptography techniques to provide privacy and security guarantees to peers. For more details about Biscotti's design, see our [Arxiv paper](https://arxiv.org/abs/1811.09904).

# Dependencies

We use the the go-python library for interfacing between the distributed system code in Go and the ML logic in Python. Unfortunately, Go-python doesn't support Python versions > 2.7.12. Please ensure that your default OS Python version is 2.7.12.

# Setting up the environment

Inside azure/azure-setup, there is an install script called azure-install.sh. Run this script to install Go and all the related dependencies. The script also clones this repo for you.

# Running Biscotti

## Local deployment

Go to the DistSys folder. Run the script localTest.sh with:

```
bash localTest.sh <numNodes> <dataset>

```
For example
```
bash localTest.sh 10 creditcard

```

## Non-local deployment

1. You must create a file in azure/azure-conf containing the list of all IPs of the peer nodes.

2. To deploy Biscotti on different machines, you need to have set up ssh-access to all other machines from your local machine using your public key.

3. On each machine, install all the dependencies using the azure-install.sh script above.

4. Deploy Biscotti on your machines by running the runBiscotti script in azure/azure-run.

```
bash runBiscotti.sh <nodesInEachVM> <totalNodes> <hostFileName> <dataset>

```

For example, if you want to deploy 100 nodes across 20 machines using the mnist dataset, then run the script as follows:

```

bash runBiscotti.sh 5 100 hostFile mnist

```
