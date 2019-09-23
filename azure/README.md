This folder contains all the scripts related to azure for Biscotti 

## Setting upVM's:

All scripts related to this are under the folder azure-setup. Inside the folder there is a README.md that contains instructions on how to set up VM's in same/different locations in azure and get all related dependencies installed in each machine. The folder contains an script called azure-install.sh which is copied over to each VM and executed to install all dependencies in step 5. The script will need to be rerwitten if using for a different project. Otherwise the procedure should remain the same. 

## Deployment:

All scripts related to this are under azure-run. The main script that I execute is runBiscotti.sh which essentially generates bootstrap files that all nodes need (pk/sk/ etc) and copies over these along with the updated Biscotti go binary to all other hosts. Each VM is then ssh'd into and the binary is executed at each host. The log files are copied over to my machine at the end of execution.

For running federated learning, all scripts under azure-run-FedSys. Deployment procedure is similar

## Utilities:
A bunch of scripts are located inside azure-util that can be used for managing VM's/deployment.
1. kill-all.sh: kills all running nodes at each VM
2. pull-all.sh: gets the updated Biscotti code from the repo at all VM's
3. stop-all-vms.sh: stops all vms in a VM resource group.
4. start-all-vms.sh: starts all vms in a VM resource group.
5. add-alias-bashrc.sh : Adds alias of format ssh$vmname in bashrc for quick sshing into each machine . After executing, can ssh into a machine by using an alias e.g sshbis0
6. ssh-key-transfer.sh: Transfers your sshkey to each VM.
