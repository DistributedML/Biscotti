#!/bin/bash -x

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters (expecting 2):"
    echo "[r-group, sg-name]"
    exit
fi

rgroup=$1  # resource group in which to find the image/create VM
sgname=$2  # name of newly created security group

# # Create security group
az network nsg create --resource-group $rgroup --name $sgname

# # Allow machines to ssh into all vm's part of group
az network nsg rule create --name ssh_allow --resource-group $rgroup \
	--nsg-name $sgname --priority 1000 \
	--destination-address-prefixes \* --source-address-prefixes \* \
	--source-port-ranges \* --destination-port-ranges 22 \
	--direction Inbound 

# Allow inbound communication between members of the group on port range 8000-9000
az network nsg rule create --name Bis_All_In --resource-group $rgroup \
	--nsg-name $sgname --priority 1100 \
	--destination-address-prefixes \* --source-address-prefixes \* \
	--source-port-ranges \* --destination-port-ranges 8000-9000 \
	--direction Inbound 

# # Allow oubound communication between members of the group range 8000-9000
az network nsg rule create --name Bis_All_Out --resource-group $rgroup \
	--nsg-name $sgname --priority 1200 --destination-address-prefixes \* \
	--source-address-prefixes \*  --source-port-ranges \* \
	--destination-port-ranges 8000-9000 --direction Outbound