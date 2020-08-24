#!/bin/bash -x

if [ "$#" -ne 8 ]; then
    echo "Illegal number of parameters (expecting 8):"
    echo "[r-group, vm-name, image-name, user-name, password vmlocation, vmsize, sshpath]"
    exit
fi

rgroup=$1    # resource group in which to find the image/create VM
vm=$2        # the name of the newly created VM
imagename=$3 # imagename from which to create the VM
user=$4      # username whose cmd line password will be changed
pws=$5       # password to set the user's password to
location=$6
vmtype=$7
sshkeypath=$8

# Create the VM
az vm create --resource-group $rgroup --name $vm --image $imagename \
   --admin-username $user --location $location --size $vmtype \
   --vnet-name ${location}VNET \
   --subnet Subnet1 \
   --ssh-key-value  $sshkeypath


# # Reset the password for the user in the vm
az vm user update --resource-group $rgroup --name $vm \
     --username $user --password $pws

# # Set the public IP address to static
az network public-ip update --name ${vm}PublicIp --resource-group $rgroup --allocation-method Static

# # Add rules for to allow machines to communicate with each other
bash add-rules-to-sg.sh $rgroup ${vm}NSG

# # Show details about the newly created VM
az vm show \
   --resource-group $rgroup \
   --name $vm \
   --show-details \
   -o table
