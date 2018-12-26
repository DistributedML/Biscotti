if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "usage:"
    echo "[rgName]"
    exit
fi

rgroup=$1    # resource group in which to find the image/create VM

location='westus'

az network vnet create \
  --name ${location}VNET \
  --resource-group $rgroup \
  --location $location \
  --address-prefixes 10.0.0.0/8 \
  --subnet-name Subnet1 \
  --subnet-prefix 10.0.0.0/8