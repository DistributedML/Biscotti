if [ "$#" -ne 1 ]; then

    echo "Illegal number of parameters"
    echo "usage:"
    echo "[rgName]"
    exit

fi

rgroup=$1    # resource group in which to find the image/create VM

echo $rgroup

locations=('westus' 'eastus' 'centralindia' 'japaneast' 'australiaeast' 'westeurope') #locations to deploy

# let numlocations=${#locations[@]}
let numlocations=6

for (( i = 0; i < numlocations; i++ )); do

  vmlocation=${locations[i]}

  echo "Creating vnet at ${vmlocation}"

  az network vnet create \
    --name ${vmlocation}VNET \
    --resource-group $rgroup \
    --location ${vmlocation} \
    --address-prefixes 10.0.${i}.0/24 \
    --subnet-name Subnet1 \
    --subnet-prefix 10.0.${i}.0/24

done

# For every vnet created, establish peering with other vnets
for (( i = 0; i < numlocations; i++ )); do

  vmlocation1=${locations[i]}

  # Get the id of first VNET.
  vNet1Id=$(az network vnet show \
    --resource-group ${rgroup} \
    --name ${vmlocation1}VNET \
    --query id --out tsv)

  echo $vNet1Id

  for (( j = 0; j < numlocations; j++ )); do

      if [[ "i" -eq "j" ]]; then
        continue
      fi

      vmlocation2=${locations[j]}

      # Get the id of second VNET.
      vNet2Id=$(az network vnet show \
        --resource-group $rgroup \
        --name ${vmlocation2}VNET \
        --query id --out tsv)

      az network vnet peering create \
        --name ${vmlocation1}VNET-${vmlocation2}VNET \
        --resource-group $rgroup \
        --vnet-name  ${vmlocation1}VNET\
        --remote-vnet-id $vNet2Id \
        --allow-vnet-access

   done



done 