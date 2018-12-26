rgroup=$1    # resource group in which to find the image/create VM
location=$2	 # same or diff location

if [[ "$location" = "same" ]]; then
	
	bash create-vnets-same.sh $rgroup	

elif [[ "$location" = "diff" ]]; then

	bash create-vnets-diff.sh $rgroup

else
	
	echo "loc should be same or diff"
	exit

fi