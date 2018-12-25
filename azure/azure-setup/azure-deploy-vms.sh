numVMs=$1    # Number of vms to deploy
rgroup=$2    # resource group in which to find the image/create VM
user=$3      # username whose cmd line password will be changed
pws=$4      # password to set the user's password to

sshPath='/home/shayan/.ssh/id_rsa.pub' #location of ssh-keys on ur machine
imagename='UBUNTULTS' #OS
vmprefix='bis' #each vmname is vmprefix+vmid
locations=('westus' 'eastus' 'centralus' \
			'centralindia' 'southeastasia' 'japaneast' 'canadacentral' \
			'australiaeast' 'northeurope' 'westeurope') #locations to deploy 

#Find number of locations to deploy
numlocations=${#locations[@]}

#create security group with rules (deprecated. not needed)
# bash create-ns-group.sh $rgroup $sgname

#deploy the VMS in the listed locations
for (( i = 6; i < numVMs; i++ )); do

	vmname=$vmprefix$i
	locIdx=$(($i % $numlocations))
	echo $locIdx
	vmlocation=${locations[locIdx]}
	echo $vmname 
	echo $vmlocation
	bash create-vm.sh $rgroup $vmname $imagename $user $pws $vmlocation

done