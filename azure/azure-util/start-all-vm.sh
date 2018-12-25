#!/bin/bash -x

rgroup='biscotti'
let numVMs=10
vmprefix='bis'

for (( i = 0; i < numVMs; i++ )); do

	vmname=$vmprefix$i

	bash start-vm.sh $rgroup $vmname
done
