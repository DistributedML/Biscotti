#!/bin/bash -x

hostfile='../azure-conf/hosts_diffDC'
rgroup='biscotti'

for ip in $(cat $hostfile);do

	bash start-vm.sh $rgroup $ip

done