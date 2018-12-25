#!/bin/bash

myuname=$1
myip=$2
pass=$3

vmuname='shayan'
hostFile='../azure-conf/hosts_diffDC'

for line in $(cat $hostFile);do

	echo $pass

	vmip=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	ssh -t $vmuname@$vmip "
		sudo apt-get install sshpass
		sshpass -p '${pass}' ssh-copy-id -f -i ~/.ssh/id_rsa.pub -o StrictHostKeyChecking=no ${myuname}@${myip}
	"
	
done

exit