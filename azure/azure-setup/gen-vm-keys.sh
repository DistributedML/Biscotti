#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters (expecting 2):"
    echo "[username, ipfile]"
    exit
fi

username=$1
ipfile=$2

ipfilepath='../azure-conf/'

hostfile=$ipfilepath$ipfile

for line in $(cat $hostfile);do

	ipaddr=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	ssh -t $username@$ipaddr '	
		ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa
	'
done

exit



