#!/bin/bash
if [ "$#" -ne 4 ]; then

    echo "Illegal number of parameters"
    echo "usage:"
    echo "[localuname, localip, localpass, ipfile]"
    exit

fi

myuname=$1
myip=$2
pass=$3
ipfile=$4

vmuname='shayan'
hostFile="../azure-conf/${ipfile}"

for line in $(cat $hostFile);do

	echo $pass

	vmip=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	ssh -t $vmuname@$vmip "
		sudo apt-get install sshpass
		sshpass -p '${pass}' ssh-copy-id -f -i ~/.ssh/id_rsa.pub -o StrictHostKeyChecking=no ${myuname}@${myip}
	"
	
done

exit
