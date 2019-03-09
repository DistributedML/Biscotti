if [ "$#" -ne 4 ]; then

    echo "Illegal number of parameters"
    echo "usage:"
    echo "[username, destPath, hostFile, DistSys/FedSys]"
    exit

fi

username=$1
destPath=$2
ipfile=$3
bisOrFed=$4

ipfilepath='../azure-conf/'

hostFile=$ipfilepath$ipfile

for line in $(cat $hostFile);do

	ipaddr=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	scp $username@$ipaddr:/home/$username/gopath/src/Biscotti/$bisOrFed/LogFiles/*.log $destPath

done
exit
