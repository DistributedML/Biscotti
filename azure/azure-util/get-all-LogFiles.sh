

if [ "$#" -ne 3 ]; then

    echo "Illegal number of parameters"
    echo "usage:"
    echo "[username, destPath, hostFile]"
    exit

fi

username=$1
destPath=$2
ipfile=$3

ipfilepath='../azure-conf/'

hostFile=$ipfilepath$ipfile



for line in $(cat $hostFile);do

	ipaddr=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	scp $username@$ipaddr:/home/$username/gopath/src/Biscotti/DistSys/LogFiles/*.log $destPath 
	break

done
exit
