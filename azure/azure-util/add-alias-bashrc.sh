IFS=$'\n'
set -f

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "usage:"
    echo "[fileName, user]"
    exit
fi

fileName=$1
echo $fileName
username=$2
echo $username
hostFile="../azure-conf/$fileName"

let vmcount=0

for line in $(cat $hostFile);do

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	echo "Adding alias for bis$vmcount"

	echo "alias sshbis$vmcount='ssh $username@$tname'" >> ~/.bashrc

	let vmcount=$vmcount+1

done

exit
