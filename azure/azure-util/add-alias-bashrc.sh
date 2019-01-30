IFS=$'\n'
set -f

# Harcoded username/hostFiles here. 
fileName=$1
echo $fileName
username='matheus'
hostFile="../azure-conf/$fileName"

let vmcount=0

for line in $(cat $hostFile);do

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	echo "Adding alias for bis$vmcount"

	echo "alias sshbis$vmcount='ssh $username@$tname'" >> ~/.bashrc

	let vmcount=$vmcount+1

done

exit
