IFS=$'\n'
set -f

# Harcoded username/hostFiles here. 
username='shayan'
hostFile='../azure-conf/hosts_diffDC'
keyPath=~/.ssh/id_rsa.pub

for line in $(cat $hostFile);do

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	echo "Transferring ssh key to $tname"

	cat $keyPath | ssh $username@$tname 'cat >> .ssh/authorized_keys'

done

exit