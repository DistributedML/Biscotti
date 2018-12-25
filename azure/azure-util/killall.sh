#!/bin/bash

IFS=$'\n'
set -f

# Harcoded username/hostFiles here. Kills Biscotti processes on every VM in host file
username='shayan'
hostFile='../azure-conf/hosts_diffDC'

for line in $(cat $hostFile);do

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	echo "killing" $tname
	ssh $username@$tname 'pkill DistSys'

done

exit


#Deprecated code
#####################################################################

# skip the first line
# if [ $cnt -eq 0 ]; then
# 	cnt=$((cnt + 1))
# 	continue
# fi

# if [[ "$tname" == "198.162.52.157" || "$tname" == "198.162.52.33" ]]; then
# 	username="clement"
# else
# 	username="cfung"
# fi
####################################################################