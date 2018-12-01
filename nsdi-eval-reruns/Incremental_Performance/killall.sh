#!/bin/bash

IFS=$'\n'
set -f

for line in $(cat tempHosts);do

	# skip the first line
	# if [ $cnt -eq 0 ]; then
	# 	cnt=$((cnt + 1))
	# 	continue
	# fi

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	if [[ "$tname" == "198.162.52.157" || "$tname" == "198.162.52.33" ]]; then
		username="clement"
	else
		username="cfung"
	fi

	echo "killing" $tname
	ssh $username@$tname 'pkill DistSys'

done

exit
