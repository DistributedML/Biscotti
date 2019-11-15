#!/bin/bash

IFS=$'\n'
set -f

# Harcoded username/hostFiles here. Kills Biscotti processes on every VM in host file
username='shayan'
fileName=$1
program=$2
hostFile="../azure-conf/$fileName"


for line in $(cat $hostFile);do

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	echo "killing" $tname
	ssh $username@$tname 'pkill '$program

done

exit
