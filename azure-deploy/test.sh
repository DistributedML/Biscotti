startingIndex=$1
nodesToRun=$2

for (( index = startingIndex ; index < startingIndex + nodesToRun; index++ )); do
	
	# thisLogFile=test1_$index\_$totalnodes.log
	# thatLogFile=log_$index\_$totalnodes.log
	
	# let thisPort=8000+$index

	# sudo timeout 120 $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard -f=peersfile.txt -a=$myAddress -p=$thisPort -pa=$myAddress > ./LogFiles/$thisLogFile 2> ./LogFiles/$thatLogFile &
	

	# sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard -f=peersfile.txt -a=$myAddress -p=$thisPort -pa=$myAddress > ./LogFiles/$thisLogFile 2> ./LogFiles/$thatLogFile &
	# sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > $thisLogFile 2> outLog.log &
	
	echo $index

	if [ $index -eq 0 ] 
	then			
		echo "Sleeping. Allowing node zero to be up and running"
		sleep 5			
	fi

done