
## Run the following biscotti runs with a different number of nodes
LOGFILE_PATH=$GOPATH/src/Biscotti/DistSys/LogFiles
AZURE_PATH=$GOPATH/src/Biscotti/azure/

NUM_RUNS=5
TEMP_LOG_STORE="biscotti"
RESULTS="results"
dataset="mnist"
ipFileName="hosts_diffDC"
thisDir=$PWD
utilDir=$GOPATH/src/Biscotti/azure/azure-util
runDir=$GOPATH/src/Biscotti/azure/azure-run

mkdir $TEMP_LOG_STORE
mkdir $RESULTS

# # Test for 40

mkdir $TEMP_LOG_STORE/40

for (( i = 4; i < $NUM_RUNS; i++ )); do

	cd $runDir
	timeout 6000 bash runBiscotti.sh 2 40 $ipFileName $dataset
	cd $utilDir
	bash killall.sh hosts_diffDC DistSys
	bash get-all-LogFiles.sh shayan $LOGFILE_PATH hosts_diffDC DistSys	
	cp -a $LOGFILE_PATH $thisDir/$TEMP_LOG_STORE/40/
	cd $thisDir
	rm -rf $TEMP_LOG_STORE/40/$i
	mv $TEMP_LOG_STORE/40/LogFiles/ $TEMP_LOG_STORE/40/$i
	
done

# Test for 80
mkdir $TEMP_LOG_STORE/80

for (( i = 4; i < $NUM_RUNS; i++ )); do

	cd $runDir
	timeout 6000 bash runBiscotti.sh 4 80 $ipFileName $dataset
	cd $utilDir
	bash killall.sh hosts_diffDC DistSys
	bash get-all-LogFiles.sh shayan $LOGFILE_PATH hosts_diffDC DistSys	
	cp -a $LOGFILE_PATH $thisDir/$TEMP_LOG_STORE/80/
	cd $thisDir
	rm -rf $TEMP_LOG_STORE/80/$i
	mv $TEMP_LOG_STORE/80/LogFiles/ $TEMP_LOG_STORE/80/$i
	
done

# Test for 100

mkdir $TEMP_LOG_STORE/100

for (( i = 4; i < $NUM_RUNS; i++ )); do

	cd $runDir
	timeout 6000 bash runBiscotti.sh 5 100 $ipFileName $dataset
	cd $utilDir
	bash killall.sh hosts_diffDC DistSys
	bash get-all-LogFiles.sh shayan $LOGFILE_PATH hosts_diffDC DistSys	
	cp -a $LOGFILE_PATH $thisDir/$TEMP_LOG_STORE/100/
	cd $thisDir
	rm -rf $TEMP_LOG_STORE/100/$i
	mv $TEMP_LOG_STORE/100/LogFiles/ $TEMP_LOG_STORE/100/$i
	
done

# Run for 60
mkdir $TEMP_LOG_STORE/60

for (( i = 4; i < $NUM_RUNS; i++ )); do

	cd $runDir
	timeout 6000 bash runBiscotti.sh 3 60 $ipFileName $dataset
	cd $utilDir
	bash killall.sh hosts_diffDC DistSys
	bash get-all-LogFiles.sh shayan $LOGFILE_PATH hosts_diffDC DistSys	
	cp -a $LOGFILE_PATH $thisDir/$TEMP_LOG_STORE/60/
	cd $thisDir
	rm -rf $TEMP_LOG_STORE/60/$i
	mv $TEMP_LOG_STORE/60/LogFiles/ $TEMP_LOG_STORE/60/$i
	
done

# To plot
# python parseLogs.py $TEMP_LOG_STORE $RESULTS 3
# python plot_incremental.py $TEMP_LOG_STORE $RESULTS 3