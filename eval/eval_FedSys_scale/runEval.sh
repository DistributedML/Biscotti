
## Run the following biscotti runs with a different number of nodes
DIST_LOG_PATH=$GOPATH/src/Biscotti/DistSys/LogFiles
FED_LOG_PATH=$GOPATH/src/Biscotti/FedSys/LogFiles
AZURE_PATH=$GOPATH/src/Biscotti/azure/

# Hardcoded constants
NUM_RUNS=3
NUM_NODES="100"
NODES_EACH_VM="5"
DIST_TEMP_LOG_STORE="biscotti_agg_26"
FED_TEMP_LOG_STORE="fed"
DIST_LOG_PATH="$GOPATH/src/Biscotti/DistSys/LogFiles"
FED_LOG_PATH="$GOPATH/src/Biscotti/FedSys/LogFiles"
FED_TEMP_LOG_STORE="fed"		
RESULTS="results"
dataset="mnist"
ipFileName="hosts_diffDC"

thisDir="$PWD"
utilDir=$GOPATH/src/Biscotti/azure/azure-util
runDir=$GOPATH/src/Biscotti/azure/azure-run

mkdir $DIST_TEMP_LOG_STORE
mkdir $FED_TEMP_LOG_STORE
mkdir $RESULTS

# Federated Learning
for (( i = 0; i < $NUM_RUNS; i++ )); do

	cd $runDir
	timeout 5000 bash runFedSys.sh $NODES_EACH_VM $NUM_NODES $ipFileName $dataset
	cd $utilDir
	bash killall.sh $ipFileName FedSys
	bash get-all-LogFiles.sh shayan $FED_LOG_PATH $ipFileName FedSys
	cp -a $FED_LOG_PATH $thisDir/$FED_TEMP_LOG_STORE		
	cd $thisDir
	rm -rf $FED_TEMP_LOG_STORE/$i
	mv $FED_TEMP_LOG_STORE/LogFiles/ $FED_TEMP_LOG_STORE/$i

done

# # Biscotti
for (( i = 0; i < $NUM_RUNS; i++ )); do

	cd $runDir
	timeout 16000 bash runBiscotti.sh $NODES_EACH_VM $NUM_NODES $ipFileName $dataset '-na=26 '
	cd $utilDir
	bash killall.sh $ipFileName DistSys
	bash get-all-LogFiles.sh shayan $DIST_LOG_PATH $ipFileName DistSys
	cp -a $DIST_LOG_PATH $thisDir/$DIST_TEMP_LOG_STORE		
	cd $thisDir
	rm -rf $DIST_TEMP_LOG_STORE/$i
	mv $DIST_TEMP_LOG_STORE/LogFiles/ $DIST_TEMP_LOG_STORE/$i
	
done
