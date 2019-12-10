azuredeployScript="$GOPATH/src/Biscotti/azure/azure-run"
logFiles="$GOPATH/src/Biscotti/DistSys/LogFiles"
ipFileName='hosts_diffDC'
eval_path="$PWD"
utilDir=$GOPATH/src/Biscotti/azure/azure-util

folder="Biscotti_100_poison_mnist"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.30  -ns=70 -ep=1.0'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_0_100.log $eval_path/$folder
cd $eval_path
