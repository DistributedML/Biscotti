azuredeployScript="$GOPATH/src/Biscotti/azure/azure-run"
logFiles="$GOPATH/src/Biscotti/DistSys/LogFiles"
ipFileName='hosts_diffDC'
eval_path="$PWD"
utilDir=$GOPATH/src/Biscotti/azure/azure-util

filename="epsilon_05.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.3 -ns=70 -ep=0.5'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path

filename="epsilon_01.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.3 -ns=70 -ep=0.1'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path

filename="epsilon_001.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.3 -ns=70 -ep=0.01'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path

filename="epsilon_1.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.3 -ns=70 -ep=1'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path

filename="epsilon_2.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.3 -ns=70 -ep=2'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path

filename="epsilon_5.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.3 -ns=70 -ep=5'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path