azuredeployScript="$GOPATH/src/Biscotti/azure/azure-run"
logFiles="$GOPATH/src/Biscotti/DistSys/LogFiles"
ipFileName='hosts_diffDC'
eval_path="$PWD"
utilDir=$GOPATH/src/Biscotti/azure/azure-util

filename="poison_0.50_70.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.50 -rs=true -ns=70'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path

filename="poison_0.45_70.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.45 -rs=true -ns=70'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path

filename="poison_0.40_70.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.40 -rs=true -ns=70'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path

filename="poison_0.30_70.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.30 -rs=true -ns=70'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path

filename="poison_0.10_70.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.10 -rs=true -ns=70'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path

filename="poison_0.10_40.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.10 -rs=true -ns=40'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path

filename="poison_0.30_40.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.30 -rs=true -ns=40'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path

filename="poison_0.40_40.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.40 -rs=true -ns=40'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path

filename="poison_0.45_40.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.45 -rs=true -ns=40'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path

filename="poison_0.50_40.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.50 -rs=true -ns=40'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path

filename="poison_0.10_20.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.10 -rs=true -ns=20'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path

filename="poison_0.30_20.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.30 -rs=true -ns=20'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename

filename="poison_0.40_20.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.40 -rs=true -ns=20'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path

filename="poison_0.45_20.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.45 -rs=true -ns=20'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path

filename="poison_0.50_20.log"
cd $azuredeployScript
timeout 8000 bash runBiscotti.sh 5 100 $ipFileName mnist '-po=0.40 -rs=true -ns=40'
cd $utilDir
bash killall.sh $ipFileName DistSys
bash get-all-LogFiles.sh shayan $logFiles $ipFileName DistSys
cd $logFiles
mv log_1_100.log $eval_path/$filename
cd $eval_path

