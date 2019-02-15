azuredeployScript="$GOPATH/src/Biscotti/azure/azure-run"
logFiles="$GOPATH/src/Biscotti/DistSys/LogFiles"

filename="epsilon_05.log"
cd $azuredeployScript
bash runBiscotti.sh 5 100 hosts_sameDC mnist '-po=0.3 -ns=70 -ep=0.5'
cd $logFiles
mv log_0_100.log ../../eval_noise_krum/$filename

# filename="epsilon_01.log"
# cd $azuredeployScript
# bash runBiscotti.sh 5 100 hosts_sameDC mnist '-po=0.3 -ns=70 -ep=1.0'
# cd $logFiles
# mv log_0_100.log ../../eval_poison_nsamples/$filename

# filename="epsilon_001.log"
# cd $azuredeployScript
# bash runBiscotti.sh 5 100 hosts_sameDC mnist '-po=0.3 -ns=70 -ep=1.0'
# cd $logFiles
# mv log_0_100.log ../../eval_poison_nsamples/$filename

# filename="epsilon_0001.log"
# cd $azuredeployScript
# bash runBiscotti.sh 5 100 hosts_sameDC mnist '-po=0.3 -ns=70 -ep=1.0'
# cd $logFiles
# mv log_0_100.log ../../eval_poison_nsamples/$filename

# filename="epsilon_5.log"
# cd $azuredeployScript
# bash runBiscotti.sh 5 100 hosts_sameDC mnist '-po=0.3 -ns=70 -ep=1.0'
# cd $logFiles
# mv log_0_100.log ../../eval_poison_nsamples/$filename