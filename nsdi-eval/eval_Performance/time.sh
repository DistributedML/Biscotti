start=`date +%s`

python plot.py
python plot.py
python plot.py
python plot.py
python plot.py
python plot.py
python plot.py

end=`date +%s`

runtime=$((end-start))
let numberOfNodes=10

thisLine=$runtime,$numberOfNodes

echo $thisLine