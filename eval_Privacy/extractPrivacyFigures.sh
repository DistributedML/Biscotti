
for file in ../DistSys/LogFiles/*.log
do
    if echo $file | grep "test1_"; then
    	le count = 0
		while IFS= read -r line
		do
		  echo "$line"
		done < "$file"
    fi
done