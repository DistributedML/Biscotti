username="cfung"
tname=198.162.52.57

cd ../DistSys
go install
scp -r $GOPATH/bin/DistSys $username@$tname:/home/$username/gopath/bin