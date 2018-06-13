#!/bin/bash

# Set needed path variables
export GOPATH=$PWD
export GOBIN=$PWD/bin

# Install dependencies in new GOPATH
go get github.com/DistributedClocks/GoVector/govec
go get github.com/kniren/gota/dataframe
go get github.com/sbinet/go-python
go get -u gonum.org/v1/gonum/mat
go get github.com/coniks-sys/coniks-go/crypto/vrf

# Build it
go install
