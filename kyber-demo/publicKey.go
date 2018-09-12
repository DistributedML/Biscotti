package main

import (
	
	// "fmt"
	// "strconv"
	// "strings"
	"github.com/dedis/kyber"
	"github.com/dedis/kyber/pairing/bn256"
	"github.com/dedis/kyber/util/random"

	// "encoding/binary"
	// "bytes"
)

// Update - data object representing a single update
type PublicKey struct {
	
	PKG1 []kyber.Point
	PKG2 []kyber.Point

}

// generates a public key based on a random seed for commitments of the updates

func (pkey *PublicKey) GenerateKey(numberOfDimensions int) {

	suite := bn256.NewSuite()

	// seed := random.New()

	pkey.PKG1 = make([]kyber.Point , numberOfDimensions)
	pkey.PKG2 = make([]kyber.Point , numberOfDimensions)

	// generate secret key
	privateKey := suite.G1().Scalar().SetInt64(int64(2))
	// fmt.Println("Private Key:" + privateKey.String()+ "\n")	
	
	// getting the generator for each group

	generatorG1  := suite.G1().Point().Mul(suite.G1().Scalar().One(),nil)
	generatorG2  := suite.G2().Point().Mul(suite.G2().Scalar().One(),nil)	
	
	// fmt.Println("Generator Group G1:" + generatorG1.String()+ "\n")
	// fmt.Println("Generator Group G2:" + generatorG2.String()+ "\n")

	previousKeyPartG1 := generatorG1
	previousKeyPartG2 := generatorG2


	for i := 0; i < (numberOfDimensions); i++ {
		
		pkey.PKG1[i] = previousKeyPartG1
		pkey.PKG2[i] = previousKeyPartG2
	
		previousKeyPartG1 = suite.G1().Point().Mul(privateKey,previousKeyPartG1)	
		previousKeyPartG2 = suite.G2().Point().Mul(privateKey,previousKeyPartG2)		
	
	}

}

func (pkey *PublicKey) GetGeneratorG1() kyber.Point{

	return pkey.PKG1[0]

}

func (pkey *PublicKey) GetGeneratorG2() kyber.Point{

	return pkey.PKG2[0]

}

func (pkey *PublicKey) GetFirstPKG2() kyber.Point{

	return pkey.PKG2[1]

}

func (pkey *PublicKey) GenerateClientKey() kyber.Scalar {

		suite := bn256.NewSuite()
		seed := random.New()

		pkey.PKG1 = make([]kyber.Point , 1)
		pkey.PKG2 = make([]kyber.Point , 1)

		privateKey := suite.G1().Scalar().Pick(seed)

		generatorG1  := suite.G1().Point().Mul(suite.G1().Scalar().One(),nil)
		generatorG2  := suite.G2().Point().Mul(suite.G2().Scalar().One(),nil)	

		pkey.PKG1[0] = suite.G1().Point().Mul(privateKey,generatorG1)
		pkey.PKG2[0] = suite.G2().Point().Mul(privateKey,generatorG2)

		return privateKey

}

func (pkey *PublicKey) SetG1Key(key kyber.Point) {

	pkey.PKG1 = make([]kyber.Point , 1)
	pkey.PKG1[0] = key

}
