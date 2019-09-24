// Package key creates asymmetric key pairs.
package key

import (
	"crypto/cipher"

	"github.com/dedis/kyber"
)

// Generator is a type that needs to implement a special case in order
// to correctly choose a key.
type Generator interface {
	NewKey(random cipher.Stream) kyber.Scalar
}

// Suite represents the list of functionalities needed by this package.
type Suite interface {
	kyber.Group
	kyber.Random
}

// Pair represents a public/private keypair together with the
// ciphersuite the key was generated from.
type Pair struct {
	Public  kyber.Point  // Public key
	Private kyber.Scalar // Private key
	Hiding  kyber.Hiding // Hidden encoding of the public key
}

// NewKeyPair directly creates a secret/public key pair
func NewKeyPair(suite Suite) *Pair {
	kp := new(Pair)
	kp.Gen(suite)
	return kp
}

// NewHidingKeyPair creates a secret/public key pair and makes sure the
// the public key is hiding-encodable.
func NewHidingKeyPair(suite Suite) *Pair {
	kp := new(Pair)
	kp.GenHiding(suite)
	return kp
}

// Gen creates a fresh public/private keypair with the given
// ciphersuite, using a given source of cryptographic randomness. If
// suite implements key.Generator, then suite.NewKey is called
// to generate the private key, otherwise the normal technique
// of choosing a random scalar from the group is used.
func (p *Pair) Gen(suite Suite) {
	random := suite.RandomStream()
	if g, ok := suite.(Generator); ok {
		p.Private = g.NewKey(random)
	} else {
		p.Private = suite.Scalar().Pick(random)
	}
	p.Public = suite.Point().Mul(p.Private, nil)
}

// GenHiding will generate key pairs repeatedly until one is found where the
// public key has the property that it can be hidden.
func (p *Pair) GenHiding(suite Suite) {
	rand := suite.RandomStream()
	p.Gen(suite)
	Xh := p.Public.(kyber.Hiding)
	for {
		Xb := Xh.HideEncode(rand) // try to encode as uniform blob
		if Xb != nil {
			p.Hiding = Xh
			return // success
		}
		p.Gen(suite)
		Xh = p.Public.(kyber.Hiding)
	}
}
